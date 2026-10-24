"""Export Nemotron-Labs-Diffusion-3B as one block-KV-cache graph, fp16 then q4f16.

Nemotron decodes like kohra's BD3LM KV path, with two differences (modeling_nemotron_labs_diffusion.py,
`generate`): the commit forward that writes a finished block into the cache is causal, and its last
logit seeds the first token of the next block. One graph serves both passes because the attention
mask is an input:

Inputs:  input_ids [1, L], position_ids [1, L] (absolute),
         attention_mask [1, 1, L, past + L] fp32, additive (0 = attend; large negative = blocked),
         past_key_{i} / past_value_{i} [1, 8, past, 128] for each of the 26 layers.
Outputs: logits [1, L, V] fp32, present_key_{i} / present_value_{i} [1, 8, past + L, 128].

Denoise steps feed an all-zero mask (the block sees the whole cache and itself) and ignore the
presents; the commit pass feeds a causal mask and keeps them. The upstream attention
(Ministral3Attention) builds exactly these two cases: no mask when `diffusion_lm`, a causal mask
otherwise.

Memory: the checkpoint is bf16 (7.66 GB). fp32 would need ~15 GB plus the export's copy, which does
not fit beside other work on a 24 GB Mac, so the graph is exported in fp16 and q4 is quantised
from the fp16 graph. External data is split into files of at most 2 GB, the layout LocalMind already
loads in ORT-web (onnx-community/gemma-4-E4B-it-ONNX).

Stages, each its own process so peak memory stays ~16 GB:
  ref     torch bf16 reference logits on fixed inputs (both mask modes) -> ref.npz
  export  torch fp16 -> model_kv_fp16.onnx (+ split external data)
  check   ORT CPU on a graph vs ref.npz (argmax agreement, max |Δlogit|)
  q4      MatMulNBits RTN 4-bit, symmetric, block 32, on the fp16 graph -> model_kv_q4f16.onnx

  .venv-nemotron/bin/python scripts/export_nemotron_kv.py ref|export|check [graph]|q4
"""

import os
import sys

import numpy as np
import torch

SRC = os.environ.get("NEMOTRON_SRC", "/Volumes/Models2/Models/diffusion/nemotron-labs-diffusion-3b")
OUT_DIR = os.environ.get(
    "NEMOTRON_OUT", os.path.join(os.path.dirname(__file__), "..", "models", "nemotron-labs-diffusion-3b-onnx"))
N_LAYERS, KV_HEADS, HEAD_DIM, VOCAB, MASK_ID = 26, 8, 128, 131072, 100
SPLIT_BYTES = 2_000_000_000
NEG = -1e9


def load_model(dtype):
    import transformers
    m = transformers.AutoModel.from_pretrained(
        SRC, trust_remote_code=True, dtype=dtype, attn_implementation="eager").eval()
    # The encoder keeps its own config copy (config.json says sdpa), so set eager there too:
    # the exporter cannot convert SDPA with grouped-query attention.
    m.encoder.config._attn_implementation = "eager"
    for layer in m.encoder.layers:          # the mask input decides causal vs bidirectional
        layer.self_attn.diffusion_lm = False
        layer.self_attn.config._attn_implementation = "eager"
    return m


class BlockWithCache(torch.nn.Module):
    """Ministral3Model.forward with an explicit 4D additive mask, plus the diffusion head."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, position_ids, attention_mask, *past):
        from transformers.cache_utils import DynamicCache
        enc = self.model.encoder
        cache = DynamicCache()
        for i in range(N_LAYERS):
            cache.update(past[2 * i], past[2 * i + 1], i)
        h = enc.embed_tokens(input_ids)
        pe = enc.rotary_emb(h, position_ids=position_ids)
        mask = attention_mask.to(h.dtype)
        cache_position = position_ids[0]
        for layer in enc.layers[:N_LAYERS]:
            h = layer(h, attention_mask=mask, position_ids=position_ids, past_key_values=cache,
                      use_cache=True, cache_position=cache_position, position_embeddings=pe)
            if isinstance(h, tuple):
                h = h[0]
        h = enc.norm(h)
        logits = self.model.diffusion_head(h).float()
        flat = []
        for i in range(N_LAYERS):
            flat += [cache.layers[i].keys, cache.layers[i].values]
        return (logits, *flat)


def names():
    past, present = [], []
    for i in range(N_LAYERS):
        past += [f"past_key_{i}", f"past_value_{i}"]
        present += [f"present_key_{i}", f"present_value_{i}"]
    return past, present


def make_mask(L, past_len, causal):
    m = np.zeros((1, 1, L, past_len + L), dtype=np.float32)
    if causal:
        for q in range(L):
            m[0, 0, q, past_len + q + 1:] = NEG
    return m


def sample_inputs(past_len, causal, L=32, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(1000, 100000, size=(1, L), dtype=np.int64)
    if not causal:
        ids[0, L // 2:] = MASK_ID
    pos = np.arange(past_len, past_len + L, dtype=np.int64).reshape(1, L)
    past = [(rng.standard_normal((1, KV_HEADS, past_len, HEAD_DIM)) * 0.5).astype(np.float32)
            for _ in range(2 * N_LAYERS)]
    return ids, pos, make_mask(L, past_len, causal), past


CASES = {"denoise": dict(past_len=48, causal=False, seed=1), "commit": dict(past_len=48, causal=True, seed=2)}


def stage_ref():
    model = load_model(torch.bfloat16)
    w = BlockWithCache(model)
    out = {}
    with torch.no_grad():
        for name, c in CASES.items():
            ids, pos, mask, past = sample_inputs(c["past_len"], c["causal"], seed=c["seed"])
            res = w(torch.from_numpy(ids), torch.from_numpy(pos), torch.from_numpy(mask),
                    *[torch.from_numpy(p).to(torch.bfloat16) for p in past])
            out[f"{name}_logits"] = res[0].float().numpy()
            out[f"{name}_key0"] = res[1].float().numpy()
            print(f"[ref {name}] logits {tuple(res[0].shape)} argmax[:8]={res[0][0, :8].argmax(-1).tolist()}")
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(os.path.join(OUT_DIR, "ref.npz"), **out)


def split_external(path):
    """Rewrite `path` so its initializers live in <name>_data, <name>_data_1, … each ≤ SPLIT_BYTES."""
    import onnx
    from onnx.external_data_helper import load_external_data_for_model, set_external_data
    m = onnx.load(path, load_external_data=False)
    sources = {e.value for t in m.graph.initializer for e in t.external_data if e.key == "location"}
    load_external_data_for_model(m, os.path.dirname(path))
    base = os.path.basename(path)
    for f in sources:                       # the files just read (one per tensor, or one .data)
        os.remove(os.path.join(os.path.dirname(path), f))
    part, used = 0, 0
    for t in m.graph.initializer:
        n = len(t.raw_data)
        if n < 1024:
            continue
        if used and used + n > SPLIT_BYTES:
            part, used = part + 1, 0
        loc = base + "_data" + ("" if part == 0 else f"_{part}")
        set_external_data(t, location=loc, offset=used, length=n)
        used += n
    # write each tensor's bytes into its file, then drop raw_data from the proto
    handles = {}
    for t in m.graph.initializer:
        if t.data_location != onnx.TensorProto.EXTERNAL:
            continue
        info = {e.key: e.value for e in t.external_data}
        fp = os.path.join(os.path.dirname(path), info["location"])
        if fp not in handles:
            handles[fp] = open(fp, "wb")
        handles[fp].seek(int(info["offset"]))
        handles[fp].write(t.raw_data)
        t.ClearField("raw_data")
    for h in handles.values():
        h.close()
    onnx.save_model(m, path)
    for fp in sorted(handles):
        print(f"  {os.path.basename(fp)}: {os.path.getsize(fp) / 1e9:.2f} GB")


def stage_export():
    model = load_model(torch.bfloat16).to(torch.float16)
    w = BlockWithCache(model)
    past_names, present_names = names()
    ids, pos, mask, past = sample_inputs(64, causal=True)
    dyn = {"input_ids": {1: "block"}, "position_ids": {1: "block"}, "logits": {1: "block"},
           "attention_mask": {2: "block", 3: "total"}}
    for n in past_names:
        dyn[n] = {2: "past"}
    for n in present_names:
        dyn[n] = {2: "total"}
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "model_kv_fp16.onnx")
    with torch.no_grad():
        torch.onnx.export(
            w, (torch.from_numpy(ids), torch.from_numpy(pos), torch.from_numpy(mask),
                *[torch.from_numpy(p).half() for p in past]),
            path, input_names=["input_ids", "position_ids", "attention_mask", *past_names],
            output_names=["logits", *present_names], dynamic_axes=dyn, opset_version=17,
            do_constant_folding=True, dynamo=False, external_data=True)
    del model, w
    print(f"exported -> {path}")
    split_external(path)


def stage_check(graph):
    import onnxruntime as ort
    ref = np.load(os.path.join(OUT_DIR, "ref.npz"))
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(os.path.join(OUT_DIR, graph), so, providers=["CPUExecutionProvider"])
    types = {i.name: i.type for i in sess.get_inputs()}
    past_names, _ = names()
    for name, c in CASES.items():
        ids, pos, mask, past = sample_inputs(c["past_len"], c["causal"], seed=c["seed"])
        feeds = {"input_ids": ids, "position_ids": pos, "attention_mask": mask}
        for n, p in zip(past_names, past):
            feeds[n] = p.astype(np.float16) if "float16" in types[n] else p
        logits, key0 = sess.run(["logits", "present_key_0"], feeds)
        rl = ref[f"{name}_logits"]
        agree = float((logits.argmax(-1) == rl.argmax(-1)).mean())
        dl = float(np.abs(logits.astype(np.float32) - rl).max())
        dk = float(np.abs(key0.astype(np.float32) - ref[f"{name}_key0"]).max())
        print(f"[{graph} {name}] argmax agree={agree:.4f}  max|Δlogit|={dl:.3f}  "
              f"max|Δpresent_key_0|={dk:.4f}  present={key0.shape}")


def stage_q4():
    import onnx
    from onnxruntime.quantization import QuantFormat
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer, RTNWeightOnlyQuantConfig
    src = os.path.join(OUT_DIR, "model_kv_fp16.onnx")
    m = onnx.load(src)
    q = MatMulNBitsQuantizer(m, block_size=32, is_symmetric=True, quant_format=QuantFormat.QOperator,
                             algo_config=RTNWeightOnlyQuantConfig())
    q.process()
    del m
    path = os.path.join(OUT_DIR, "model_kv_q4f16.onnx")
    q.model.save_model_to_file(path, use_external_data_format=True)
    print(f"saved -> {path}")
    split_external(path)


if __name__ == "__main__":
    stage = sys.argv[1]
    if stage == "ref":
        stage_ref()
    elif stage == "export":
        stage_export()
    elif stage == "check":
        stage_check(sys.argv[2] if len(sys.argv) > 2 else "model_kv_fp16.onnx")
    elif stage == "q4":
        stage_q4()
    else:
        raise SystemExit(__doc__)
