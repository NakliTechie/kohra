"""Export BD3LM with a block KV cache: a graph that denoises one block against a cache.

Inputs:  input_ids [1, L] (the current block), position_ids [1, L] (absolute positions),
         past_key_{i} / past_value_{i} [1, 8, past, 128] for each of the 28 layers.
Outputs: logits [1, L, V] (the block only), present_key_{i} / present_value_{i}
         [1, 8, past + L, 128].

The block attends to the whole cache plus itself with no mask: BD3LM is block-causal, so
every cached key belongs to an earlier, finished block. scripts/kv_reference.py shows the
cached loop reproduces the cache-free one token for token (torch, greedy).

Stages (each checked for parity against torch on the same inputs):
  model_kv_fp32.onnx          plain export
  model_kv_fp32_fused.onnx    RMSNorm/RoPE fusion (model_type=qwen3), attention left decomposed
  model_kv_fp16_fused.onnx    fp16 weights and KV I/O; logits stay fp32
  model_kv_q4f16_rtn_sym.onnx (--q4) RTN 4-bit MatMulNBits on the fused fp32 graph, then fp16;
                              the same recipe as the shipped q4 graphs (optimize_onnx.py --q4)

  .venv/bin/python scripts/export_bd3lm_kv.py [--q4]
"""

import os
import sys

import numpy as np
import onnx
import onnxruntime as ort
import torch
import transformers
from transformers.cache_utils import DynamicCache

sys.path.insert(0, os.path.dirname(__file__))
from kv_reference import MODEL_ID, load_a2d  # noqa: E402

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "qwen3-0.6b-bd3lm-onnx")
N_LAYERS, KV_HEADS, HEAD_DIM = 28, 8, 128


class BlockWithCache(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, position_ids, *past):
        cache = DynamicCache()
        for i in range(N_LAYERS):
            cache.update(past[2 * i], past[2 * i + 1], i)
        L = input_ids.shape[1]
        total = past[0].shape[2] + L
        mask = torch.zeros(1, 1, L, total, dtype=past[0].dtype)   # attend to cache + block
        out = self.model(input_ids=input_ids, position_ids=position_ids, attention_mask=mask,
                         past_key_values=cache, use_cache=True)
        pkv = out.past_key_values
        flat = []
        for i in range(N_LAYERS):
            flat += [pkv.layers[i].keys, pkv.layers[i].values]
        return (out.logits, *flat)


def names():
    past, present = [], []
    for i in range(N_LAYERS):
        past += [f"past_key_{i}", f"past_value_{i}"]
        present += [f"present_key_{i}", f"present_value_{i}"]
    return past, present


def sample_inputs(mask_id, past_len, L=32, dtype=np.float32, seed=0):
    rng = np.random.default_rng(seed)
    ids = rng.integers(0, 151000, size=(1, L), dtype=np.int64)
    ids[0, L // 2:] = mask_id
    pos = np.arange(past_len, past_len + L, dtype=np.int64).reshape(1, L)
    past = [rng.standard_normal((1, KV_HEADS, past_len, HEAD_DIM)).astype(dtype) * 0.5
            for _ in range(2 * N_LAYERS)]
    return ids, pos, past


def check(path, ref, feeds_np, label, out_names):
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    feed_dtype = {i.name: i.type for i in sess.get_inputs()}
    feeds = {k: (v.astype(np.float16) if "float16" in feed_dtype[k] else v) for k, v in feeds_np.items()}
    got = sess.run(out_names[:3], feeds)
    logits = got[0].astype(np.float32)
    dl = float(np.abs(logits - ref[0]).max())
    am = float((logits.argmax(-1) == ref[0].argmax(-1)).mean())
    dk = float(np.abs(got[1].astype(np.float32) - ref[1]).max())
    print(f"[{label}] max|Δlogit|={dl:.4f}  argmax agree={am:.4f}  max|Δpresent_key_0|={dk:.4f}  "
          f"present shape={got[1].shape}")
    return am


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    load_a2d()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    mask_id = tok.mask_token_id
    model = transformers.AutoModelForMaskedLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager").eval()
    wrapper = BlockWithCache(model)
    past_names, present_names = names()
    out_names = ["logits", *present_names]

    fp32 = os.path.join(OUT_DIR, "model_kv_fp32.onnx")
    if not os.path.exists(fp32):
        ids, pos, past = sample_inputs(mask_id, past_len=64)
        dyn = {"input_ids": {1: "block"}, "position_ids": {1: "block"}, "logits": {1: "block"}}
        for n in past_names:
            dyn[n] = {2: "past"}
        for n in present_names:
            dyn[n] = {2: "total"}
        with torch.no_grad():
            torch.onnx.export(
                wrapper, (torch.from_numpy(ids), torch.from_numpy(pos), *map(torch.from_numpy, past)),
                fp32, input_names=["input_ids", "position_ids", *past_names], output_names=out_names,
                dynamic_axes=dyn, opset_version=17, do_constant_folding=True, dynamo=False)
        print(f"exported -> {fp32}")

    # parity reference: a different past length than the export example (dynamic axes)
    ids, pos, past = sample_inputs(mask_id, past_len=96, seed=1)
    with torch.no_grad():
        r = wrapper(torch.from_numpy(ids), torch.from_numpy(pos), *map(torch.from_numpy, past))
    ref = [r[0].numpy(), r[1].numpy()]
    feeds = {"input_ids": ids, "position_ids": pos, **dict(zip(past_names, past))}
    assert check(fp32, ref, feeds, "kv-fp32", out_names) > 0.99, "fp32 parity failed"

    from onnxruntime.transformers import optimizer
    from onnxruntime.transformers.fusion_options import FusionOptions
    from onnxruntime.transformers.onnx_model import OnnxModel

    fused = os.path.join(OUT_DIR, "model_kv_fp32_fused.onnx")
    if not os.path.exists(fused):
        opts = FusionOptions("qwen3")
        opts.enable_attention = False          # keep attention decomposed: GQA contrib ops are causal
        m = optimizer.optimize_model(fp32, model_type="qwen3", num_heads=16, hidden_size=1024,
                                     optimization_options=opts, opt_level=0)
        m.save_model_to_file(fused, use_external_data_format=True)
        print(f"saved -> {fused}")
    check(fused, ref, feeds, "kv-fp32-fused", out_names)

    fp16 = os.path.join(OUT_DIR, "model_kv_fp16_fused.onnx")
    if not os.path.exists(fp16):
        m = OnnxModel(onnx.load(fused))
        m.convert_float_to_float16(keep_io_types=["logits"])
        m.save_model_to_file(fp16, use_external_data_format=True)
        print(f"saved -> {fp16}")
    am = check(fp16, ref, feeds, "kv-fp16-fused", out_names)
    ops = sorted({n.op_type for n in onnx.load(fp16, load_external_data=False).graph.node})
    print("fp16 graph ops:", ", ".join(o for o in ops if o in
          {"SimplifiedLayerNormalization", "SkipSimplifiedLayerNormalization", "RotaryEmbedding",
           "GroupQueryAttention", "MultiHeadAttention", "Attention", "Softmax"}))
    print("PARITY OK" if am > 0.9 else "PARITY LOW")

    if "--q4" in sys.argv:
        from onnxruntime.quantization import QuantFormat
        from onnxruntime.quantization.matmul_nbits_quantizer import (
            MatMulNBitsQuantizer, RTNWeightOnlyQuantConfig)
        q4 = os.path.join(OUT_DIR, "model_kv_q4f16_rtn_sym.onnx")
        if not os.path.exists(q4):
            q = MatMulNBitsQuantizer(onnx.load(fused), block_size=32, is_symmetric=True,
                                     quant_format=QuantFormat.QOperator,
                                     algo_config=RTNWeightOnlyQuantConfig())
            q.process()
            qm = OnnxModel(q.model.model)
            qm.convert_float_to_float16(keep_io_types=["logits"])
            qm.save_model_to_file(q4, use_external_data_format=True)
            print(f"saved -> {q4} ({os.path.getsize(q4 + '.data') / 1e6:.0f} MB data)")
        # q4 on random inputs agrees less than fp16 (the shipped q4 graphs show the same);
        # the real check is generation: gencheck_bd3lm_kv.py --model <q4> and the browser.
        check(q4, ref, feeds, "kv-q4f16-rtn", out_names)


if __name__ == "__main__":
    main()
