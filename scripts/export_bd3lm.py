"""Export Tiny-A2D Qwen3-0.6B BD3LM (block diffusion) to ONNX for ORT-web.

BD3LM = block diffusion: same A2DQwen3 architecture as the MDLM checkpoint (so the
fuse + RTN-q4 pipeline in optimize_onnx.py applies directly), but a separately-trained
checkpoint with **block-causal attention** — query block q attends to key blocks <= q
(bidirectional within a block, causal across blocks). Stronger benchmarks than MDLM
(GSM8K 46.3 vs 29.3, HumanEval 46.3 vs 30.5).

The dllm sampler uses a block-KV cache for speed; for the browser we go cache-free:
a full forward each step with a block-causal **4D additive** attention_mask, denoising
block-by-block on a fixed canvas. So unlike MDLM (no mask), this graph takes a second
input: attention_mask [1,1,T,T] (0 = attend, large-negative = block). The model uses a
4D mask as-is (modeling_qwen3.py NEW-code path), so eager attention adds it directly.

Usage:
  .venv/bin/python scripts/export_bd3lm.py          # fp32 export + parity
  .venv/bin/python scripts/export_bd3lm.py --fp16   # fp16 export + parity
"""

import argparse
import os
import sys

import numpy as np
import torch

import dllm  # noqa: F401 (registers a2d-qwen3 with Auto* classes)
import transformers

MODEL_ID = "dllm-hub/Qwen3-0.6B-diffusion-bd3lm-v0.1"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "qwen3-0.6b-bd3lm-onnx")
BLOCK_SIZE = 32
NEG = -1e9   # additive "block" value (eager attention adds this before softmax)


def block_causal_mask(T, block_size, dtype):
    """[1,1,T,T] additive block-causal mask: allow[i,j] = (block(j) <= block(i)).

    Mirrors dllm bd3lm `_prepare_for_sampling` (base_mask = bid_k <= bid_q)."""
    pos = torch.arange(T)
    bid = torch.div(pos, block_size, rounding_mode="floor")
    allow = bid.view(1, T) <= bid.view(T, 1)            # [T(query), T(key)]
    m = torch.where(allow, torch.zeros(()), torch.full((), NEG))
    return m.view(1, 1, T, T).to(dtype)


class LogitsMasked(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):   # attention_mask: [1,1,T,T] additive
        return self.model(input_ids=input_ids, attention_mask=attention_mask,
                          use_cache=False).logits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp16", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    dtype = torch.float16 if args.fp16 else torch.float32
    path = os.path.join(OUT_DIR, "model_fp16.onnx" if args.fp16 else "model_fp32.onnx")

    tok = transformers.AutoTokenizer.from_pretrained(MODEL_ID)
    mask_id = tok.mask_token_id
    print(f"mask_token_id={mask_id}  block_size={BLOCK_SIZE}  dtype={dtype}")

    model = transformers.AutoModelForMaskedLM.from_pretrained(
        MODEL_ID, dtype=dtype, attn_implementation="eager"
    ).eval()
    wrapper = LogitsMasked(model)

    if os.path.exists(path):
        print(f"{path} exists, skipping export")
    else:
        T0 = 48
        ex_ids = torch.randint(0, 151000, (1, T0), dtype=torch.long)
        ex_ids[0, 20:] = mask_id
        ex_mask = block_causal_mask(T0, BLOCK_SIZE, dtype)
        torch.onnx.export(
            wrapper, (ex_ids, ex_mask), path,
            input_names=["input_ids", "attention_mask"], output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 2: "sequence", 3: "sequence"},
                "logits": {0: "batch", 1: "sequence"},
            },
            opset_version=17, do_constant_folding=True, dynamo=False,
        )
        print(f"exported -> {path} ({os.path.getsize(path)/1e6:.0f} MB + external data)")

    # parity: torch vs ONNX on the SAME (input_ids, block-causal mask)
    import onnxruntime as ort
    T = 64
    torch.manual_seed(0)
    ids = torch.randint(0, 151000, (1, T), dtype=torch.long)
    ids[0, 40:] = mask_id
    m = block_causal_mask(T, BLOCK_SIZE, dtype)
    with torch.no_grad():
        ref = wrapper(ids, m).float().numpy()
    so = ort.SessionOptions()
    if args.fp16:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    got = sess.run(["logits"], {"input_ids": ids.numpy(),
                                "attention_mask": m.numpy()})[0].astype(np.float32)  # mask in model dtype
    masked = float((ref[0, 40:].argmax(-1) == got[0, 40:].argmax(-1)).mean())
    print(f"[bd3lm {'fp16' if args.fp16 else 'fp32'}] max|Δ|={np.max(np.abs(ref-got)):.4f}  "
          f"masked-argmax={masked:.4f}")
    if masked < (0.98 if not args.fp16 else 0.9):
        print("[bd3lm] PARITY FAIL"); sys.exit(1)
    print("[bd3lm] parity OK")
    print("NEXT: optimize_onnx.py with KOHRA_MODEL_DIR=models/qwen3-0.6b-bd3lm-onnx "
          "(fuse + RTN-q4); then a block-causal sampler in kohra.js.")


if __name__ == "__main__":
    main()
