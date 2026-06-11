"""Export Dream-7B (Dream-org/Dream-v0-Instruct-7B) to ONNX for ORT-web.

Dream is a Qwen2.5-7B-based MASKED DIFFUSION LM (bidirectional attention, no KV cache) —
same loop family as our Tiny-A2D Qwen3-0.6B, just bigger. `DreamModel` carries the lm_head
and returns `.logits`; its attention is bidirectional (`is_causal=False`), so a no-cache
full-sequence forward is exactly what the diffusion loop needs.

⚠ RUN THIS ON THE 64GB MAC STUDIO, not the 24GB laptop (7B export peaks ~30GB fp16 / ~56GB
fp32). See plan/dream-7b-runbook.md. fp16 export keeps peak ~30GB.

Pipeline (mirrors the 0.6B): export fp16 → optimize_onnx.py fuses RMSNorm→SimplifiedLayerNorm
(the fp16-WebGPU fix) → RTN q4 (the WebGPU-compatible quantizer) → ~4GB q4f16.

Usage:
  .venv/bin/python scripts/export_dream.py --fp16   # fp16 ONNX export + parity
"""

import argparse
import os
import sys

import numpy as np
import torch
import transformers

MODEL_ID = "Dream-org/Dream-v0-Instruct-7B"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "dream-7b-onnx")
MASK_ID = 151666          # config.mask_token_id
EOS_ID = 151643           # bos == eos == pad
# Qwen2.5-7B dims (from config.json) — passed to the fuser/quantizer downstream.
NUM_HEADS, HIDDEN, KV_HEADS, LAYERS, VOCAB = 28, 3584, 4, 28, 152064


class LogitsOnly(torch.nn.Module):
    """Bidirectional, no-cache forward → logits over the whole canvas."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        out = self.model(input_ids=input_ids, use_cache=False, return_dict=True)
        return out.logits


def load(dtype):
    # DreamModel is registered via auto_map (trust_remote_code); it has the lm_head.
    model = transformers.AutoModel.from_pretrained(
        MODEL_ID, trust_remote_code=True, dtype=dtype, attn_implementation="eager"
    ).eval()
    return LogitsOnly(model)


def export(wrapper, path, dtype):
    # short canvas trace; dynamic seq axis. Masked tail mirrors a real diffusion canvas.
    example = torch.randint(0, 151000, (1, 48), dtype=torch.long)
    example[0, 20:] = MASK_ID
    torch.onnx.export(
        wrapper, (example,), path,
        input_names=["input_ids"], output_names=["logits"],
        dynamic_axes={"input_ids": {0: "batch", 1: "sequence"},
                      "logits": {0: "batch", 1: "sequence"}},
        opset_version=17, do_constant_folding=True, dynamo=False,
    )
    print(f"exported -> {path} ({os.path.getsize(path) / 1e6:.0f} MB graph + external data)")


def parity(wrapper, onnx_path, t_len=64):
    import onnxruntime as ort
    torch.manual_seed(0)
    ids = torch.randint(0, 151000, (1, t_len), dtype=torch.long)
    ids[0, 40:] = MASK_ID
    with torch.no_grad():
        ref = wrapper(ids).float().numpy()
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    got = ort.InferenceSession(onnx_path, so, providers=["CPUExecutionProvider"]).run(
        ["logits"], {"input_ids": ids.numpy()})[0].astype(np.float32)
    masked = float((ref[0, 40:].argmax(-1) == got[0, 40:].argmax(-1)).mean())
    print(f"[dream] max|Δ|={np.max(np.abs(ref - got)):.4f}  masked-argmax={masked:.4f}")
    if masked < 0.95:
        print("[dream] PARITY WARN (<0.95)");


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp16", action="store_true", help="export in fp16 (lower peak RAM)")
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)
    dtype = torch.float16 if args.fp16 else torch.float32
    path = os.path.join(OUT_DIR, "model_fp16.onnx" if args.fp16 else "model_fp32.onnx")
    print(f"dims: heads={NUM_HEADS} hidden={HIDDEN} kv={KV_HEADS} layers={LAYERS} vocab={VOCAB} "
          f"mask={MASK_ID} dtype={dtype}")
    wrapper = load(dtype)
    if os.path.exists(path):
        print(f"{path} exists, skipping export")
    else:
        export(wrapper, path, dtype)
    parity(wrapper, path)
    print("NEXT: scripts/optimize_onnx.py with Dream dims (--num-heads 28 --hidden 3584), "
          "then --q4 (RTN). See plan/dream-7b-runbook.md.")


if __name__ == "__main__":
    main()
