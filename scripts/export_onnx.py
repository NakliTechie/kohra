"""Export Tiny-A2D Qwen3-0.6B MDLM to ONNX for ORT-web.

The graph is loop-free: input_ids [B, T] -> logits [B, T, vocab]. The whole
denoising loop lives outside (Python ref / JS port). No KV cache, bidirectional
attention, attention_mask omitted (B=1, no padding => all-ones inside the model).

Usage:
  .venv/bin/python scripts/export_onnx.py            # fp32 export + parity check
  .venv/bin/python scripts/export_onnx.py --fp16     # + fp16 conversion + parity
  .venv/bin/python scripts/export_onnx.py --q4f16    # + MatMul4Bits (symmetric) on fp16
"""

import argparse
import os
import sys

import numpy as np
import torch

import dllm  # noqa: F401  (registers a2d-qwen3 with transformers Auto* classes)
import transformers

MODEL_ID = "dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1"
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "qwen3-0.6b-mdlm-onnx")
MASK_ID = 151669
EOS_ID = 151645


class LogitsOnly(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids):
        out = self.model(input_ids=input_ids, use_cache=False)
        return out.logits


def load_fp32():
    model = transformers.AutoModelForMaskedLM.from_pretrained(
        MODEL_ID, dtype=torch.float32, attn_implementation="eager"
    ).eval()
    return LogitsOnly(model)


def export_fp32(wrapper, path):
    example = torch.randint(0, 151000, (1, 48), dtype=torch.long)
    example[0, 20:] = MASK_ID
    torch.onnx.export(
        wrapper,
        (example,),
        path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )
    print(f"exported fp32 -> {path} ({os.path.getsize(path) / 1e6:.0f} MB + external data)")


def parity(wrapper, onnx_path, atol, label, t_len=72):
    import onnxruntime as ort

    torch.manual_seed(0)
    ids = torch.randint(0, 151000, (1, t_len), dtype=torch.long)
    ids[0, 40:] = MASK_ID
    with torch.no_grad():
        ref = wrapper(ids).float().numpy()

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    got = sess.run(["logits"], {"input_ids": ids.numpy()})[0].astype(np.float32)

    max_abs = float(np.max(np.abs(ref - got)))
    # what actually matters for the sampler: argmax + confidence at masked positions
    am_ref, am_got = ref.argmax(-1), got.argmax(-1)
    agree = float((am_ref == am_got).mean())
    masked_agree = float((am_ref[0, 40:] == am_got[0, 40:]).mean())
    print(f"[{label}] max|Δlogit|={max_abs:.4f}  argmax agree={agree:.4f}  (masked: {masked_agree:.4f})")
    ok = masked_agree >= (1.0 if label == "fp32" else 0.98)
    if max_abs > atol and not ok:
        print(f"[{label}] PARITY FAIL (atol={atol})")
        sys.exit(1)
    print(f"[{label}] parity OK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--q4f16", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    fp32_path = os.path.join(OUT_DIR, "model_fp32.onnx")

    wrapper = load_fp32()
    export_fp32(wrapper, fp32_path)
    parity(wrapper, fp32_path, atol=2e-3, label="fp32")

    if args.fp16 or args.q4f16:
        import onnx
        from onnxconverter_common import float16

        m = onnx.load(fp32_path)
        m16 = float16.convert_float_to_float16(
            m, keep_io_types=True, op_block_list=["Softmax"]
        )
        fp16_path = os.path.join(OUT_DIR, "model_fp16.onnx")
        onnx.save(
            m16, fp16_path, save_as_external_data=True,
            all_tensors_to_one_file=True, location="model_fp16.onnx_data",
        )
        print(f"converted fp16 -> {fp16_path}")
        parity(wrapper, fp16_path, atol=0.5, label="fp16")

    if args.q4f16:
        from onnxruntime.quantization.matmul_nbits_quantizer import (
            DefaultWeightOnlyQuantConfig,
            MatMulNBitsQuantizer,
        )
        import onnx

        fp16_path = os.path.join(OUT_DIR, "model_fp16.onnx")
        m = onnx.load(fp16_path)
        # symmetric block quant — the constraint that matters on ORT-web WebGPU
        cfg = DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True)
        q = MatMulNBitsQuantizer(m, algo_config=cfg)
        q.process()
        q4_path = os.path.join(OUT_DIR, "model_q4f16.onnx")
        q.model.save_model_to_file(q4_path, use_external_data_format=True)
        print(f"quantized q4f16 -> {q4_path} ({os.path.getsize(q4_path) / 1e6:.0f} MB)")
        parity(wrapper, q4_path, atol=5.0, label="q4f16")


if __name__ == "__main__":
    main()
