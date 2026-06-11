"""Offline transformer fusion + fp16 for the MDLM Qwen3 graph (the G1d fp16 fix).

Why this exists: the plain fp16 graph runs fine on CPU/wasm but SILENTLY ZEROS on
WebGPU. CPU/wasm fp16 kernels accumulate in fp32; WebGPU fp16 kernels are native
fp16, so the *decomposed* RMSNorm (`Pow(x,2) -> ReduceMean`) overflows fp16's 65504
ceiling -> inf -> rsqrt -> 0 -> the whole forward goes to zero, no error.

Fix: fuse the decomposed RMSNorm into ORT's `SimplifiedLayerNormalization` contrib op
(which reduces in fp32 internally) BEFORE fp16-converting. ORT 1.26 ships a `qwen3`
model type that also fuses RoPE / GQA / SkipLayerNorm. Fused contrib ops are exactly
what onnx-community's WebGPU builds ship, and those run fp16-on-WebGPU fine.

Usage:
  .venv/bin/python scripts/optimize_onnx.py            # fuse fp32 -> parity
  .venv/bin/python scripts/optimize_onnx.py --fp16     # + fp16 convert -> parity
"""

import argparse
import collections
import os
import sys

import numpy as np
import onnx
import onnxruntime as ort

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "qwen3-0.6b-mdlm-onnx")
FP32 = os.path.join(OUT_DIR, "model_fp32.onnx")
MASK_ID = 151669

# Qwen3-0.6B (this checkpoint): hidden 1024, 16 query heads (head_dim 128), 8 KV heads.
NUM_HEADS = 16
HIDDEN = 1024

CONTRIB_OPS = {
    "SimplifiedLayerNormalization",
    "SkipSimplifiedLayerNormalization",
    "LayerNormalization",
    "SkipLayerNormalization",
    "RotaryEmbedding",
    "Attention",
    "MultiHeadAttention",
    "GroupQueryAttention",
    "FastGelu",
    "BiasGelu",
    "QuickGelu",
    "MatMulNBits",
}


def histogram(model, label):
    ops = collections.Counter(n.op_type for n in model.graph.node)
    contrib = {k: v for k, v in ops.items() if k in CONTRIB_OPS}
    print(f"--- {label}: {len(model.graph.node)} nodes ---")
    print("  contrib/fused ops:", dict(contrib) if contrib else "(none)")
    for key in ("Pow", "ReduceMean", "Sqrt", "Softmax", "Sin", "Cos"):
        if ops.get(key):
            print(f"  still-decomposed {key}: {ops[key]}")
    return ops


def make_sess(path, disable_opt=True):
    so = ort.SessionOptions()
    if disable_opt:
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    return ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])


def parity(ref_sess, cand_path, label, t_len=72, disable_opt=True):
    rng = np.random.default_rng(0)
    ids = rng.integers(0, 151000, size=(1, t_len), dtype=np.int64)
    ids[0, 40:] = MASK_ID
    ref = ref_sess.run(["logits"], {"input_ids": ids})[0].astype(np.float32)
    cand = make_sess(cand_path, disable_opt).run(["logits"], {"input_ids": ids})[0].astype(np.float32)

    if not np.isfinite(cand).all():
        print(f"[{label}] NON-FINITE in candidate logits ({np.isnan(cand).sum()} nan, "
              f"{np.isinf(cand).sum()} inf)")
    if np.abs(cand).max() == 0.0:
        print(f"[{label}] ALL-ZERO candidate logits — overflow/zeroing reproduced on CPU")

    max_abs = float(np.max(np.abs(ref - cand)))
    am_ref, am_cand = ref.argmax(-1), cand.argmax(-1)
    masked = float((am_ref[0, 40:] == am_cand[0, 40:]).mean())
    allagree = float((am_ref == am_cand).mean())
    print(f"[{label}] max|Δlogit|={max_abs:.4f}  argmax agree={allagree:.4f}  (masked: {masked:.4f})")
    return masked


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--q4", action="store_true",
                    help="MatMulNBits 4-bit on the fused fp32 graph, then fp16-convert (q4f16)")
    ap.add_argument("--q4-from-fp16", action="store_true",
                    help="MatMulNBits directly on the fused fp16 graph (onnx-community order; "
                         "no post-quantize fp16-convert that may mangle the MatMulNBits layout)")
    ap.add_argument("--q4-fp32scales", action="store_true",
                    help="diagnostic: 4-bit weights but fp32 scales + fp32 activations "
                         "(tests whether WebGPU MatMulNBits mis-reads fp16 scales)")
    ap.add_argument("--exclude-lm-head", action="store_true",
                    help="keep lm_head in fp16 (better logit fidelity, +~230MB)")
    ap.add_argument("--symmetric", action="store_true",
                    help="symmetric q4 (no zero-point); default asymmetric")
    ap.add_argument("--accuracy-level", type=int, default=None,
                    help="MatMulNBits accuracy_level (4=int8 activations, the WebGPU-optimized path)")
    ap.add_argument("--opt-level", type=int, default=0,
                    help="0 = python fusions only (avoids ORT C++ opt crash on inserted casts)")
    args = ap.parse_args()

    from onnxruntime.transformers import optimizer
    from onnxruntime.transformers.fusion_options import FusionOptions
    from onnxruntime.transformers.onnx_model import OnnxModel

    ref_sess = make_sess(FP32, disable_opt=False)  # fp32 reference (exact-parity to torch)
    fused_path = os.path.join(OUT_DIR, "model_fp32_fused.onnx")

    if os.path.exists(fused_path):
        print(f"fused fp32 exists, skipping fusion ({fused_path})")
    else:
        print("== fuse fp32 graph (model_type=qwen3) ==")
        opts = FusionOptions("qwen3")
        m = optimizer.optimize_model(
            FP32,
            model_type="qwen3",
            num_heads=NUM_HEADS,
            hidden_size=HIDDEN,
            optimization_options=opts,
            opt_level=args.opt_level,
        )
        histogram(m.model, "fused fp32")
        m.save_model_to_file(fused_path, use_external_data_format=True)
        print(f"saved -> {fused_path}")
        parity(ref_sess, fused_path, "fused-fp32", disable_opt=True)

    if args.fp16:
        print("\n== fp16-convert the fused graph (fresh disk load) ==")
        # Load fresh so external-data paths resolve against the model dir; reusing the
        # in-memory post-save model leaves dangling external refs (cwd-relative).
        m = OnnxModel(onnx.load(fused_path))
        m.convert_float_to_float16(keep_io_types=True)
        histogram(m.model, "fused fp16")
        fp16_path = os.path.join(OUT_DIR, "model_fp16_fused.onnx")
        m.save_model_to_file(fp16_path, use_external_data_format=True)
        print(f"saved -> {fp16_path}")
        parity(ref_sess, fp16_path, "fused-fp16", disable_opt=True)

    if args.q4:
        # Quantize the FUSED FP32 graph (clean weight initializers), not the
        # cast-littered fp16 graph (the 2026-06-10 dead end: most MatMuls skipped
        # because their weights were Cast outputs, not initializers).
        from onnxruntime.quantization.matmul_nbits_quantizer import (
            DefaultWeightOnlyQuantConfig,
            MatMulNBitsQuantizer,
        )

        # THE WEBGPU FIX (2026-06-11): use RTNWeightOnlyQuantConfig (the genai /
        # neural_compressor RTN path), NOT DefaultWeightOnlyQuantConfig. Same MatMulNBits
        # attrs, but the RTN path packs weights the way ORT-web's WebGPU kernel expects.
        # DefaultWeightOnlyQuantConfig → CPU-correct but WebGPU garbage (masked 0.17);
        # RTNWeightOnlyQuantConfig → coherent WebGPU generation. Symmetric + QOperator,
        # matching onnx-community's producer (onnxruntime-genai). See reference doc.
        from onnxruntime.quantization.matmul_nbits_quantizer import RTNWeightOnlyQuantConfig
        from onnxruntime.quantization import QuantFormat

        print(f"\n== q4 quantize fused fp32 (RTN, symmetric, exclude_lm_head={args.exclude_lm_head}) ==")
        model = onnx.load(fused_path)
        exclude = []
        if args.exclude_lm_head:
            exclude = [n.name for n in model.graph.node
                       if n.op_type == "MatMul" and n.output and n.output[0] == "logits"
                       or n.name == "/model/lm_head/MatMul"]
        cfg = RTNWeightOnlyQuantConfig()
        q = MatMulNBitsQuantizer(model, block_size=32, is_symmetric=True,
                                 quant_format=QuantFormat.QOperator,
                                 algo_config=cfg, nodes_to_exclude=exclude)
        q.process()
        histogram(q.model.model, "q4 RTN (fp32 scales)")

        # fp16-convert the remainder: MatMulNBits packed weights are uint8 (untouched),
        # scales -> fp16, attention/elementwise -> fp16. This is q4f16 (onnx-community shape).
        qm = OnnxModel(q.model.model)
        qm.convert_float_to_float16(keep_io_types=True)
        histogram(qm.model, "q4f16")
        q4_path = os.path.join(OUT_DIR, "model_q4f16_rtn_sym.onnx")  # the WebGPU-working q4
        qm.save_model_to_file(q4_path, use_external_data_format=True)
        sz = os.path.getsize(q4_path + ".data") / 1e6
        print(f"saved -> {q4_path} ({sz:.0f} MB data)")
        parity(ref_sess, q4_path, "q4f16-rtn", disable_opt=True)
        print("NOTE: run on WebGPU with ORT-web >= 1.26.0-dev.20260416 (transformers.js's build). "
              "Synthetic-probe masked-argmax ~0.5 understates it; real generation is coherent.")

    if args.q4_fp32scales:
        # Diagnostic: 4-bit weights but KEEP fp32 scales + fp32 activations (no
        # fp16-convert). Isolates whether WebGPU MatMulNBits mis-reads fp16 scales.
        from onnxruntime.quantization.matmul_nbits_quantizer import (
            DefaultWeightOnlyQuantConfig,
            MatMulNBitsQuantizer,
        )

        print(f"\n== q4 on fused fp32, fp32 scales (symmetric={args.symmetric}) ==")
        model = onnx.load(fused_path)
        cfg = DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=args.symmetric)
        q = MatMulNBitsQuantizer(model, algo_config=cfg)
        q.process()
        histogram(q.model.model, "q4 fp32-scales")
        tag = "sym" if args.symmetric else "asym"
        q4_path = os.path.join(OUT_DIR, f"model_q4_fp32scales_{tag}.onnx")
        q.model.save_model_to_file(q4_path, use_external_data_format=True)
        sz = os.path.getsize(q4_path + ".data") / 1e6
        print(f"saved -> {q4_path} ({sz:.0f} MB data)")
        parity(ref_sess, q4_path, "q4-fp32scales", disable_opt=True)

    if args.q4_from_fp16:
        from onnxruntime.quantization.matmul_nbits_quantizer import (
            DefaultWeightOnlyQuantConfig,
            MatMulNBitsQuantizer,
        )

        fp16_path = os.path.join(OUT_DIR, "model_fp16_fused.onnx")
        print(f"\n== q4 quantize the FUSED FP16 graph directly (symmetric={args.symmetric}, "
              f"accuracy_level={args.accuracy_level}) ==")
        model = onnx.load(fp16_path)
        cfg_kw = dict(block_size=32, is_symmetric=args.symmetric)
        if args.accuracy_level is not None:
            cfg_kw["accuracy_level"] = args.accuracy_level
        cfg = DefaultWeightOnlyQuantConfig(**cfg_kw)
        q = MatMulNBitsQuantizer(model, algo_config=cfg)
        q.process()
        histogram(q.model.model, "q4f16 (from fp16)")
        tag = "sym" if args.symmetric else "asym"
        if args.accuracy_level is not None:
            tag += f"_acc{args.accuracy_level}"
        q4_path = os.path.join(OUT_DIR, f"model_q4f16_fromfp16_{tag}.onnx")
        q.model.save_model_to_file(q4_path, use_external_data_format=True)
        sz = os.path.getsize(q4_path + ".data") / 1e6
        print(f"saved -> {q4_path} ({sz:.0f} MB data)")
        parity(ref_sess, q4_path, "q4f16-fromfp16", disable_opt=True)


if __name__ == "__main__":
    main()
