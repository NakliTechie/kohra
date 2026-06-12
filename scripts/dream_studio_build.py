"""Dream-7B: fp32 ONNX → WebGPU-ready q4f16. Run ON the 64GB Studio after export_dream.py.

Why this isn't just `optimize_onnx.py`: a 7B is too big for the proven fp32→fuse→fp16 path
on a 64GB box — fusing the 28GB fp32 graph OOM-kills (shape inference doubles it past 64GB).
So Dream goes fp16-FIRST, all IN ONE PROCESS (critical: never re-load the >2GB graph from disk —
onnx.load keeps external refs for >2GB models, so a load→modify→save-to-same-file corrupts the
weights). The chain, in memory:

  1. convert fp32 → fp16 (14GB; fuse then fits ~28GB peak)
  2. fuse fp16 — Qwen2.5/Dream DOES fuse RMSNorm on fp16 (57 SimplifiedLayerNorm), unlike Qwen3
  3. fix the RoPE type clash IN MEMORY: fp16-first fusion leaves cos/sin caches fp32 vs fp16 q/k
     → "RotaryEmbedding bound to float16 and float", model won't load. Convert caches to fp16.
  4. RTN q4 IN MEMORY (RTNWeightOnlyQuantConfig, symmetric, QOperator — WebGPU-compatible).

Result: model_q4f16_rtn_sym.onnx (~5GB) that loads + generates.

  .venv/bin/python scripts/dream_studio_build.py
"""

import collections
import os

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, numpy_helper
from onnxruntime.quantization import QuantFormat
from onnxruntime.quantization.matmul_nbits_quantizer import (
    MatMulNBitsQuantizer,
    RTNWeightOnlyQuantConfig,
)
from onnxruntime.transformers import optimizer

OUT = os.path.join(os.path.dirname(__file__), "..", "models", "dream-7b-onnx")
FP32 = os.path.join(OUT, "model_fp32.onnx")
FP16 = os.path.join(OUT, "model_fp16.onnx")
FUSED = os.path.join(OUT, "model_fp16_fused.onnx")
Q4 = os.path.join(OUT, "model_q4f16_rtn_sym.onnx")
NUM_HEADS, HIDDEN = 28, 3584


def fix_rope_inplace(proto):
    """Convert fp32 RotaryEmbedding cos/sin cache initializers to fp16 (in-memory proto)."""
    inits = {i.name: i for i in proto.graph.initializer}
    fixed = 0
    for n in proto.graph.node:
        if n.op_type == "RotaryEmbedding":
            for inp in n.input:
                t = inits.get(inp)
                if t is not None and t.data_type == TensorProto.FLOAT:
                    t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float16), inp))
                    fixed += 1
    return fixed


def main():
    if not os.path.exists(FP16):
        print("convert fp32 -> fp16 (peak ~42GB)…", flush=True)
        from onnxruntime.transformers.onnx_model import OnnxModel
        m = OnnxModel(onnx.load(FP32))
        m.convert_float_to_float16(keep_io_types=True)
        m.save_model_to_file(FP16, use_external_data_format=True)

    print("fuse fp16 (RMSNorm -> SimplifiedLayerNorm; peak ~28GB)…", flush=True)
    mm = optimizer.optimize_model(FP16, model_type="qwen3", num_heads=NUM_HEADS,
                                  hidden_size=HIDDEN, opt_level=0)
    proto = mm.model  # ModelProto, weights in memory — do NOT reload from disk
    ops = collections.Counter(n.op_type for n in proto.graph.node)
    print("  contrib:", {k: ops[k] for k in
          ("SimplifiedLayerNormalization", "SkipSimplifiedLayerNormalization", "RotaryEmbedding")
          if k in ops}, "| still Pow:", ops.get("Pow"), flush=True)

    print("fix RoPE fp32 cos/sin caches -> fp16 (in memory)…", flush=True)
    print("  fixed", fix_rope_inplace(proto), "caches", flush=True)

    # RTN q4 BEFORE saving anything: save_model_to_file externalizes the in-memory proto
    # (moves weights to disk + sets dangling external refs), which then breaks the quantizer's
    # in-memory read. So quantize the in-memory `proto` first.
    print("RTN q4 (in memory, WebGPU-compatible)…", flush=True)
    q = MatMulNBitsQuantizer(proto, block_size=32, is_symmetric=True,
                             quant_format=QuantFormat.QOperator, algo_config=RTNWeightOnlyQuantConfig())
    q.process()
    q.model.save_model_to_file(Q4, use_external_data_format=True)
    print(f"  saved {Q4} ({os.path.getsize(Q4 + '.data') / 1e9:.1f}GB)", flush=True)

    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    ort.InferenceSession(Q4, so, providers=["CPUExecutionProvider"])  # q4 must load
    print("  q4 loads OK", flush=True)
    print("NEXT: gencheck_dream.py --model models/dream-7b-onnx/model_q4f16_rtn_sym.onnx", flush=True)


if __name__ == "__main__":
    main()
