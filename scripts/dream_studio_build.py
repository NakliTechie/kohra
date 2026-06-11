"""Dream-7B: fp32 ONNX → WebGPU-ready q4f16. Run ON the 64GB Studio after export_dream.py.

Why this isn't just `optimize_onnx.py`: a 7B is too big for the proven fp32→fuse→fp16 path
on a 64GB box — fusing the 28GB fp32 graph OOM-kills (shape inference doubles it past 64GB).
So Dream uses the **fp16-first** order, with two extra fixes learned the hard way:

  1. Convert fp32 → fp16 FIRST (halves the graph to 14GB → fuse fits ~28GB peak).
  2. Fuse the fp16 graph. Qwen2.5/Dream DOES fuse RMSNorm on fp16 (57 SimplifiedLayerNorm) —
     unlike Qwen3/bd3lm, which only fused RoPE. (Fewer/cleaner RMSNorms, no q/k-norm.)
  3. **Fix the RoPE type clash**: fp16-first fusion leaves the RotaryEmbedding cos/sin cache
     initializers in fp32 while q/k are fp16 → "RotaryEmbedding bound to float16 and float",
     and the model won't even load. Convert those 2 caches to fp16.
  4. RTN q4 (the WebGPU-compatible quantizer; RTNWeightOnlyQuantConfig, symmetric, QOperator).

Result: model_q4f16_rtn_sym.onnx (~5GB) that loads + runs.

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
from onnxruntime.transformers.onnx_model import OnnxModel

OUT = os.path.join(os.path.dirname(__file__), "..", "models", "dream-7b-onnx")
FP32 = os.path.join(OUT, "model_fp32.onnx")
FP16 = os.path.join(OUT, "model_fp16.onnx")
FUSED = os.path.join(OUT, "model_fp16_fused.onnx")
Q4 = os.path.join(OUT, "model_q4f16_rtn_sym.onnx")
NUM_HEADS, HIDDEN = 28, 3584


def step(name):
    print(f"\n=== {name} ===", flush=True)


def main():
    if not os.path.exists(FP16):
        step("convert fp32 -> fp16 (peak ~42GB)")
        m = OnnxModel(onnx.load(FP32))
        m.convert_float_to_float16(keep_io_types=True)
        m.save_model_to_file(FP16, use_external_data_format=True)

    if not os.path.exists(FUSED):
        step("fuse fp16 (RMSNorm -> SimplifiedLayerNorm; peak ~28GB)")
        mm = optimizer.optimize_model(FP16, model_type="qwen3", num_heads=NUM_HEADS,
                                      hidden_size=HIDDEN, opt_level=0)
        ops = collections.Counter(n.op_type for n in mm.model.graph.node)
        print("  contrib:", {k: ops[k] for k in
              ("SimplifiedLayerNormalization", "SkipSimplifiedLayerNormalization", "RotaryEmbedding")
              if k in ops}, "| still Pow:", ops.get("Pow"))
        mm.save_model_to_file(FUSED, use_external_data_format=True)

    step("fix RoPE fp32 cos/sin caches -> fp16 (else 'RotaryEmbedding bound to float16 and float')")
    m = onnx.load(FUSED)
    inits = {i.name: i for i in m.graph.initializer}
    fixed = 0
    for n in m.graph.node:
        if n.op_type == "RotaryEmbedding":
            for inp in n.input:
                t = inits.get(inp)
                if t is not None and t.data_type == TensorProto.FLOAT:
                    t.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(t).astype(np.float16), inp))
                    fixed += 1
    print(f"  fixed {fixed} fp32 RoPE caches")
    onnx.save(m, FUSED, save_as_external_data=True, all_tensors_to_one_file=True,
              location="model_fp16_fused.onnx.data")
    so = ort.SessionOptions(); so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    ort.InferenceSession(FUSED, so, providers=["CPUExecutionProvider"])  # must load
    print("  fused fp16 loads OK")

    step("RTN q4 (WebGPU-compatible)")
    q = MatMulNBitsQuantizer(onnx.load(FUSED), block_size=32, is_symmetric=True,
                             quant_format=QuantFormat.QOperator, algo_config=RTNWeightOnlyQuantConfig())
    q.process()
    q.model.save_model_to_file(Q4, use_external_data_format=True)
    print(f"  saved {Q4} ({os.path.getsize(Q4 + '.data') / 1e9:.1f}GB)")
    print("NEXT: gencheck_dream.py --model models/dream-7b-onnx/model_q4f16_rtn_sym.onnx, then pull back.")


if __name__ == "__main__":
    main()
