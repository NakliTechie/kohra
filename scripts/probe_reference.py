"""Generate a fixed probe input + ground-truth argmax for the browser fp16 probe.

Runs the verified fp32 ONNX (CPU) on one deterministic input and writes
web/probe_input.json = {input_ids, masked_positions, expected_argmax, max_abs}.
The browser probe (web/probe.html) loads any candidate model, runs the SAME
input on WebGPU, and checks: finite? all-zero? argmax agreement at masked
positions vs this ground truth. That's the fast A/B for "does fp16 zero on WebGPU".
"""

import json
import os

import numpy as np
import onnxruntime as ort

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "qwen3-0.6b-mdlm-onnx")
WEB = os.path.join(os.path.dirname(__file__), "..", "web")
FP32 = os.path.join(OUT_DIR, "model_fp32.onnx")
MASK_ID = 151669
T_LEN = 64
MASK_FROM = 40

rng = np.random.default_rng(7)
ids = rng.integers(0, 151000, size=(1, T_LEN), dtype=np.int64)
ids[0, MASK_FROM:] = MASK_ID

so = ort.SessionOptions()  # fp32: default opts fine
sess = ort.InferenceSession(FP32, so, providers=["CPUExecutionProvider"])
logits = sess.run(["logits"], {"input_ids": ids})[0][0]  # [T, vocab]

masked_positions = list(range(MASK_FROM, T_LEN))
expected_argmax = [int(logits[p].argmax()) for p in masked_positions]

out = {
    "input_ids": ids[0].tolist(),
    "masked_positions": masked_positions,
    "expected_argmax": expected_argmax,
    "max_abs": float(np.abs(logits).max()),
}
path = os.path.join(WEB, "probe_input.json")
with open(path, "w") as f:
    json.dump(out, f)
print(f"wrote {path}")
print(f"  T={T_LEN} masked={len(masked_positions)} max|logit|={out['max_abs']:.2f}")
print(f"  expected_argmax[:8]={expected_argmax[:8]}")
