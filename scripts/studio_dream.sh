#!/usr/bin/env bash
# Dream-7B → browser q4, run ON the 64GB Mac Studio. Driven from the laptop via:
#   ssh $S 'cd ~/code/chirag/kohra && nohup bash scripts/studio_dream.sh > dream.log 2>&1 </dev/null & disown'
#   ssh $S 'tail -f ~/code/chirag/kohra/dream.log'
# Re-runnable: each step skips if its output exists. fp32 path (fp16-direct fuse fails —
# RMSNorm won't fuse through the casts; see the bd3lm commit). Peaks ~56GB; the Studio takes it.
set -euo pipefail
cd "$(dirname "$0")/.."
P=.venv/bin/python

echo "=== [1/3] export Dream fp32 (skip parity to bound RAM) ==="
$P scripts/export_dream.py --skip-parity

echo "=== [2/3] convert fp16 -> fuse -> fix RoPE -> RTN q4 ==="
# NB: NOT optimize_onnx.py's fp32-fuse path — that OOM-kills on a 7B (28GB graph > 64GB during
# shape inference). dream_studio_build.py does the fp16-first chain + the RoPE fp32->fp16 fix.
$P scripts/dream_studio_build.py

echo "=== [3/3] gencheck (q4f16, CPU) — coherent prose = gate passed ==="
$P scripts/gencheck_dream.py --model models/dream-7b-onnx/model_q4f16_rtn_sym.onnx \
  --max-new-tokens 32 --steps 16 || true

echo "sizes:"; ls -lh models/dream-7b-onnx/*.onnx.data 2>/dev/null | awk '{print "  "$5,$9}'
echo STUDIO_DREAM_DONE
