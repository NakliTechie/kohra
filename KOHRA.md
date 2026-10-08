# KOHRA.md — the founding document

The argument, the gate ladder, the measurements and the hard-won gotchas behind
[kohra](README.md). The README gets you running; this file says why it exists and what has
been learned.

## Why

Google's [DiffusionGemma](https://huggingface.co/google/diffusiongemma-26B-A4B-it) (weights launched 2026-06-10, Apache 2.0) put text diffusion on the map: instead of token-by-token autoregression, the model generates whole 256-token blocks in parallel via iterative denoising (~48 steps) — up to 4× faster on GPUs (1000+ tok/s on an H100). But it's a 26B-A4B MoE (~18 GB quantized — past browser physics), and **no browser runtime anywhere supports a diffusion generation loop** — Transformers.js, onnxruntime-web, and MLC/WebLLM are all autoregressive-only.

The irony: parallel block-denoising plays to WebGPU batch throughput *exactly* where sequential AR decode is the browser's bottleneck. Diffusion should eventually be a better fit for in-browser inference than AR is. kohra builds the missing stack.

## The thesis

Two missing pieces, built as one vertically-sliced project (they're only testable against each other):

1. **ONNX exports of small diffusion LMs** — the riskier half (custom arch configs, MoE quantization constraints).
2. **A JS denoising/sampling loop** over raw ONNX forward passes — the easier half (~few hundred lines: start masked → forward → lock high-confidence tokens → repeat).

## Gate ladder

- **G1 — first coherent block in a browser. ✅ DONE** (fp32 then fused-fp16, ~9.8 tok/s). Target: [`dllm-collection/Qwen3-0.6B-diffusion-mdlm-v0.1`](https://huggingface.co/dllm-collection/Qwen3-0.6B-diffusion-mdlm-v0.1) (Tiny-A2D: Qwen3-0.6B adapted to masked diffusion). onnx-community's AR Qwen3-0.6B export is the conversion template. q4f16 (~680MB) now runs on WebGPU via RTN quantization. The [bd3lm sibling](https://huggingface.co/dllm-collection/Qwen3-0.6B-diffusion-bd3lm-v0.1) (block diffusion) is also exported and runs in-browser — both ship as model-picker options (fp16 + q4).
- **G2 — sampler quality + perf. ✅ done.** AR-vs-diffusion same-browser A/B (see Benchmark below); step/block schedules measured; confidence-threshold decoding (Fast-dLLM) shipped (~2× fewer forwards, byte-identical output).
- **G3 — a genuinely useful model.** [`inclusionAI/LLaDA-MoE-7B-A1B-Instruct`](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct) (+ `-Instruct-TD`, trajectory-distilled for fewer denoise steps). First open MoE diffusion LM: 7B total / 1.4B active, quality ≈ Qwen2.5-3B-Instruct. Size class already proven in-browser by LFM2-8B-A1B.
- **G4 — LocalMind integration** via its runtime-adapter `MODELS` pattern.
- **North star — DiffusionGemma** ([`google/diffusiongemma-26B-A4B-it`](https://huggingface.co/google/diffusiongemma-26B-A4B-it), Apache 2.0). **Weights launched 2026-06-10** (Gemma-4 backbone, 26B total / 3.8B active MoE, multimodal, 256K context; GGUF/MLX/vLLM out). 26B total (~18GB q4) is past browser physics — desktop play now actionable via MLX 4-bit on a big-RAM Mac. The browser angle is a future small/distilled diffusion-Gemma variant (watch [Gemma 4 E2B/E4B](https://huggingface.co/google/gemma-4-E2B) — those small sizes are AR today, but a diffusion variant at that scale would be the browser-feasible ambitious target).

## Benchmark: autoregressive vs masked diffusion (same browser)

Same Qwen3-0.6B lineage, same browser/WebGPU, fp16: **AR** ([onnx-community/Qwen3-0.6B-ONNX](https://huggingface.co/onnx-community/Qwen3-0.6B-ONNX) via transformers.js) vs **diffusion** (this project, fused fp16). Harness: `web/bench.html` (load `?mode=ar` and `?mode=diff` in separate fresh tabs — two ORT-web runtimes in one page contend). 128 new tokens, MacBook Pro M4 Pro (20-core GPU, 26 GB), ORT-web 1.30.0, math prompt. Re-measured 2026-10-08 with macOS thermal state `nominal` for every run.

| mode | tok/s | forwards | latency | quality |
|---|---|---|---|---|
| **AR** (sequential + KV cache) | **29.2** (33.9 on a second run) | 117 | 4.0 s | coherent |
| diffusion · 128 steps | 10.1 | 128 | 12.2 s | coherent |
| diffusion · 64 steps | 20.3 | 64 | 6.0 s | coherent |
| diffusion · 32 steps | 39.1 | 32 | 3.0 s | glitches ("48 * 8  384", a stray ".") |
| diffusion · conf≥0.8, 128 steps | 26.4 | 50 | 4.7 s | same answer as 128 steps |

**Thermal state decides the absolute numbers.** A forward takes 83–105 ms when macOS reports thermal state `nominal` and 217–335 ms when it reports `fair`, on the same page and build. That is the ~3 vs ~10 tok/s swing seen on 2026-10-07. Ruled out on 2026-10-08: ORT version, a hidden browser pane, `index.html` canvas rendering, and memory pressure (5 models resident, 29% free, no slowdown). `session.run` (GPU work plus the logits readback) is 10.7 s of a 12.1 s run; the JS sampler is 1.3 s. Benchmark only in `nominal`; check with `ProcessInfo.processInfo.thermalState`.

**Fast-dLLM confidence-threshold decoding** (`generate({threshold: 0.8})`, or the `conf≥` field in the demo): instead of revealing a fixed count per step, unmask *every* position above a confidence bar.

**Threshold sweep, 2026-10-07** (ORT-web 1.30.0, fp16, 128 steps, 3 prompts; forward passes summed over the prompts):

| threshold | MDLM fwd | BD3LM fwd | text |
|---|---|---|---|
| off (fixed 128 steps) | 384 | 328 | baseline |
| 0.95 | 252 | 174 | same answers |
| 0.9 | 229 | 155 | same answers |
| **0.8** | **194** | **137** | MDLM text identical to fixed-step on 2 of 3 prompts; BD3LM identical to 0.9 |
| 0.7 | 172 | 113 | MDLM breaks on the math prompt ("48 * 8 = 38 km km") |

0.8 is the demo default for both models.

**Tried and dropped: S2PD serial→parallel decoding** ([arXiv:2610.06847](https://arxiv.org/abs/2610.06847)) — denoise each block left-to-right down to a masked fraction τ, then finish the whole canvas in parallel. Measured 2026-10-07 on both fp16 graphs, 3 prompts: it cuts forwards linearly (MDLM 128→71 at τ 0.6) but doubles adjacent tokens from τ 0.2 on ("is is", "than than"), and BD3LM output collapses from τ 0.4. Threshold decoding gets fewer forwards (61) with unchanged text, and combining the two is worse than either (88). The paper's gain comes from batching amortising weight reads; with no KV cache every kohra forward is already full-canvas, so the only lever is tokens revealed per forward — which the threshold already pulls adaptively.

**BD3LM block KV cache, 2026-10-08** (fp16, WebGPU, 128 tokens, thermal `nominal`, same output as without the cache on all 6 runs):

| decoding | prompt | no cache | KV cache | speedup |
|---|---|---|---|---|
| 128 steps | math | 108 fwd · 11.27 s | 112 fwd · 6.16 s | 1.83× |
| 128 steps | sky | 110 fwd · 11.04 s | 114 fwd · 6.12 s | 1.80× |
| 128 steps | poem | 110 fwd · 12.76 s | 114 fwd · 6.21 s | 2.05× |
| conf≥0.8 | math | 51 fwd · 9.27 s | 55 fwd · 3.00 s | 3.09× (slow baseline run) |
| conf≥0.8 | sky | 50 fwd · 4.87 s | 54 fwd · 2.85 s | 1.71× |
| conf≥0.8 | poem | 36 fwd · 3.38 s | 40 fwd · 2.03 s | 1.67× |

The cache adds one commit forward per finished block (4 here) and still halves wall time.

**Takeaways.** (1) Diffusion cost is **linear in steps**: 128→64→32 steps gives 10.1→20.3→39.1 tok/s. Text stays clean down to 64 steps and glitches at 32. (2) **AR still wins at equal quality**: 29–34 tok/s against 26.4 for diffusion at conf≥0.8, the fastest setting with unchanged output. Each AR step is a width-1 matmul with a KV cache; each diffusion step is a full-canvas forward with no cache. Diffusion's parallel-denoising bet pays off at scale and on throughput-bound hardware (DiffusionGemma's speedup is measured on H100s) and through step reduction. (3) The 2026-06 table (diffusion-128 at 3.2 tok/s) was measured in a throttled state; the 2026-10-08 table replaces it.

## Requirements and runtime notes

- **Browser:** Chrome/Edge 121+ (WebGPU + fp16). No WebGPU → it won't run; there's no wasm fallback wired up.
- **Smaller (q4) build:** a 4-bit variant (`onnx/model_q4f16_rtn_sym.onnx`, ~680 MB vs fp16's ~1.5 GB) is published for **both** models and is a model-picker option. It runs on WebGPU via the RTN-quantized `MatMulNBits` path (quantizer: `RTNWeightOnlyQuantConfig`) on stable ORT-web 1.30.0, kohra.js's default since 2026-10-07 (it needed the `1.26.0-dev.20260416` build before). Pass `graphOptimizationLevel: 'all'`; the picker does. Checked 2026-10-07: MDLM q4, BD3LM q4 and MDLM fp16 all generate coherent text on 1.30.0, matching the dev build's output. Speed was not compared: tok/s in that session swung between ~3 and ~11 independent of the ORT version (a hidden browser pane throttles timers; not yet isolated). The q4-vs-fp16 speed at 0.6B was measured on the old dev build (q4 slower); on 1.30.0 it has not been compared in one fresh session yet.
- **Hosting your own model:** any URL that serves the `.onnx` and its `.onnx.data` side-by-side with
  permissive CORS works (Hugging Face `resolve/` URLs do). kohra auto-detects the external-data file.
- **Perf:** fused-fp16 Qwen3-0.6B-MDLM runs at **~10 tok/s** on an M4 Pro in thermal state `nominal` (128 denoise
  forwards). No KV cache — cost is steps × forward, not tokens. The export recipe + WebGPU forensics
  (why fp16 needs RMSNorm fused first) are in [`reference/MDLM-algorithm.md`](reference/MDLM-algorithm.md).

## Known constraints / gotchas

- **ORT-web WebGPU cannot run asymmetric/zero-point QMoE** → all MoE quantization must be symmetric (lesson carried over from LocalMind's LFM2-8B-A1B work).
- **fp16 on WebGPU needs the RMSNorm fused first.** A decomposed `Pow(x,2)` RMSNorm overflows native fp16 on WebGPU → silent all-zero logits (CPU/wasm reduce in fp32 and hide it). Fix: ORT's offline transformer optimizer (`model_type=qwen3`) fuses it to `SimplifiedLayerNormalization` before fp16-convert. See `reference/MDLM-algorithm.md`.
- **Dense q4 (`MatMulNBits`) on WebGPU needs RTN packing + a newer ORT-web.** `DefaultWeightOnlyQuantConfig` decodes correctly on CPU but yields sane-magnitude-but-wrong logits on WebGPU (every sym/asym/opt-level/scale-dtype variant). The fix: `RTNWeightOnlyQuantConfig` (the genai / neural-compressor RTN path) packs weights the way ORT-web's WebGPU kernel expects → coherent generation, on ORT-web ≥ `1.26.0-dev.20260416`, and on stable 1.30.0. It was the weight packing, not a kernel bug. fp16 stays the 0.6B default; q4 is for the smaller download and for models too big for fp16.
- **Transformers.js `generate()` is AR-only** — bypass it; call the model's forward directly (or use a raw ORT-web session).
- **MDLM has no KV cache** (bidirectional attention, full forward per denoise step) — the perf profile is steps × canvas, not tokens. **BD3LM has an exact block KV cache** (2026-10-08): block-causal attention means a finished block's keys/values never change, so `onnx/model_kv_fp16_fused.onnx` denoises only the current block against a cache of earlier ones (`past_key_i`/`present_key_i` I/O, positions fed explicitly, attention left decomposed because GQA contrib ops are causal). One commit pass per finished block writes it into the cache. Token-identical to the cache-free graph in the browser (6/6 runs), 1.7–2.1× faster at 128 tokens; the gain grows with output length. Export: `scripts/export_bd3lm_kv.py`; checks: `scripts/kv_reference.py` (torch) and `scripts/gencheck_bd3lm_kv.py` (ONNX vs torch, 128/128 ids in fp32).
- **Watch item:** if Transformers.js ships native diffusion-loop support, fold into it rather than compete.

## Reference code

- [dLLM toolkit](https://github.com/ZHZisZZ/dllm) (Apache-2.0) — unified samplers (`dllm/core/samplers/`), A2D conversion + Tiny-A2D training/inference scripts (`examples/a2d`), Fast-dLLM caching + confidence-threshold decode.
- [LLaDA official](https://github.com/ML-GSAI/LLaDA) · [dInfer](https://github.com/inclusionAI/dInfer) (inclusionAI's diffusion-LM inference framework).
- Models: [dllm-collection](https://huggingface.co/dllm-collection) (Tiny-A2D) · [inclusionAI](https://huggingface.co/inclusionAI) (LLaDA-MoE, LLaDA2.0).

## Siblings

- **kiln** (`~/Code/kiln/`) — the MLC/WebLLM port track (compiler-level work). kohra is the ONNX + JS-loop track: no compiler, no TVM.
- **LocalMind** (`~/Code/naklios-universe/LocalMind/`) — the consumer surface, gate G4.
