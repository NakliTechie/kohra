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

Same Qwen3-0.6B lineage, same browser/WebGPU, fp16: **AR** ([onnx-community/Qwen3-0.6B-ONNX](https://huggingface.co/onnx-community/Qwen3-0.6B-ONNX) via transformers.js) vs **diffusion** (this project, fused fp16). Harness: `web/bench.html` (load `?mode=ar` and `?mode=diff` in separate fresh tabs — two ORT-web runtimes in one page contend). 128 new tokens, M-series Mac.

| mode | tok/s | forwards | latency | quality |
|---|---|---|---|---|
| **AR** (sequential + KV cache) | **27.7** | 117 | 4.2 s | coherent |
| diffusion · 128 steps | 3.2 | 128 | 38.7 s | coherent |
| diffusion · 64 steps | 6.1 | 64 | 20.3 s | coherent |
| diffusion · 32 steps | 12.0 | 32 | 10.1 s | coherent (minor repetition) |

**Fast-dLLM confidence-threshold decoding** (`generate({threshold: 0.9})`, or the `conf≥` field in the demo): instead of revealing a fixed count per step, unmask *every* position above a confidence bar. Measured: **128→61 forwards, ~2× faster, byte-identical output** on the math prompt — a free speedup when the model is confident.

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

**Takeaways.** (1) Diffusion cost is **linear in steps** — halving steps doubles throughput (128→64→32 ⇒ 3.2→6.1→12.0 tok/s), and output stays coherent down to ~64 steps. Step-reduction is *the* speed lever (and threshold decoding does it adaptively, for free). (2) **At 0.6B on a laptop, AR wins**: each AR step is a width-1 matmul + KV cache; each diffusion step is a full-width forward with *no* cache (MDLM has no KV cache), so 128 steps ≈ 128 full forwards. Diffusion's parallel-denoising bet pays off at scale and on throughput-bound hardware (DiffusionGemma's 4× is measured on H100s, not a laptop 0.6B), and via step-reduction (Fast-dLLM, trajectory distillation). (3) **Caveat:** absolute tok/s drifts with GPU/session state (a fresh browser measured diffusion-128 at ~9.8 tok/s vs 3.2 late in a long session); the AR/diffusion *ratio* and the linear step-scaling are the robust results.

## Requirements and runtime notes

- **Browser:** Chrome/Edge 121+ (WebGPU + fp16). No WebGPU → it won't run; there's no wasm fallback wired up.
- **Smaller (q4) build:** a 4-bit variant (`onnx/model_q4f16_rtn_sym.onnx`, ~680 MB vs fp16's ~1.5 GB) is published for **both** models and is a model-picker option. It runs on WebGPU via the RTN-quantized `MatMulNBits` path (quantizer: `RTNWeightOnlyQuantConfig`) on stable ORT-web 1.30.0, kohra.js's default since 2026-10-07 (it needed the `1.26.0-dev.20260416` build before). Pass `graphOptimizationLevel: 'all'`; the picker does. Checked 2026-10-07: MDLM q4, BD3LM q4 and MDLM fp16 all generate coherent text on 1.30.0, matching the dev build's output. Speed was not compared: tok/s in that session swung between ~3 and ~11 independent of the ORT version (a hidden browser pane throttles timers; not yet isolated). The q4-vs-fp16 speed at 0.6B was measured on the old dev build (q4 slower); on 1.30.0 it has not been compared in one fresh session yet.
- **Hosting your own model:** any URL that serves the `.onnx` and its `.onnx.data` side-by-side with
  permissive CORS works (Hugging Face `resolve/` URLs do). kohra auto-detects the external-data file.
- **Perf:** fused-fp16 Qwen3-0.6B-MDLM runs at **~9.8 tok/s** on an M-series Mac (128 denoise
  forwards). No KV cache — cost is steps × forward, not tokens. The export recipe + WebGPU forensics
  (why fp16 needs RMSNorm fused first) are in [`reference/MDLM-algorithm.md`](reference/MDLM-algorithm.md).

## Known constraints / gotchas

- **ORT-web WebGPU cannot run asymmetric/zero-point QMoE** → all MoE quantization must be symmetric (lesson carried over from LocalMind's LFM2-8B-A1B work).
- **fp16 on WebGPU needs the RMSNorm fused first.** A decomposed `Pow(x,2)` RMSNorm overflows native fp16 on WebGPU → silent all-zero logits (CPU/wasm reduce in fp32 and hide it). Fix: ORT's offline transformer optimizer (`model_type=qwen3`) fuses it to `SimplifiedLayerNormalization` before fp16-convert. See `reference/MDLM-algorithm.md`.
- **Dense q4 (`MatMulNBits`) on WebGPU needs RTN packing + a newer ORT-web.** `DefaultWeightOnlyQuantConfig` decodes correctly on CPU but yields sane-magnitude-but-wrong logits on WebGPU (every sym/asym/opt-level/scale-dtype variant). The fix: `RTNWeightOnlyQuantConfig` (the genai / neural-compressor RTN path) packs weights the way ORT-web's WebGPU kernel expects → coherent generation, on ORT-web ≥ `1.26.0-dev.20260416`, and on stable 1.30.0. It was the weight packing, not a kernel bug. fp16 stays the 0.6B default; q4 is for the smaller download and for models too big for fp16.
- **Transformers.js `generate()` is AR-only** — bypass it; call the model's forward directly (or use a raw ORT-web session).
- **MDLM has no KV cache** (bidirectional attention, full forward per denoise step) — the perf profile is steps × block-length, not tokens. BD3LM's architecture supports block-level KV caching, but kohra runs it **cache-free** in the browser (a static block-causal mask + full forward), since the loop-free ONNX export has no cache state to thread.
- **Watch item:** if Transformers.js ships native diffusion-loop support, fold into it rather than compete.

## Reference code

- [dLLM toolkit](https://github.com/ZHZisZZ/dllm) (Apache-2.0) — unified samplers (`dllm/core/samplers/`), A2D conversion + Tiny-A2D training/inference scripts (`examples/a2d`), Fast-dLLM caching + confidence-threshold decode.
- [LLaDA official](https://github.com/ML-GSAI/LLaDA) · [dInfer](https://github.com/inclusionAI/dInfer) (inclusionAI's diffusion-LM inference framework).
- Models: [dllm-collection](https://huggingface.co/dllm-collection) (Tiny-A2D) · [inclusionAI](https://huggingface.co/inclusionAI) (LLaDA-MoE, LLaDA2.0).

## Siblings

- **kiln** (`~/Code/kiln/`) — the MLC/WebLLM port track (compiler-level work). kohra is the ONNX + JS-loop track: no compiler, no TVM.
- **LocalMind** (`~/Code/naklios-universe/LocalMind/`) — the consumer surface, gate G4.
