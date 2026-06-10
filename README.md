# kohra

> Text-diffusion language models running in the browser — a JS denoising loop over ONNX Runtime Web / WebGPU. To our knowledge, nobody has shipped this yet.

कोहरा — *fog*. Diffusion generation starts as noise and denoises, pass by pass, into clear text.

## Why

Google's DiffusionGemma announcement (2026-06) put text diffusion on the map: instead of token-by-token autoregression, the model generates whole blocks in parallel via iterative denoising — up to 4× faster on GPUs (1000+ tok/s on an H100). But the release is a 26B-A4B MoE with no ONNX export, and **no browser runtime anywhere supports a diffusion generation loop** — Transformers.js, onnxruntime-web, and MLC/WebLLM are all autoregressive-only.

The irony: parallel block-denoising plays to WebGPU batch throughput *exactly* where sequential AR decode is the browser's bottleneck. Diffusion should eventually be a better fit for in-browser inference than AR is. kohra builds the missing stack.

## The thesis

Two missing pieces, built as one vertically-sliced project (they're only testable against each other):

1. **ONNX exports of small diffusion LMs** — the riskier half (custom arch configs, MoE quantization constraints).
2. **A JS denoising/sampling loop** over raw ONNX forward passes — the easier half (~few hundred lines: start masked → forward → lock high-confidence tokens → repeat).

## Gate ladder

- **G1 — first coherent block in a browser.** Target: [`dllm-collection/Qwen3-0.6B-diffusion-mdlm-v0.1`](https://huggingface.co/dllm-collection/Qwen3-0.6B-diffusion-mdlm-v0.1) (Tiny-A2D: Qwen3-0.6B adapted to masked diffusion; ~500MB at q4f16 — runs anywhere). onnx-community's AR Qwen3-0.6B export is the conversion template. Sibling [bd3lm variant](https://huggingface.co/dllm-collection/Qwen3-0.6B-diffusion-bd3lm-v0.1) for block diffusion later.
- **G2 — sampler quality + perf.** Confidence-threshold decoding (Fast-dLLM), step/block schedules. Benchmark AR Qwen3-0.6B vs diffusion Qwen3-0.6B **in the same browser** — same weight lineage, clean A/B of the two generation paradigms. This comparison is a publishable artifact on its own.
- **G3 — a genuinely useful model.** [`inclusionAI/LLaDA-MoE-7B-A1B-Instruct`](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct) (+ `-Instruct-TD`, trajectory-distilled for fewer denoise steps). First open MoE diffusion LM: 7B total / 1.4B active, quality ≈ Qwen2.5-3B-Instruct. Size class already proven in-browser by LFM2-8B-A1B.
- **G4 — LocalMind integration** via its runtime-adapter `MODELS` pattern.
- **North star — DiffusionGemma** (`google/diffusiongemma-26B-A4B-it`, Apache 2.0) when a small variant or community distillation appears. 26B total (~13–14GB q4) is past browser physics today. Desktop play is available anytime via mlx-community 4-bit on a big-RAM Mac.

## Reference code

- [dLLM toolkit](https://github.com/ZHZisZZ/dllm) (Apache-2.0) — unified samplers (`dllm/core/samplers/`), A2D conversion + Tiny-A2D training/inference scripts (`examples/a2d`), Fast-dLLM caching + confidence-threshold decode.
- [LLaDA official](https://github.com/ML-GSAI/LLaDA) · [dInfer](https://github.com/inclusionAI/dInfer) (inclusionAI's diffusion-LM inference framework).
- Models: [dllm-collection](https://huggingface.co/dllm-collection) (Tiny-A2D) · [inclusionAI](https://huggingface.co/inclusionAI) (LLaDA-MoE, LLaDA2.0).

## Known constraints / gotchas

- **ORT-web WebGPU cannot run asymmetric/zero-point QMoE** → all MoE quantization must be symmetric (lesson carried over from LocalMind's LFM2-8B-A1B work).
- **Transformers.js `generate()` is AR-only** — bypass it; call the model's forward directly (or use a raw ORT-web session).
- **MDLM has no KV cache** (bidirectional attention, full forward per denoise step) — the perf profile is steps × block-length, not tokens. BD3LM restores block-level KV caching.
- **Watch item:** if Transformers.js ships native diffusion-loop support, fold into it rather than compete.

## Siblings

- **kiln** (`~/Code/kiln/`) — the MLC/WebLLM port track (compiler-level work). kohra is the ONNX + JS-loop track: no compiler, no TVM.
- **LocalMind** (`~/Code/naklios-universe/LocalMind/`) — the consumer surface, gate G4.
