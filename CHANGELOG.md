# Changelog

All notable changes to kohra are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [SemVer](https://semver.org/).

## [0.2.1] — 2026-10-08

Documentation only; `kohra.js` is unchanged from 0.2.0.

### Fixed
- README: a runnable BD3LM KV-cache example, the Python setup line for the export scripts,
  the correct test count (13), and an accurate description of what the bench page measures.

## [0.2.0] — 2026-10-08

### Added
- BD3LM block KV cache: `from_pretrained({ ..., kvCache: true })` with
  `onnx/model_kv_fp16_fused.onnx`. Each denoise step runs only the current block against a
  cache of finished blocks; output is token-identical to the cache-free graph and about 2×
  faster at 128 tokens. A q4 variant, `onnx/model_kv_q4f16_rtn_sym.onnx`, is token-identical to
  the plain q4 graph and also about 2× faster. Both demo BD3LM options use the cache.
- `scripts/export_bd3lm_kv.py`, `scripts/kv_reference.py`, `scripts/gencheck_bd3lm_kv.py`.
- 2 KV-cache sampler tests (13 total).

### Fixed
- bench.html: TTFT now times the first generated token, not transformers.js's prompt echo.

## [0.1.0] — 2026-10-08

First release on npm, as `kohra.js` (npm refused `kohra` as too close to `koa`/`ora`). `kohra.js` is a single ES module for the browser; it loads
onnxruntime-web 1.30.0 and the tokenizer from a CDN at runtime.

### Added
- `pipeline('text-diffusion', { model, tokenizer })` and `DiffusionLM.from_pretrained()` /
  `generate()`: a JS masked-diffusion denoising loop over raw ONNX forward passes on WebGPU.
- Block diffusion (BD3LM) via `blockCausal: true`: a static block-causal attention mask on
  the `pos // blockSize` grid.
- Fast-dLLM confidence-threshold decoding (`threshold`); 0.8 is the demo default.
- Per-step `onStep` callback (`x`, `fresh`, `block`, `forward`) for animating the canvas.
- Published models: `naklitechie/Qwen3-0.6B-diffusion-mdlm-ONNX` and
  `naklitechie/Qwen3-0.6B-diffusion-bd3lm-ONNX`, each fused fp16 and RTN q4f16.
- `npm test`: 11 sampler tests against a fake ONNX session (Node >= 22, no install).
- Live demo with a 4-way model picker; `web/bench.html` (AR vs diffusion) and `web/probe.html`.

### Changed
- Default onnxruntime-web is stable 1.30.0; q4 no longer needs the 1.26.0-dev build.

### Fixed
- The yield `MessageChannel` holds a Node ref only while a yield is pending, so the module
  neither hangs nor exits early under Node.

[0.2.1]: https://github.com/NakliTechie/kohra/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/NakliTechie/kohra/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/NakliTechie/kohra/releases/tag/v0.1.0
