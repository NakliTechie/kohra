# Changelog

All notable changes to kohra are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow [SemVer](https://semver.org/).

## [0.4.0] — 2026-10-30

### Added
- `denoise.js`: the block-diffusion decoding loop as a module with no runtime inside. The caller
  supplies one `forward({ mode: 'preview' | 'commit', start, ids, wantLogits })` callback.
  diskformer's `clover_moe` engine drives it for Clover-35B-A3B-Preview.
- Clover's decoding rules as options and `PRESETS.clover`: strict `> 0.95` threshold, mask logit
  suppressed, 32-preview limit per block, response-only presence / frequency / repetition
  penalties with the length and think-end ramps (fp32-exact against `CloverPenaltyState.apply`),
  stop after a resolved block, full last block trimmed to `maxNewTokens`.
- Optional `top1` preview fast path for greedy, penalty-free runs; falls back to full logits.
- `onBlock({ tokens })` event after each resolved block, for streaming.

### Changed
- `DiffusionLM.generate` runs on `denoise.js`. Same tokens, forwards, session calls and step
  events as 0.3.1 on 240 fake-session configurations.

## [0.3.1] — 2026-10-24

### Added
- `from_pretrained({ progress_callback })`: transformers.js-style `initiate` / `progress` /
  `done` events for each graph file while it downloads.
- Graph files are kept in Cache Storage `kohra-models` (`cacheName` option; `null` turns it
  off), so a reload reads them from disk. Nemotron 3B: first load downloads 2.7 GB, a reload
  is ready in about 7 s with no download.
- `fetchCached(url, { cache, progress_callback })` is exported; 4 download tests (17 total).

## [0.3.0] — 2026-10-24

### Added
- Nemotron-Labs-Diffusion-3B support: a "seeded" block-KV graph takes an additive
  `attention_mask` input, so one graph both denoises a block (no mask) and commits it to the
  cache (causal mask); the commit's last logit seeds the next block's first token. kohra.js
  detects such a graph and stops after a block that contains EOS.
- `chatFormat` option on `from_pretrained` to wrap prompts in a model's chat template.
- Multi-file external data: `<model>.onnx_data`, `_data_1`, … are found and loaded.
- Demo picker entry "Nemotron 3B · q4 · large (2.7 GB)" (`?arch=nemotron-q4`), loading
  `naklitechie/Nemotron-Labs-Diffusion-3B-ONNX` from Hugging Face.
- `scripts/export_nemotron_kv.py` (fp16 export, q4 RTN), `scripts/gencheck_nemotron.py`,
  `scripts/push_nemotron_hf.py`, `web/nemotron.html`; NVIDIA licence in `reference/nemotron/`.

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

[0.4.0]: https://github.com/NakliTechie/kohra/compare/v0.3.1...v0.4.0
[0.3.1]: https://github.com/NakliTechie/kohra/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/NakliTechie/kohra/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/NakliTechie/kohra/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/NakliTechie/kohra/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/NakliTechie/kohra/releases/tag/v0.1.0
