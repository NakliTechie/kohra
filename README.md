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

## Use it — drop-in snippet (G1, working today)

`web/kohra.js` is a single-file, transformers.js-style ES module: import it, point it at the
ONNX graph, get diffusion text generation on WebGPU. It self-loads onnxruntime-web + the
tokenizer from a CDN, so this is the whole integration:

```js
import { pipeline } from './kohra.js';

const generate = await pipeline('text-diffusion', {
  model: 'models/qwen3-0.6b-mdlm-onnx/model_fp16_fused.onnx',  // fused fp16, 1.4GB
  tokenizer: 'dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1',
});

const { text, tokensPerSecond } = await generate('Lily runs 12 km/h for 4 hours. How far in 8 hours?');
console.log(text, `(${tokensPerSecond.toFixed(1)} tok/s)`);
```

Or the class form, with a per-step callback for the "fog lifting" visualization and sampler knobs:

```js
import { DiffusionLM } from './kohra.js';

const lm = await DiffusionLM.from_pretrained({ model, tokenizer });
const out = await lm.generate(prompt, {
  maxNewTokens: 128, steps: 128, blockSize: 32,
  temperature: 0,            // 0 = deterministic argmax; >0 = Gumbel sampling
  stripThink: true,          // drop the leading empty <think></think> block
  onStep: ({ x, P, fresh, forward }) => renderCanvas(x, P, fresh),
});
// out: { text, tokenIds, tokens, forwards, seconds, tokensPerSecond }
```

`web/index.html` is a ~60-line reference harness built entirely on this API. Serve the repo over
HTTP (`python3 -m http.server 8123`) and open `web/index.html`; first load downloads the model +
compiles WebGPU shaders. **Status:** fused-fp16 Qwen3-0.6B-MDLM runs at **~9.8 tok/s** on an M-series
Mac (128 denoise forwards, 1.4GB). The export recipe + WebGPU forensics are in
[`reference/MDLM-algorithm.md`](reference/MDLM-algorithm.md).

## Gate ladder

- **G1 — first coherent block in a browser. ✅ DONE** (fp32 then fused-fp16, ~9.8 tok/s). Target: [`dllm-collection/Qwen3-0.6B-diffusion-mdlm-v0.1`](https://huggingface.co/dllm-collection/Qwen3-0.6B-diffusion-mdlm-v0.1) (Tiny-A2D: Qwen3-0.6B adapted to masked diffusion). onnx-community's AR Qwen3-0.6B export is the conversion template. q4f16 (~500MB) is built and CPU-correct but blocked on a WebGPU `MatMulNBits` kernel bug (see gotchas). Sibling [bd3lm variant](https://huggingface.co/dllm-collection/Qwen3-0.6B-diffusion-bd3lm-v0.1) for block diffusion later.
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
- **fp16 on WebGPU needs the RMSNorm fused first.** A decomposed `Pow(x,2)` RMSNorm overflows native fp16 on WebGPU → silent all-zero logits (CPU/wasm reduce in fp32 and hide it). Fix: ORT's offline transformer optimizer (`model_type=qwen3`) fuses it to `SimplifiedLayerNormalization` before fp16-convert. See `reference/MDLM-algorithm.md`.
- **Dense q4 (`MatMulNBits`) miscomputes on WebGPU here.** On this Apple GPU + ORT-web 1.26.0, every q4f16 config (sym/asym, opt-level, scale-dtype, accuracy_level) decodes correctly on CPU but yields sane-magnitude-but-wrong logits on WebGPU. q4 is parked pending an upstream fix / newer ORT-web; fp16 ships.
- **Transformers.js `generate()` is AR-only** — bypass it; call the model's forward directly (or use a raw ORT-web session).
- **MDLM has no KV cache** (bidirectional attention, full forward per denoise step) — the perf profile is steps × block-length, not tokens. BD3LM restores block-level KV caching.
- **Watch item:** if Transformers.js ships native diffusion-loop support, fold into it rather than compete.

## Siblings

- **kiln** (`~/Code/kiln/`) — the MLC/WebLLM port track (compiler-level work). kohra is the ONNX + JS-loop track: no compiler, no TVM.
- **LocalMind** (`~/Code/naklios-universe/LocalMind/`) — the consumer surface, gate G4.
