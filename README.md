# kohra

> Text-diffusion language models running in the browser — a JS denoising loop over ONNX Runtime Web / WebGPU. To our knowledge, the first in-browser *text*-diffusion generation.

कोहरा — *fog*. Diffusion generation starts as noise and denoises, pass by pass, into clear text.

**[Try it live](https://naklitechie.github.io/kohra)** (needs a WebGPU browser; first load pulls ~1.4GB) · also on [Hugging Face Spaces](https://huggingface.co/spaces/naklitechie/kohra) · model: [naklitechie/Qwen3-0.6B-diffusion-mdlm-ONNX](https://huggingface.co/naklitechie/Qwen3-0.6B-diffusion-mdlm-ONNX)

## Why

Google's [DiffusionGemma](https://huggingface.co/google/diffusiongemma-26B-A4B-it) (weights launched 2026-06-10, Apache 2.0) put text diffusion on the map: instead of token-by-token autoregression, the model generates whole 256-token blocks in parallel via iterative denoising (~48 steps) — up to 4× faster on GPUs (1000+ tok/s on an H100). But it's a 26B-A4B MoE (~18 GB quantized — past browser physics), and **no browser runtime anywhere supports a diffusion generation loop** — Transformers.js, onnxruntime-web, and MLC/WebLLM are all autoregressive-only.

The irony: parallel block-denoising plays to WebGPU batch throughput *exactly* where sequential AR decode is the browser's bottleneck. Diffusion should eventually be a better fit for in-browser inference than AR is. kohra builds the missing stack.

## The thesis

Two missing pieces, built as one vertically-sliced project (they're only testable against each other):

1. **ONNX exports of small diffusion LMs** — the riskier half (custom arch configs, MoE quantization constraints).
2. **A JS denoising/sampling loop** over raw ONNX forward passes — the easier half (~few hundred lines: start masked → forward → lock high-confidence tokens → repeat).

## Use it in a browser (G1, working today)

`kohra.js` is a single-file, transformers.js-style ES module: import it, point it at an ONNX
graph, get masked-diffusion text generation on WebGPU. It self-loads onnxruntime-web (from a CDN)
and the tokenizer (from Hugging Face), so there's no build step and no server-side inference.

The fp16 model is published at
[**naklitechie/Qwen3-0.6B-diffusion-mdlm-ONNX**](https://huggingface.co/naklitechie/Qwen3-0.6B-diffusion-mdlm-ONNX).

### 1. Get the loader

`kohra.js` is one file with no bundler dependency. Copy it into your project:

```sh
# from this repo
cp kohra.js  your-app/kohra.js
# or grab it straight from the model repo
curl -O https://huggingface.co/naklitechie/Qwen3-0.6B-diffusion-mdlm-ONNX/resolve/main/kohra.js
```

(A jsDelivr/npm import will be the one-liner once the package is published; for now copy the file —
importing JS from a Hugging Face `resolve/` URL is blocked by MIME type, but the **model** loads
from there fine.)

### 2. Drop it in — a complete, working page

```html
<!doctype html>
<meta charset="utf-8">
<button id="go">Generate</button>
<pre id="out"></pre>
<script type="module">
import { pipeline } from './kohra.js';

const HF = 'https://huggingface.co/naklitechie/Qwen3-0.6B-diffusion-mdlm-ONNX/resolve/main';
const generate = await pipeline('text-diffusion', {
  model: `${HF}/onnx/model_fp16_fused.onnx`,           // fp16, ~1.4 GB (browser-cached)
  tokenizer: 'naklitechie/Qwen3-0.6B-diffusion-mdlm-ONNX',
});

document.getElementById('go').onclick = async () => {
  const { text, tokensPerSecond } = await generate('Explain WebGPU in one sentence.', {
    maxNewTokens: 128, steps: 128, stripThink: true,
  });
  document.getElementById('out').textContent = `${text}\n\n(${tokensPerSecond.toFixed(1)} tok/s)`;
};
</script>
```

Serve it over **https or localhost** (WebGPU and ES modules both require a secure context):
`python3 -m http.server 8000` then open `http://localhost:8000`. First run downloads the model and
compiles WebGPU shaders (tens of seconds); it's cached afterward.

### 3. Stream the denoising for a "fog lifting" UI

The class form exposes a per-step callback and the sampler knobs:

```js
import { DiffusionLM } from './kohra.js';

const lm = await DiffusionLM.from_pretrained({ model, tokenizer });
const out = await lm.generate(prompt, {
  maxNewTokens: 128, steps: 128, blockSize: 32,
  temperature: 0,            // 0 = deterministic argmax; >0 = Gumbel sampling
  stripThink: true,          // drop the leading empty <think></think> block
  onStep: ({ x, P, fresh, forward }) => render(x, P, fresh),  // canvas snapshot per reveal
});
// out: { text, tokenIds, tokens, forwards, seconds, tokensPerSecond }
```

`x` is the full token canvas (masked positions are `lm.maskId`); `fresh` is the set of positions
revealed this step — colour those to animate the fog lifting. `index.html` is a ~60-line
reference harness built entirely on this API (it's the live demo).

### Requirements & notes

- **Browser:** Chrome/Edge 121+ (WebGPU + fp16). No WebGPU → it won't run; there's no wasm fallback wired up.
- **Hosting your own model:** any URL that serves the `.onnx` and its `.onnx.data` side-by-side with
  permissive CORS works (Hugging Face `resolve/` URLs do). kohra auto-detects the external-data file.
- **Perf:** fused-fp16 Qwen3-0.6B-MDLM runs at **~9.8 tok/s** on an M-series Mac (128 denoise
  forwards). No KV cache — cost is steps × forward, not tokens. The export recipe + WebGPU forensics
  (why fp16 needs RMSNorm fused first) are in [`reference/MDLM-algorithm.md`](reference/MDLM-algorithm.md).

## Gate ladder

- **G1 — first coherent block in a browser. ✅ DONE** (fp32 then fused-fp16, ~9.8 tok/s). Target: [`dllm-collection/Qwen3-0.6B-diffusion-mdlm-v0.1`](https://huggingface.co/dllm-collection/Qwen3-0.6B-diffusion-mdlm-v0.1) (Tiny-A2D: Qwen3-0.6B adapted to masked diffusion). onnx-community's AR Qwen3-0.6B export is the conversion template. q4f16 (~500MB) is built and CPU-correct but blocked on a WebGPU `MatMulNBits` kernel bug (see gotchas). Sibling [bd3lm variant](https://huggingface.co/dllm-collection/Qwen3-0.6B-diffusion-bd3lm-v0.1) for block diffusion later.
- **G2 — sampler quality + perf.** Confidence-threshold decoding (Fast-dLLM), step/block schedules. Benchmark AR Qwen3-0.6B vs diffusion Qwen3-0.6B **in the same browser** — same weight lineage, clean A/B of the two generation paradigms. This comparison is a publishable artifact on its own.
- **G3 — a genuinely useful model.** [`inclusionAI/LLaDA-MoE-7B-A1B-Instruct`](https://huggingface.co/inclusionAI/LLaDA-MoE-7B-A1B-Instruct) (+ `-Instruct-TD`, trajectory-distilled for fewer denoise steps). First open MoE diffusion LM: 7B total / 1.4B active, quality ≈ Qwen2.5-3B-Instruct. Size class already proven in-browser by LFM2-8B-A1B.
- **G4 — LocalMind integration** via its runtime-adapter `MODELS` pattern.
- **North star — DiffusionGemma** ([`google/diffusiongemma-26B-A4B-it`](https://huggingface.co/google/diffusiongemma-26B-A4B-it), Apache 2.0). **Weights launched 2026-06-10** (Gemma-4 backbone, 26B total / 3.8B active MoE, multimodal, 256K context; GGUF/MLX/vLLM out). 26B total (~18GB q4) is past browser physics — desktop play now actionable via MLX 4-bit on a big-RAM Mac. The browser angle is a future small/distilled diffusion-Gemma variant (watch [Gemma 4 E2B/E4B](https://huggingface.co/google/gemma-4-E2B) — those small sizes are AR today, but a diffusion variant at that scale would be the browser-feasible ambitious target).

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
