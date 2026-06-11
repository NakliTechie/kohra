// kohra — masked-diffusion text generation in the browser, transformers.js-style.
//
// Drop-in usage (this is the whole snippet):
//
//   import { pipeline } from './kohra.js';
//   const gen = await pipeline('text-diffusion', {
//     model: '../models/qwen3-0.6b-mdlm-onnx/model_fp16_fused.onnx',
//     tokenizer: 'dllm-hub/Qwen3-0.6B-diffusion-mdlm-v0.1',
//   });
//   const { text } = await gen('Lily runs 12 km/h for 4 hours. How far in 8 hours?');
//
// Or the class form, with a per-step callback for the "fog lifting" visualization:
//
//   import { DiffusionLM } from './kohra.js';
//   const lm = await DiffusionLM.from_pretrained({ model, tokenizer });
//   const out = await lm.generate(prompt, { steps: 128, onStep: s => render(s) });
//
// The model graph is a loop-free Qwen3-MDLM ONNX export (input_ids -> logits over the
// full canvas); the whole denoising loop lives here in JS. Algorithm spec + the export
// recipe (fused fp16 for WebGPU) are in reference/MDLM-algorithm.md.

import * as ortDefault from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.26.0/dist/ort.webgpu.min.mjs';
import { AutoTokenizer } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4/+esm';

// Tiny-A2D Qwen3-0.6B MDLM token ids (tokenizer.mask_token_id / <|im_end|>).
export const MASK_ID = 151669n;
export const EOS_ID = 151645n;

const GEN_DEFAULTS = {
  maxNewTokens: 128,
  steps: 128,
  blockSize: 32,
  temperature: 0,        // 0 => deterministic argmax (the verified path)
  remasking: 'low_confidence', // or 'random'
  chatTemplate: true,    // wrap prompt in Qwen3 ChatML + generation prompt
  stripThink: false,     // drop a leading empty <think>…</think> block from the result
  onStep: null,          // ({ x, P, fresh, block, numBlocks, forward, elapsedMs }) => void
  yieldEvery: 1,         // await a frame every N forwards (keeps the UI responsive)
};

// Cooperative yield that releases the event loop so the UI can paint, WITHOUT blocking
// on the compositor. requestAnimationFrame stalls when the tab is hidden or the
// compositor isn't ticking (and would freeze the whole denoise loop); a MessageChannel
// macrotask always fires and isn't background-throttled like setTimeout.
const yieldToEventLoop = (() => {
  if (typeof MessageChannel === 'undefined') {
    return () => new Promise((resolve) => setTimeout(resolve, 0));
  }
  const ch = new MessageChannel();
  const queue = [];
  ch.port1.onmessage = () => { queue.shift()?.(); };
  return () => new Promise((resolve) => { queue.push(resolve); ch.port2.postMessage(0); });
})();

// Per-block reveal counts for the linear-alpha (MDLM) scheduler: deterministically
// round remaining/(steps-i) per step, skip zero entries. Mirrors dllm
// get_num_transfer_tokens and scripts/sample_onnx.py.
function transferSchedule(nMasked, steps) {
  const ks = [];
  let remaining = nMasked;
  for (let i = 0; i < steps && remaining > 0; i++) {
    const k = Math.min(Math.round(remaining / (steps - i)), remaining);
    if (k > 0) { ks.push(k); remaining -= k; }
  }
  return ks;
}

export class DiffusionLM {
  constructor({ session, tokenizer, ort, maskId = MASK_ID, eosId = EOS_ID }) {
    this.session = session;
    this.tokenizer = tokenizer;
    this.ort = ort;
    this.maskId = maskId;
    this.eosId = eosId;
  }

  // transformers.js-style loader. `model` is a URL to the .onnx graph; external data
  // (<name>.onnx.data / _data) is auto-detected. `tokenizer` is an HF id or local path.
  static async from_pretrained({
    model,
    tokenizer,
    ort,                       // explicit onnxruntime-web instance (optional)
    ortVersion,                // …or an npm version string to dynamic-import (optional)
    executionProviders = ['webgpu'],
    // The fused-fp16 graph keeps boundary casts that trip ORT-web's runtime
    // SimplifiedLayerNormFusion, so default to disabled (see reference doc).
    graphOptimizationLevel = 'disabled',
    maskId = MASK_ID,
    eosId = EOS_ID,
  } = {}) {
    if (!model) throw new Error('kohra: `model` URL is required');
    if (!ort) ort = ortVersion
      ? await import(`https://cdn.jsdelivr.net/npm/onnxruntime-web@${ortVersion}/dist/ort.webgpu.min.mjs`)
      : ortDefault;
    const tk = await AutoTokenizer.from_pretrained(tokenizer ?? model);

    const opts = { executionProviders, graphOptimizationLevel };
    // External-data path must match the location string recorded inside the proto,
    // which is the file's basename + suffix.
    const base = model.split('/').pop();
    for (const suffix of ['.data', '_data']) {
      try {
        const dataUrl = model + suffix;
        if ((await fetch(dataUrl, { method: 'HEAD' })).ok) {
          opts.externalData = [{ path: base + suffix, data: dataUrl }];
          break;
        }
      } catch { /* no external data at this suffix */ }
    }

    const session = await ort.InferenceSession.create(model, opts);
    return new DiffusionLM({ session, tokenizer: tk, ort, maskId, eosId });
  }

  // Build prompt token ids. transformers.js doesn't fetch chat_template.jinja for this
  // repo, so ChatML is constructed by hand (verified vs Python apply_chat_template).
  encodePrompt(text, chatTemplate = true) {
    const s = chatTemplate
      ? `<|im_start|>user\n${text}<|im_end|>\n<|im_start|>assistant\n`
      : text;
    return [...this.tokenizer(s, { add_special_tokens: false }).input_ids.data].map(Number);
  }

  decode(ids, { skip_special_tokens = true } = {}) {
    return this.tokenizer.decode(ids.map(Number), { skip_special_tokens });
  }

  // Run the masked-diffusion denoising loop. `prompt` is a string or an array of token
  // ids. Returns { text, tokenIds, tokens, forwards, seconds, tokensPerSecond, x, P }.
  async generate(prompt, options = {}) {
    const cfg = { ...GEN_DEFAULTS, ...options };
    const ort = this.ort;
    const MASK = this.maskId, EOS = this.eosId;

    const promptIds = Array.isArray(prompt)
      ? prompt.map(Number)
      : this.encodePrompt(prompt, cfg.chatTemplate);
    const P = promptIds.length;
    const T = P + cfg.maxNewTokens;

    const x = new BigInt64Array(T).fill(EOS);
    promptIds.forEach((t, i) => { x[i] = BigInt(t); });
    for (let i = P; i < T; i++) x[i] = MASK;

    const numBlocks = Math.ceil(cfg.maxNewTokens / cfg.blockSize);
    const stepsPerBlock = Math.ceil(cfg.steps / numBlocks);

    let forward = 0;
    const t0 = performance.now();

    for (let b = 0; b < numBlocks; b++) {
      const start = P + b * cfg.blockSize;
      const end = Math.min(start + cfg.blockSize, T);
      let nMasked = 0;
      for (let i = start; i < end; i++) if (x[i] === MASK) nMasked++;

      for (const k of transferSchedule(nMasked, stepsPerBlock)) {
        const out = await this.session.run({
          input_ids: new ort.Tensor('int64', x.slice(), [1, T]),
        });
        forward++;
        const logits = out.logits.data;       // Float32Array, [T * V] (ORT upcasts fp16)
        const V = out.logits.dims[2];

        // Score every masked position in the current block, then commit the top-k by
        // confidence. temperature 0 = plain argmax; >0 = Gumbel-max (reference does this
        // in f64; JS numbers are f64). Confidence is always softmax(logits)[token].
        const cand = [];
        for (let p = start; p < end; p++) {
          if (x[p] !== MASK) continue;
          const off = p * V;
          let maxL = -Infinity;
          for (let v = 0; v < V; v++) { const l = logits[off + v]; if (l > maxL) maxL = l; }
          let best = 0, bestScore = -Infinity, sumExp = 0;
          for (let v = 0; v < V; v++) {
            const l = logits[off + v];
            sumExp += Math.exp(l - maxL);
            const score = cfg.temperature > 0
              ? l - cfg.temperature * Math.log(-Math.log(Math.random() + 1e-12) + 1e-12)
              : l;
            if (score > bestScore) { bestScore = score; best = v; }
          }
          const conf = cfg.remasking === 'random'
            ? Math.random()
            : Math.exp(logits[off + best] - maxL) / sumExp;
          cand.push({ p, tok: best, conf });
        }

        cand.sort((a, b2) => b2.conf - a.conf);
        const fresh = new Set();
        for (const { p, tok } of cand.slice(0, k)) { x[p] = BigInt(tok); fresh.add(p); }

        cfg.onStep?.({
          x, P, fresh, block: b, numBlocks, forward,
          elapsedMs: performance.now() - t0,
        });
        if (cfg.yieldEvery && forward % cfg.yieldEvery === 0) {
          await yieldToEventLoop();
        }
      }
    }

    const seconds = (performance.now() - t0) / 1000;

    // Trim at the first EOS after the prompt.
    let genEnd = T;
    for (let i = P; i < T; i++) if (x[i] === EOS) { genEnd = i; break; }
    const tokenIds = [...x.slice(P, genEnd)].map(Number);
    let text = this.decode(tokenIds, { skip_special_tokens: true });
    if (cfg.stripThink) text = stripThinkBlock(text);

    return {
      text,
      tokenIds,
      tokens: tokenIds.length,
      forwards: forward,
      seconds,
      tokensPerSecond: tokenIds.length / seconds,
      x,
      P,
    };
  }
}

// Drop a leading Qwen3 <think>…</think> block (often empty) from generated text.
function stripThinkBlock(text) {
  return text.replace(/^\s*<think>[\s\S]*?<\/think>\s*/, '');
}

// transformers.js-style convenience factory.
export async function pipeline(task, options = {}) {
  if (task !== 'text-diffusion') {
    throw new Error(`kohra: unsupported task '${task}' (only 'text-diffusion')`);
  }
  const lm = await DiffusionLM.from_pretrained(options);
  const fn = (prompt, genOptions) => lm.generate(prompt, genOptions);
  fn.model = lm;
  return fn;
}

export default DiffusionLM;
