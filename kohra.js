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

import * as ortDefault from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.webgpu.min.mjs';
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
  threshold: null,       // Fast-dLLM: if set (e.g. 0.9), unmask ALL block positions with
                         // confidence >= threshold per step (min 1) instead of a fixed
                         // count → fewer forwards when the model is confident. null = use `steps`.
  blockCausal: false,    // bd3lm (block diffusion): pass a static [1,1,T,T] additive
                         // block-causal attention_mask each forward + denoise on the
                         // physical pos//blockSize grid. false = MDLM (single-input, no mask).
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
  // ref/unref exist only in Node: hold the port open while a yield is pending, so an
  // idle port never keeps the process alive and a pending one never lets it exit.
  ch.port1.onmessage = () => { queue.shift()?.(); if (!queue.length) ch.port1.unref?.(); };
  ch.port1.unref?.();
  return () => new Promise((resolve) => { queue.push(resolve); ch.port1.ref?.(); ch.port2.postMessage(0); });
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
    // A BD3LM KV-cache graph takes input_ids, position_ids and past_key_i/past_value_i,
    // and returns block logits plus present_key_i/present_value_i.
    this.pastNames = (session.inputNames ?? []).filter((n) => n.startsWith('past_'));
    this.presentNames = this.pastNames.map((n) => n.replace('past_', 'present_'));
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
    // KV-cache graphs (past_key_*/present_* I/O): keep outputs on the GPU so the cache
    // never crosses to JS between blocks. Logits are then read back with getData().
    kvCache = false,
    maskId = MASK_ID,
    eosId = EOS_ID,
  } = {}) {
    if (!model) throw new Error('kohra: `model` URL is required');
    if (!ort) ort = ortVersion
      ? await import(`https://cdn.jsdelivr.net/npm/onnxruntime-web@${ortVersion}/dist/ort.webgpu.min.mjs`)
      : ortDefault;
    const tk = await AutoTokenizer.from_pretrained(tokenizer ?? model);

    const opts = { executionProviders, graphOptimizationLevel };
    if (kvCache && executionProviders.includes('webgpu')) opts.preferredOutputLocation = 'gpu-buffer';
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

    // Block ranges to denoise, left-to-right. MDLM: blocks start at the prompt end
    // (P + b·blockSize). bd3lm: blocks live on the absolute pos//blockSize grid, so we
    // denoise from the physical block containing P — its prompt prefix is fixed, its
    // masked tail is the first generated tokens (bidirectional within that block).
    const blocks = [];
    if (cfg.blockCausal) {
      for (let blk = Math.floor(P / cfg.blockSize); blk * cfg.blockSize < T; blk++) {
        blocks.push([blk * cfg.blockSize, Math.min((blk + 1) * cfg.blockSize, T)]);
      }
    } else {
      const numBlocks = Math.ceil(cfg.maxNewTokens / cfg.blockSize);
      for (let b = 0; b < numBlocks; b++) {
        const start = P + b * cfg.blockSize;
        blocks.push([start, Math.min(start + cfg.blockSize, T)]);
      }
    }
    const numBlocks = blocks.length;
    const stepsPerBlock = Math.ceil(cfg.steps / numBlocks);

    // bd3lm's static block-causal additive mask: 0 where block(key) ≤ block(query),
    // else -1e9. Built once and fed every forward (the graph's 2nd input, fp32). The
    // mask never changes during sampling; block-causality is what makes the cache-free
    // full-canvas forward give correct current-block logits despite still-masked future
    // blocks (a future block can't influence an earlier one).
    // KV-cache path (BD3LM only): finished blocks never change under block-causal
    // attention, so each step runs just the current block against a cache of the earlier
    // ones. Exact, not approximate: scripts/gencheck_bd3lm_kv.py matches the cache-free
    // sampler token for token in fp32.
    const kv = cfg.blockCausal && this.pastNames.length > 0;
    let attnTensor = null;
    if (cfg.blockCausal && !kv) {
      const m = new Float32Array(T * T);
      for (let q = 0; q < T; q++) {
        const bq = Math.floor(q / cfg.blockSize);
        const row = q * T;
        for (let k = 0; k < T; k++) {
          m[row + k] = Math.floor(k / cfg.blockSize) <= bq ? 0 : -1e9;
        }
      }
      attnTensor = new ort.Tensor('float32', m, [1, 1, T, T]);
    }

    let forward = 0;
    const t0 = performance.now();

    const inBlockMasked = (start, end) => {
      let n = 0; for (let i = start; i < end; i++) if (x[i] === MASK) n++; return n;
    };

    // Cache helpers. `past` holds one tensor per past_* input; a run over [s, e) feeds
    // those tokens at their absolute positions.
    const blockFeeds = (s, e, past) => {
      const pos = new BigInt64Array(e - s);
      for (let i = 0; i < pos.length; i++) pos[i] = BigInt(s + i);
      const feeds = {
        input_ids: new ort.Tensor('int64', x.slice(s, e), [1, e - s]),
        position_ids: new ort.Tensor('int64', pos, [1, e - s]),
      };
      this.pastNames.forEach((n, i) => { feeds[n] = past[i]; });
      return feeds;
    };
    const extend = async (s, e, past) => {        // write [s, e) into the cache
      const out = await this.session.run(blockFeeds(s, e, past), this.presentNames);
      forward++;
      for (const t of past) t.dispose?.();
      return this.presentNames.map((n) => out[n]);
    };
    let past = null;
    if (kv) {
      const meta = (n) => this.session.inputMetadata?.find?.((m) => m.name === n);
      past = this.pastNames.map((n) => {
        const m = meta(n);
        const dims = [1, Number(m?.shape?.[1] ?? 8), 0, Number(m?.shape?.[3] ?? 128)];
        return m?.type === 'float32'
          ? new ort.Tensor('float32', new Float32Array(0), dims)
          : new ort.Tensor('float16', new Uint16Array(0), dims);
      });
      if (blocks[0][0] > 0) past = await extend(0, blocks[0][0], past);   // prefill
    }

    for (let b = 0; b < numBlocks; b++) {
      const [start, end] = blocks[b];
      const nMasked = inBlockMasked(start, end);
      const last = b === numBlocks - 1;
      if (nMasked === 0) {
        if (kv && !last) past = await extend(start, end, past);
        continue;
      }

      // Fixed-steps mode precomputes per-step reveal counts; threshold (Fast-dLLM) mode
      // loops until the block is clear, revealing however many clear the bar each step.
      const schedule = cfg.threshold == null ? transferSchedule(nMasked, stepsPerBlock) : null;
      let step = 0;

      while (inBlockMasked(start, end) > 0) {
        if (schedule && step >= schedule.length) break;
        let out, rowBase;                      // logits row for position p is p - rowBase
        if (kv) {
          out = await this.session.run(blockFeeds(start, end, past), ['logits']);
          rowBase = start;
        } else {
          const feeds = { input_ids: new ort.Tensor('int64', x.slice(), [1, T]) };
          if (attnTensor) feeds.attention_mask = attnTensor;
          out = await this.session.run(feeds);
          rowBase = 0;
        }
        forward++;
        const logits = out.logits.location === 'gpu-buffer'
          ? await out.logits.getData(true)     // download, then release the GPU buffer
          : out.logits.data;                   // Float32Array, [rows * V]
        const V = out.logits.dims[2];

        // Score every masked position in the current block. temperature 0 = plain argmax;
        // >0 = Gumbel-max (f64). Confidence is always softmax(logits)[token].
        const cand = [];
        for (let p = start; p < end; p++) {
          if (x[p] !== MASK) continue;
          const off = (p - rowBase) * V;
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

        // Pick which positions to commit this step.
        let commit;
        if (cfg.threshold == null) {
          cand.sort((a, b2) => b2.conf - a.conf);
          commit = cand.slice(0, schedule[step]);
        } else {
          commit = cand.filter((c) => c.conf >= cfg.threshold);
          if (commit.length === 0) {        // guarantee progress: take the single best
            let top = cand[0];
            for (const c of cand) if (c.conf > top.conf) top = c;
            commit = [top];
          }
        }
        step++;

        const fresh = new Set();
        for (const { p, tok } of commit) { x[p] = BigInt(tok); fresh.add(p); }

        cfg.onStep?.({
          x, P, fresh, block: b, numBlocks, forward,
          elapsedMs: performance.now() - t0,
        });
        if (cfg.yieldEvery && forward % cfg.yieldEvery === 0) {
          await yieldToEventLoop();
        }
      }
      // Commit: one clean forward writes the finished block into the cache. The last
      // block has no successor, so it skips this.
      if (kv && !last) past = await extend(start, end, past);
    }
    if (past) for (const t of past) t.dispose?.();

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
