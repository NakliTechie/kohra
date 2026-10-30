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
import { denoise } from './denoise.js';
import { AutoTokenizer } from 'https://cdn.jsdelivr.net/npm/@huggingface/transformers@4/+esm';

// Cache Storage name for downloaded graph files (from_pretrained's `cacheName`).
export const CACHE_NAME = 'kohra-models';

// Fetch `url` as bytes. With a Cache Storage `cache`, a hit is read from disk and a miss is
// stored there after download. Reports transformers.js-style progress events either way.
export async function fetchCached(url, { cache = null, progress_callback = null } = {}) {
  const file = url.split('/').pop();
  const emit = (e) => progress_callback?.({ file, name: url, ...e });
  emit({ status: 'initiate' });
  let res = cache ? await cache.match(url) : null;
  const hit = !!res;
  if (!res) {
    res = await fetch(url);
    if (!res.ok) throw new Error(`kohra: ${res.status} fetching ${url}`);
  }
  const total = Number(res.headers.get('content-length')) || 0;
  // Write straight into one buffer when the size is known (2 GB files: no second copy);
  // otherwise, or once more bytes arrive than announced, collect chunks and join at the end.
  const bytes = new Uint8Array(total);
  const chunks = [];
  let loaded = 0, filled = 0;
  let body = res.body, stored = null;
  if (cache && !hit) {      // one branch to Cache Storage, one read here; no second copy
    const [mine, theirs] = body.tee();
    body = mine;
    stored = cache.put(url, new Response(theirs, total ? { headers: { 'content-length': String(total) } } : {}))
      .catch(() => {});     // quota full: the model still loads, it just downloads again next time
  }
  const reader = body.getReader();
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    if (!chunks.length && filled + value.length <= total) { bytes.set(value, filled); filled += value.length; }
    else chunks.push(value);
    loaded += value.length;
    emit({ status: 'progress', loaded, total, progress: total ? Math.min(100, (100 * loaded) / total) : 0 });
  }
  let out = bytes;
  if (loaded !== total) {
    out = new Uint8Array(loaded);
    out.set(bytes.subarray(0, filled), 0);
    let o = filled;
    for (const c of chunks) { out.set(c, o); o += c.length; }
  }
  await stored;
  emit({ status: 'done', loaded, total: loaded });
  return out;
}

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

export class DiffusionLM {
  constructor({ session, tokenizer, ort, maskId = MASK_ID, eosId = EOS_ID, chatFormat = null }) {
    this.session = session;
    this.tokenizer = tokenizer;
    this.ort = ort;
    this.maskId = maskId;
    this.eosId = eosId;
    this.chatFormat = chatFormat;
    // A BD3LM KV-cache graph takes input_ids, position_ids and past_key_i/past_value_i,
    // and returns block logits plus present_key_i/present_value_i.
    this.pastNames = (session.inputNames ?? []).filter((n) => n.startsWith('past_'));
    this.presentNames = this.pastNames.map((n) => n.replace('past_', 'present_'));
    // A seeded KV graph (Nemotron-Labs-Diffusion) also takes attention_mask: zeros while
    // denoising a block, causal for the commit pass, whose last logit seeds the next block.
    this.seeded = this.pastNames.length > 0 && (session.inputNames ?? []).includes('attention_mask');
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
    chatFormat = null,         // (text) => prompt string; null = Qwen3 ChatML
    // transformers.js-style progress events while the graph files download:
    // { status: 'initiate' | 'progress' | 'done', file, loaded, total, progress }.
    progress_callback = null,
    // Cache Storage name for the graph files, so a reload reads them from disk instead of
    // downloading again. null = no Cache Storage (the browser's HTTP cache only).
    cacheName = CACHE_NAME,
  } = {}) {
    if (!model) throw new Error('kohra: `model` URL is required');
    if (!ort) ort = ortVersion
      ? await import(`https://cdn.jsdelivr.net/npm/onnxruntime-web@${ortVersion}/dist/ort.webgpu.min.mjs`)
      : ortDefault;
    const tk = await AutoTokenizer.from_pretrained(tokenizer ?? model);

    const opts = { executionProviders, graphOptimizationLevel };
    if (kvCache && executionProviders.includes('webgpu')) opts.preferredOutputLocation = 'gpu-buffer';
    const cache = cacheName && globalThis.caches ? await caches.open(cacheName) : null;
    // External-data path must match the location string recorded inside the proto,
    // which is the file's basename + suffix.
    const base = model.split('/').pop();
    // Large graphs split their data into <name>_data, <name>_data_1, … (≤ 2 GB each).
    const exists = async (url) => {
      if (cache && await cache.match(url)) return true;
      try { return (await fetch(url, { method: 'HEAD' })).ok; } catch { return false; }
    };
    const files = [];                 // [path recorded in the proto, url]
    for (const suffix of ['.data', '_data']) {
      if (!(await exists(model + suffix))) continue;
      files.push([base + suffix, model + suffix]);
      for (let i = 1; await exists(`${model}${suffix}_${i}`); i++) {
        files.push([`${base}${suffix}_${i}`, `${model}${suffix}_${i}`]);
      }
      break;
    }

    const get = (url) => fetchCached(url, { cache, progress_callback });
    if (files.length) {
      opts.externalData = [];
      for (const [path, url] of files) opts.externalData.push({ path, data: await get(url) });
    }
    const session = await ort.InferenceSession.create(await get(model), opts);
    return new DiffusionLM({ session, tokenizer: tk, ort, maskId, eosId, chatFormat });
  }

  // Build prompt token ids. transformers.js doesn't fetch chat_template.jinja for this
  // repo, so ChatML is constructed by hand (verified vs Python apply_chat_template).
  encodePrompt(text, chatTemplate = true) {
    const s = !chatTemplate ? text
      : this.chatFormat ? this.chatFormat(text)
      : `<|im_start|>user\n${text}<|im_end|>\n<|im_start|>assistant\n`;
    return [...this.tokenizer(s, { add_special_tokens: false }).input_ids.data].map(Number);
  }

  decode(ids, { skip_special_tokens = true } = {}) {
    return this.tokenizer.decode(ids.map(Number), { skip_special_tokens });
  }

  // Run the block-diffusion denoising loop (denoise.js) over this ONNX session. `prompt` is
  // a string or an array of token ids. Returns { text, tokenIds, tokens, forwards, seconds,
  // tokensPerSecond, x, P }.
  async generate(prompt, options = {}) {
    const cfg = { ...GEN_DEFAULTS, ...options };
    const ort = this.ort;
    const MASK = this.maskId, EOS = this.eosId;

    const promptIds = Array.isArray(prompt)
      ? prompt.map(Number)
      : this.encodePrompt(prompt, cfg.chatTemplate);
    const P = promptIds.length;
    const T = P + cfg.maxNewTokens;

    // `x` is the BigInt canvas callers see in onStep and the result; denoise.js works on
    // its own Int32Array canvas and this mirrors it.
    const x = new BigInt64Array(T).fill(EOS);
    promptIds.forEach((t, i) => { x[i] = BigInt(t); });
    for (let i = P; i < T; i++) x[i] = MASK;
    const sync = (canvas, s, e) => { for (let i = s; i < e; i++) x[i] = BigInt(canvas[i]); };

    // KV-cache path (BD3LM and seeded graphs): finished blocks never change under
    // block-causal attention, so each preview runs just the current block against a cache
    // of the earlier ones. Exact, not approximate: scripts/gencheck_bd3lm_kv.py matches the
    // cache-free sampler token for token in fp32. Without a cache, every preview runs the
    // full canvas and commits run nothing.
    const kv = (cfg.blockCausal || this.seeded) && this.pastNames.length > 0;
    let forwards = 0;

    // bd3lm cache-free: a static block-causal additive mask, 0 where block(key) ≤
    // block(query), else -1e9. Block-causality is what makes the full-canvas forward give
    // correct current-block logits despite still-masked future blocks.
    let attnTensor = null;
    if (cfg.blockCausal && !kv) {
      const m = new Float32Array(T * T);
      for (let q = 0; q < T; q++) {
        const bq = Math.floor(q / cfg.blockSize);
        const row = q * T;
        for (let k = 0; k < T; k++) m[row + k] = Math.floor(k / cfg.blockSize) <= bq ? 0 : -1e9;
      }
      attnTensor = new ort.Tensor('float32', m, [1, 1, T, T]);
    }

    const readLogits = async (t) => (t.location === 'gpu-buffer'
      ? await t.getData(true)          // download, then release the GPU buffer
      : t.data);

    // A run over ids at [s, s + L) against a cache covering [0, s). A seeded graph's mask
    // is [1, 1, L, s + L]: zeros to denoise, causal to commit.
    const blockFeeds = (s, ids, past, causal) => {
      const L = ids.length, e = s + L;
      const pos = new BigInt64Array(L);
      for (let i = 0; i < L; i++) pos[i] = BigInt(s + i);
      const feeds = {
        input_ids: new ort.Tensor('int64', BigInt64Array.from(ids, BigInt), [1, L]),
        position_ids: new ort.Tensor('int64', pos, [1, L]),
      };
      if (this.seeded) {
        const m = new Float32Array(L * e);
        if (causal) for (let q = 0; q < L; q++) m.fill(-1e9, q * e + s + q + 1, (q + 1) * e);
        feeds.attention_mask = new ort.Tensor('float32', m, [1, 1, L, e]);
      }
      this.pastNames.forEach((n, i) => { feeds[n] = past[i]; });
      return feeds;
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
    }
    const canvasX = new BigInt64Array(T);   // cache-free path: the full canvas each preview
    canvasX.set(x);

    // The forward() callback denoise.js drives (see the contract at the top of denoise.js).
    const forward = async ({ mode, start, ids, wantLogits }) => {
      if (!kv) {
        for (let i = 0; i < ids.length; i++) canvasX[start + i] = BigInt(ids[i]);
        if (mode === 'commit') return null;
        const feeds = { input_ids: new ort.Tensor('int64', canvasX.slice(), [1, T]) };
        if (attnTensor) feeds.attention_mask = attnTensor;
        const out = await this.session.run(feeds);
        forwards++;
        const V = out.logits.dims[2];
        return (await readLogits(out.logits)).slice(start * V, (start + ids.length) * V);
      }
      if (mode === 'preview') {
        const out = await this.session.run(blockFeeds(start, ids, past, false), ['logits']);
        forwards++;
        return readLogits(out.logits);
      }
      // Commit: write the block into the cache. A seeded graph runs it causally; its last
      // logit picks the next block's first token.
      const names = this.seeded ? ['logits', ...this.presentNames] : this.presentNames;
      const out = await this.session.run(blockFeeds(start, ids, past, this.seeded), names);
      forwards++;
      for (const t of past) t.dispose?.();
      past = this.presentNames.map((n) => out[n]);
      if (wantLogits !== 'last') { out.logits?.dispose?.(); return null; }
      const V = out.logits.dims[2];
      return (await readLogits(out.logits)).slice((ids.length - 1) * V, ids.length * V);
    };

    let res;
    try {
      res = await denoise({
        forward,
        promptIds,
        maskId: Number(MASK),
        eosIds: [Number(EOS)],
        blockSize: cfg.blockSize,
        maxNewTokens: cfg.maxNewTokens,
        threshold: cfg.threshold,
        steps: cfg.steps,
        temperature: cfg.temperature,
        remasking: cfg.remasking,
        seed: this.seeded,
        alignToGrid: cfg.blockCausal && !this.seeded,
        stopAtEos: this.seeded,              // as upstream generate does
        yieldEvery: cfg.yieldEvery,
        onStep: cfg.onStep && ((s) => {
          sync(s.canvas, s.start, s.end);
          cfg.onStep({ x, P, fresh: s.fresh, block: s.block, numBlocks: s.numBlocks, forward: forwards, elapsedMs: s.elapsedMs });
        }),
      });
    } finally {
      if (past) for (const t of past) t.dispose?.();
    }
    sync(res.canvas, 0, T);

    const tokenIds = res.tokenIds;
    let text = this.decode(tokenIds, { skip_special_tokens: true });
    if (cfg.stripThink) text = stripThinkBlock(text);

    return {
      text,
      tokenIds,
      tokens: tokenIds.length,
      forwards,
      seconds: res.seconds,
      tokensPerSecond: tokenIds.length / res.seconds,
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
