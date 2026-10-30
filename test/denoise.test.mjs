// denoise.js tests: the runtime-free loop against a scripted forward(), with Clover's rules.
// Run: npm test
import { test as nodeTest } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { denoise, applyPenalties, PRESETS } from '../denoise.js';

const test = (name, fn) => nodeTest(name, { timeout: 10_000 }, fn);

const V = 40, MASK = 39, EOS = 38;
const want = (p) => p % 20;

// Scripted runtime: preview row i scores slot start + i (same-position logits).
// `conf(p, step)` sets the softmax probability of want(p) at that preview.
function runtime({ conf = () => 0.99, eosAt = null, top1 = false } = {}) {
  const calls = [];
  const cache = [];
  let step = 0;
  const forward = async ({ mode, start, ids, wantLogits }) => {
    calls.push({ mode, start, n: ids.length, wantLogits, ids: [...ids] });
    if (mode === 'commit') {
      assert.equal(start, cache.length, 'commit start equals the cache length');
      cache.push(...ids);
      return null;
    }
    if (wantLogits === 'top1') {
      if (!top1) throw new Error('top1 unsupported');
      const token = new Int32Array(ids.length), c = new Float32Array(ids.length);
      ids.forEach((_, i) => { token[i] = start + i === eosAt ? EOS : want(start + i); c[i] = conf(start + i, step); });
      step++;
      return { token, conf: c };
    }
    const out = new Float32Array(ids.length * V);
    ids.forEach((_, i) => {
      const p = start + i, t = p === eosAt ? EOS : want(p), q = conf(p, step);
      // softmax over V: target gets q, the rest share 1 - q evenly (mask included).
      const rest = Math.log((1 - q) / (V - 1)), hit = Math.log(q);
      for (let v = 0; v < V; v++) out[i * V + v] = v === t ? hit : rest;
    });
    step++;
    return out;
  };
  return { forward, calls, cache };
}

const PROMPT = [1, 2, 3, 4, 5, 6, 7];
const clover = { ...PRESETS.clover, maskId: MASK, eosIds: [EOS], penalties: null };

test('clover rules: prefill commit, 32-slot previews, commit each block but the last, full last block trimmed', async () => {
  const r = runtime();
  const out = await denoise({ forward: r.forward, promptIds: PROMPT, ...clover, maxNewTokens: 70 });
  assert.deepEqual(out.tokenIds, Array.from({ length: 70 }, (_, i) => want(7 + i)));
  assert.deepEqual(r.calls.filter((c) => c.mode === 'commit').map((c) => [c.start, c.n, c.wantLogits]),
    [[0, 7, 'none'], [7, 32, 'none'], [39, 32, 'none']]);
  const previews = r.calls.filter((c) => c.mode === 'preview');
  assert.deepEqual(previews.map((c) => [c.start, c.n]), [[7, 32], [39, 32], [71, 32]]);
  assert.ok(previews.every((c) => c.ids.every((t) => t === MASK)), 'Clover blocks start fully masked');
});

test('strict threshold: confidence equal to the bar does not fill', async () => {
  const at = (p, step) => (p % 2 === 0 ? 0.95 : 0.96);   // even slots sit exactly on the bar
  // top1 reports confidences exactly, so 0.95 is exactly on the bar.
  const run = (strict) => denoise({ forward: runtime({ conf: (p) => Math.fround(at(p)), top1: true }).forward,
    promptIds: PROMPT, ...clover, threshold: Math.fround(0.95), strict, top1: true, maxNewTokens: 32 });
  const out = await run(true);
  assert.ok(out.previews > 1, 'the >= rule would clear the block in one preview');
  const loose = await run(false);
  assert.equal(loose.previews, 1);
});

test('progress floor: below the bar, one slot per preview, most confident first', async () => {
  const order = [];
  const conf = (p) => 0.1 + (p % 32) / 100;
  await denoise({ forward: runtime({ conf }).forward, promptIds: PROMPT, ...clover, maxNewTokens: 32,
    onStep: ({ fresh }) => order.push(...fresh) });
  assert.equal(order.length, 32);
  const c = order.map(conf);
  for (let i = 1; i < c.length; i++) assert.ok(c[i] <= c[i - 1]);
});

test('step limit: masks left after maxStepsPerBlock previews throw', async () => {
  await assert.rejects(
    denoise({ forward: runtime({ conf: () => 0.1 }).forward, promptIds: PROMPT, ...clover, maxStepsPerBlock: 5, maxNewTokens: 32 }),
    /still has masks after 5 previews/);
});

test('stop: EOS ends the run after its block resolves, with no further commit', async () => {
  const r = runtime({ eosAt: 7 + 40 });
  const blocks = [];
  const out = await denoise({ forward: r.forward, promptIds: PROMPT, ...clover, maxNewTokens: 128,
    onBlock: ({ tokens }) => blocks.push(tokens.length) });
  assert.deepEqual(out.tokenIds, Array.from({ length: 40 }, (_, i) => want(7 + i)));
  assert.equal(out.stopToken, EOS);
  assert.deepEqual(blocks, [32, 8]);
  assert.deepEqual(r.calls.filter((c) => c.mode === 'commit').map((c) => c.start), [0, 7]);
});

test('suppressMask: the mask token is never chosen', async () => {
  const forward = async ({ mode, ids }) => {
    if (mode === 'commit') return null;
    const out = new Float32Array(ids.length * V);
    for (let i = 0; i < ids.length; i++) { out[i * V + MASK] = 50; out[i * V + 3] = 10; }
    return out;
  };
  const out = await denoise({ forward, promptIds: PROMPT, ...clover, maxNewTokens: 32 });
  assert.deepEqual(out.tokenIds, Array(32).fill(3));
});

test('seed: a commit asks for its last logit and the argmax fills the next block\'s first slot', async () => {
  const calls = [];
  const forward = async ({ mode, start, ids, wantLogits }) => {
    calls.push([mode, start, wantLogits, ids[0]]);
    if (mode === 'commit') { const l = new Float32Array(V); l[17] = 5; return wantLogits === 'last' ? l : null; }
    const out = new Float32Array(ids.length * V);
    for (let i = 0; i < ids.length; i++) out[i * V + 4] = 20;
    return out;
  };
  const out = await denoise({ forward, promptIds: PROMPT, maskId: MASK, eosIds: [EOS], threshold: 0.9, seed: true, maxNewTokens: 64 });
  assert.deepEqual(calls.filter((c) => c[0] === 'commit').map((c) => c.slice(0, 3)), [['commit', 0, 'last'], ['commit', 7, 'last']]);
  assert.ok(calls.filter((c) => c[0] === 'preview').every((c) => c[3] === 17), 'each block starts from its seed');
  assert.equal(out.tokenIds[0], 17);
  assert.equal(out.tokenIds[32], 17);
});

test('top1: used when greedy and penalty-free, with the same tokens as the full-logits path', async () => {
  const fast = runtime({ top1: true });
  const a = await denoise({ forward: fast.forward, promptIds: PROMPT, ...clover, top1: true, maxNewTokens: 64 });
  assert.ok(fast.calls.filter((c) => c.mode === 'preview').every((c) => c.wantLogits === 'top1'));
  const b = await denoise({ forward: runtime().forward, promptIds: PROMPT, ...clover, maxNewTokens: 64 });
  assert.deepEqual(a.tokenIds, b.tokenIds);
});

test('top1: a runtime without it throws once, then the loop stays on full logits', async () => {
  const r = runtime({ top1: false });
  const out = await denoise({ forward: r.forward, promptIds: PROMPT, ...clover, top1: true, maxNewTokens: 64 });
  assert.equal(out.tokenIds.length, 64);
  assert.deepEqual(r.calls.filter((c) => c.mode === 'preview').map((c) => c.wantLogits), ['top1', 'all', 'all']);
});

test('top1: never used with penalties on', async () => {
  const r = runtime({ top1: true });
  await denoise({ forward: r.forward, promptIds: PROMPT, ...clover, penalties: PRESETS.clover.penalties, top1: true, maxNewTokens: 32 });
  assert.ok(r.calls.filter((c) => c.mode === 'preview').every((c) => c.wantLogits === 'all'));
});

// Fixtures from CloverPenaltyState.apply in Diffuspace/Clover-35B-A3B-Preview sampling_clover_moe.py,
// run in PyTorch (fp32) on random logits.
test('penalties match CloverPenaltyState.apply', () => {
  const fx = JSON.parse(readFileSync(new URL('./fixtures/clover-penalties.json', import.meta.url)));
  for (const c of fx.cases) {
    const g = c.gen;
    const p = {
      presence: g.presence_penalty ?? 0, frequency: g.frequency_penalty ?? 0, repetition: g.repetition_penalty ?? 1,
      length: g.dllm_length_penalty ?? 0, lengthStart: g.dllm_length_penalty_start ?? 8192, lengthWindow: g.dllm_length_penalty_window ?? 8192,
      thinkEnd: g.dllm_think_end_bias ?? 0, thinkEndStart: g.dllm_think_end_start ?? 4096, thinkEndWindow: g.dllm_think_end_window ?? 4096,
      thinkEndId: g.dllm_think_end_token_id, stopIds: fx.stop,
    };
    const counts = new Map();
    for (const t of c.counts) counts.set(t, (counts.get(t) ?? 0) + 1);
    const logits = Float32Array.from(c.before.flat());
    applyPenalties(logits, fx.R, fx.V, { counts, length: c.length, block: Int32Array.from(c.block), maskId: fx.MASK, p });
    const want = Float32Array.from(c.after.flat());
    for (let i = 0; i < want.length; i++) assert.equal(logits[i], want[i], `case ${JSON.stringify(g)} index ${i}`);
  }
});

test('penalty history: prompt never counts, a block never penalises itself, committed tokens do', async () => {
  // Token 5 leads token 6 by 0.5. Presence 0.9 flips that only once 5 is committed, so block 1
  // is all 5s (the prompt's 6s and the block's own 5s do not count) and block 2 is all 6s.
  const forward = async ({ mode, ids }) => {
    if (mode === 'commit') return null;
    const out = new Float32Array(ids.length * V);
    for (let i = 0; i < ids.length; i++) { out[i * V + 5] = 10; out[i * V + 6] = 9.5; }
    return out;
  };
  const out = await denoise({
    forward, promptIds: [6, 6, 6], ...clover, maxNewTokens: 64,
    penalties: { presence: 0.9, repetition: 1, frequency: 0 },
  });
  assert.deepEqual(out.tokenIds, [...Array(32).fill(5), ...Array(32).fill(6)]);
});
