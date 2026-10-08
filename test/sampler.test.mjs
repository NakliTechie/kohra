// Sampler tests: the real kohra.js denoising loop against a fake ONNX session.
// Run: npm test   (no install; Node >= 22)
import { test as nodeTest } from 'node:test';

// Every test gets a deadline: a sampler bug that stops making progress loops forever.
const test = (name, fn) => nodeTest(name, { timeout: 10_000 }, fn);
import assert from 'node:assert/strict';
import { DiffusionLM } from '../kohra.js';

const V = 32;                 // tiny vocabulary
const MASK = 31n, EOS = 30n;  // token ids the sampler treats specially
const PROMPT = [1, 2, 3, 4, 5, 6, 7];   // P = 7: not block-aligned, on purpose

// The token the fake model "wants" at position p, and how sure it is.
const want = (p) => p % 20;
const sureness = (p) => 2 + (p % 7);     // logit margin; varies so commit order is defined

// Fake session: logits [1, T, V] with `want(p)` ahead by `sureness(p)` (or `margin`).
// Records every feed so tests can inspect inputs.
function fakeSession({ margin = null, eosAt = null } = {}) {
  const calls = [];
  return {
    calls,
    async run(feeds) {
      calls.push(feeds);
      const T = feeds.input_ids.dims[1];
      const data = new Float32Array(T * V);
      for (let p = 0; p < T; p++) {
        const tok = p === eosAt ? Number(EOS) : want(p);
        data[p * V + tok] = margin ?? sureness(p);
      }
      return { logits: { data, dims: [1, T, V] } };
    },
  };
}

const tokenizer = { decode: (ids) => ids.join(',') };

function model(session) {
  return new DiffusionLM({ session, tokenizer, ort: { Tensor: class { constructor(t, d, s) { this.type = t; this.data = d; this.dims = s; } } }, maskId: MASK, eosId: EOS });
}

const gen = (lm, opts = {}) =>
  lm.generate(PROMPT, { maxNewTokens: 128, steps: 128, blockSize: 32, ...opts });

const expected = (P, n) => Array.from({ length: n }, (_, i) => want(P + i));

test('fixed steps: one forward per step, every position unmasked to the argmax', async () => {
  for (const steps of [128, 64, 32]) {
    const out = await gen(model(fakeSession()), { steps });
    assert.equal(out.forwards, steps, `steps ${steps}`);
    assert.equal([...out.x].filter((t) => t === MASK).length, 0);
    assert.deepEqual(out.tokenIds, expected(PROMPT.length, 128));
  }
});

test('blocks are denoised left to right: a later block never commits before an earlier one finishes', async () => {
  const blockOf = [];
  await gen(model(fakeSession()), {
    steps: 64,
    onStep: ({ fresh, block }) => { for (const p of fresh) blockOf.push([p, block]); },
  });
  const P = PROMPT.length;
  for (const [p, b] of blockOf) assert.equal(Math.floor((p - P) / 32), b);
  const order = blockOf.map(([, b]) => b);
  assert.deepEqual(order, [...order].sort((a, b) => a - b));
});

test('each step commits the most confident masked positions first', async () => {
  const P = PROMPT.length;
  const commits = [];
  await gen(model(fakeSession()), { onStep: ({ fresh }) => commits.push([...fresh]) });
  // steps 128 over 4 blocks of 32: one reveal per forward, in confidence order within a block.
  const block0 = commits.slice(0, 32).flat();
  const sure = block0.map(sureness);
  for (let i = 1; i < sure.length; i++) assert.ok(sure[i] <= sure[i - 1], `order ${sure}`);
  assert.deepEqual([...block0].sort((a, b) => a - b), Array.from({ length: 32 }, (_, i) => P + i));
});

test('onStep reveals each generated position exactly once', async () => {
  const seen = [];
  await gen(model(fakeSession()), { onStep: ({ fresh }) => seen.push(...fresh) });
  const P = PROMPT.length;
  assert.deepEqual([...seen].sort((a, b) => a - b), Array.from({ length: 128 }, (_, i) => P + i));
});

test('threshold: a confident model clears each block in one forward', async () => {
  const out = await gen(model(fakeSession({ margin: 20 })), { threshold: 0.8 });
  assert.equal(out.forwards, 4);            // 4 blocks of 32
  assert.deepEqual(out.tokenIds, expected(PROMPT.length, 128));
});

test('threshold: an unreachable bar still makes progress, one token per forward', async () => {
  const out = await gen(model(fakeSession({ margin: 0.1 })), { threshold: 0.99 });
  assert.equal(out.forwards, 128);
  assert.equal([...out.x].filter((t) => t === MASK).length, 0);
});

test('threshold: lowering the bar never adds forwards', async () => {
  const fwd = [];
  for (const threshold of [0.95, 0.9, 0.8, 0.7]) {
    fwd.push((await gen(model(fakeSession()), { threshold })).forwards);
  }
  for (let i = 1; i < fwd.length; i++) assert.ok(fwd[i] <= fwd[i - 1], `forwards ${fwd}`);
});

test('output is trimmed at the first EOS', async () => {
  const P = PROMPT.length;
  const out = await gen(model(fakeSession({ eosAt: P + 5 })));
  assert.deepEqual(out.tokenIds, expected(P, 5));
  assert.equal(out.text, expected(P, 5).join(','));
});

test('MDLM sends input_ids only', async () => {
  const s = fakeSession();
  await gen(model(s), { steps: 4 });
  assert.deepEqual(Object.keys(s.calls[0]), ['input_ids']);
});

test('BD3LM sends a static block-causal mask on the physical pos//blockSize grid', async () => {
  const s = fakeSession();
  const out = await gen(model(s), { blockCausal: true, steps: 16 });
  const T = PROMPT.length + 128;
  const m = s.calls[0].attention_mask;
  assert.deepEqual(m.dims, [1, 1, T, T]);
  for (let q = 0; q < T; q += 5) {
    for (let k = 0; k < T; k += 3) {
      const ok = Math.floor(k / 32) <= Math.floor(q / 32);
      assert.equal(m.data[q * T + k], ok ? 0 : -1e9, `q ${q} k ${k}`);
    }
  }
  assert.ok(s.calls.every((f) => f.attention_mask === m), 'mask is built once and reused');
  assert.equal([...out.x].filter((t) => t === MASK).length, 0);
  assert.deepEqual(out.tokenIds, expected(PROMPT.length, 128));
});

test('the prompt is never overwritten', async () => {
  for (const blockCausal of [false, true]) {
    const out = await gen(model(fakeSession()), { blockCausal, threshold: 0.8 });
    assert.deepEqual([...out.x.slice(0, PROMPT.length)].map(Number), PROMPT);
  }
});
