// Download tests: fetchCached's progress events, Cache Storage reuse, and byte assembly
// when content-length is missing or wrong. Fake fetch and an in-memory cache; no network.
import { test as nodeTest } from 'node:test';
import assert from 'node:assert/strict';
import { fetchCached } from '../kohra.js';

const test = (name, fn) => nodeTest(name, { timeout: 10_000 }, fn);
const URL_ = 'https://example.test/onnx/model.onnx_data';
const DATA = Uint8Array.from({ length: 1000 }, (_, i) => i % 251);

// A response that streams DATA in 7 chunks, with a chosen content-length header (or none).
function respond(length = String(DATA.length)) {
  let i = 0;
  const step = Math.ceil(DATA.length / 7);
  const body = new ReadableStream({
    pull(c) {
      if (i >= DATA.length) return c.close();
      c.enqueue(DATA.slice(i, i + step));
      i += step;
    },
  });
  return new Response(body, length === null ? {} : { headers: { 'content-length': length } });
}

function fakeCache() {
  const store = new Map();
  return {
    store,
    async match(u) { const b = store.get(u); return b ? new Response(b, { headers: { 'content-length': String(b.length) } }) : undefined; },
    async put(u, res) { store.set(u, new Uint8Array(await res.arrayBuffer())); },
  };
}

function withFetch(make, fn) {
  const real = globalThis.fetch;
  let calls = 0;
  globalThis.fetch = async () => { calls++; return make(); };
  return fn(() => calls).finally(() => { globalThis.fetch = real; });
}

test('miss downloads with progress, then a hit reads the cache without fetching', () =>
  withFetch(() => respond(), async (calls) => {
    const cache = fakeCache();
    const events = [];
    const a = await fetchCached(URL_, { cache, progress_callback: (e) => events.push(e) });
    assert.deepEqual(a, DATA);
    assert.equal(calls(), 1);
    assert.deepEqual(cache.store.get(URL_), DATA);
    assert.equal(events[0].status, 'initiate');
    assert.equal(events.at(-1).status, 'done');
    const prog = events.filter((e) => e.status === 'progress');
    assert.ok(prog.length >= 7);
    assert.ok(prog.every((e, i) => i === 0 || e.loaded > prog[i - 1].loaded));
    assert.equal(prog.at(-1).loaded, DATA.length);
    assert.equal(prog.at(-1).total, DATA.length);
    assert.equal(prog[0].file, 'model.onnx_data');

    const b = await fetchCached(URL_, { cache });
    assert.deepEqual(b, DATA);
    assert.equal(calls(), 1);
  }));

test('no content-length: chunks are joined in order', () =>
  withFetch(() => respond(null), async () => {
    assert.deepEqual(await fetchCached(URL_), DATA);
  }));

test('content-length smaller than the body: every byte is kept', () =>
  withFetch(() => respond('300'), async () => {
    assert.deepEqual(await fetchCached(URL_), DATA);
  }));

test('a failed status throws', () =>
  withFetch(() => new Response('nope', { status: 404 }), async () => {
    await assert.rejects(fetchCached(URL_), /404/);
  }));
