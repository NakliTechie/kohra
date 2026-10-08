// Module hook: kohra.js imports onnxruntime-web and transformers.js from a CDN, which
// Node cannot load. Map those two URLs to local stubs so the real sampler runs in Node.
const STUBS = [
  ['https://cdn.jsdelivr.net/npm/onnxruntime-web@', './stubs/ort.mjs'],
  ['https://cdn.jsdelivr.net/npm/@huggingface/transformers@', './stubs/transformers.mjs'],
];

export async function resolve(specifier, context, next) {
  for (const [prefix, stub] of STUBS) {
    if (specifier.startsWith(prefix)) {
      return { url: new URL(stub, import.meta.url).href, shortCircuit: true };
    }
  }
  return next(specifier, context);
}
