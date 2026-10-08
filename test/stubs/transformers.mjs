// Stand-in for transformers.js under Node: tests inject their own tokenizer.
export const AutoTokenizer = {
  from_pretrained() { throw new Error('test stub: inject a fake tokenizer instead'); },
};
