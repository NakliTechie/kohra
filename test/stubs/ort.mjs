// Stand-in for onnxruntime-web under Node: kohra.js only constructs Tensors itself;
// the session is injected by the test.
export class Tensor {
  constructor(type, data, dims) { this.type = type; this.data = data; this.dims = dims; }
}
export const InferenceSession = {
  create() { throw new Error('test stub: inject a fake session instead'); },
};
