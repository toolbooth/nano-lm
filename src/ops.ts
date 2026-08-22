/** Tiny tensor ops for the hand-rolled forward pass. Row-major throughout. */

export function layerNorm(
  x: Float32Array, // [seq, dim]
  seq: number,
  dim: number,
  weight: Float32Array,
  bias: Float32Array,
  eps: number
): Float32Array {
  const out = new Float32Array(seq * dim);
  for (let s = 0; s < seq; s++) {
    const off = s * dim;
    let mean = 0;
    for (let d = 0; d < dim; d++) mean += x[off + d];
    mean /= dim;
    let variance = 0;
    for (let d = 0; d < dim; d++) {
      const diff = x[off + d] - mean;
      variance += diff * diff;
    }
    variance /= dim;
    const inv = 1 / Math.sqrt(variance + eps);
    for (let d = 0; d < dim; d++) {
      out[off + d] = (x[off + d] - mean) * inv * weight[d] + bias[d];
    }
  }
  return out;
}

/** y = x·Wᵀ + b, with W stored [out, in] (torch nn.Linear layout). */
export function linear(
  x: Float32Array, // [seq, inDim]
  seq: number,
  inDim: number,
  outDim: number,
  w: Float32Array, // [outDim, inDim]
  b: Float32Array | null
): Float32Array {
  const out = new Float32Array(seq * outDim);
  for (let s = 0; s < seq; s++) {
    const xo = s * inDim;
    const oo = s * outDim;
    for (let o = 0; o < outDim; o++) {
      const wo = o * inDim;
      let acc = b ? b[o] : 0;
      for (let i = 0; i < inDim; i++) acc += x[xo + i] * w[wo + i];
      out[oo + o] = acc;
    }
  }
  return out;
}

const GELU_C = Math.sqrt(2 / Math.PI);

/** gelu_new — the tanh approximation GPT-2/Neo were trained with. */
export function geluNew(x: Float32Array): Float32Array {
  const out = new Float32Array(x.length);
  for (let i = 0; i < x.length; i++) {
    const v = x[i];
    out[i] = 0.5 * v * (1 + Math.tanh(GELU_C * (v + 0.044715 * v * v * v)));
  }
  return out;
}

/** In-place softmax over a row segment. */
export function softmaxRow(arr: Float32Array, start: number, len: number): void {
  let max = -Infinity;
  for (let i = 0; i < len; i++) max = Math.max(max, arr[start + i]);
  let sum = 0;
  for (let i = 0; i < len; i++) {
    const e = Math.exp(arr[start + i] - max);
    arr[start + i] = e;
    sum += e;
  }
  for (let i = 0; i < len; i++) arr[start + i] /= sum;
}
