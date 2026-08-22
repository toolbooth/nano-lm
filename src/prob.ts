/** Pure probability helpers — unit-tested, no model dependency. */

export interface TokenProb {
  id: number;
  p: number;
}

/** Temperature-adjusted softmax over the top-k logits. */
export function softmaxTopK(logits: Float32Array, k: number, temperature: number): TokenProb[] {
  const t = Math.max(0.05, temperature);
  const idx = [...logits.keys()].sort((a, b) => logits[b] - logits[a]).slice(0, k);
  const max = logits[idx[0]] / t;
  const exps = idx.map((i) => Math.exp(logits[i] / t - max));
  const Z = exps.reduce((a, b) => a + b, 0);
  return idx.map((id, r) => ({ id, p: exps[r] / Z }));
}

/** Sample an entry from a distribution using rand ∈ [0,1). */
export function sampleFrom(dist: TokenProb[], rand: number): TokenProb {
  let acc = 0;
  for (const d of dist) {
    acc += d.p;
    if (rand < acc) return d;
  }
  return dist[dist.length - 1];
}

/** Index of the largest logit — greedy decoding's pick (ties → lowest index). */
export function argmax(logits: Float32Array): number {
  let best = 0;
  for (let t = 1; t < logits.length; t++) if (logits[t] > logits[best]) best = t;
  return best;
}

/** Ids of the k largest logits, descending. */
export function topK(logits: Float32Array, k: number): number[] {
  return [...logits.keys()].sort((a, b) => logits[b] - logits[a]).slice(0, k);
}

/** Temperature-adjusted softmax over the whole logit vector (a new array). */
export function softmax(logits: Float32Array, temperature = 1): Float32Array {
  const t = Math.max(0.05, temperature);
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) max = Math.max(max, logits[i] / t);
  const out = new Float32Array(logits.length);
  let Z = 0;
  for (let i = 0; i < logits.length; i++) {
    const e = Math.exp(logits[i] / t - max);
    out[i] = e;
    Z += e;
  }
  for (let i = 0; i < out.length; i++) out[i] /= Z;
  return out;
}

export interface SampleOptions {
  /** Restrict sampling to the k most likely tokens (default 8). */
  k?: number;
  /** Softmax temperature (default 1; floored at 0.05). */
  temperature?: number;
  /** Uniform random number in [0, 1) — inject for determinism (default Math.random()). */
  rand?: number;
}

/** One top-k / temperature sampling step: `sampleFrom(softmaxTopK(...))`. */
export function sampleNext(logits: Float32Array, opts: SampleOptions = {}): TokenProb {
  const dist = softmaxTopK(logits, opts.k ?? 8, opts.temperature ?? 1);
  return sampleFrom(dist, opts.rand ?? Math.random());
}
