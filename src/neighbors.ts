/** Cosine nearest-neighbor search over the model's own embedding table —
 *  Act 2's "map of meaning", computed live from the weights already in memory. */

export interface Neighbor {
  id: number;
  sim: number;
}

export interface NeighborOptions {
  /** Candidate id range — GPT-2 BPE ids are roughly frequency-ordered, so
   *  capping the range skips most never-trained byte junk. */
  minId?: number;
  maxId?: number;
  /** Skip candidates whose norm is below this fraction of the query norm —
   *  untrained embeddings sit near their tiny random init. */
  minNormRatio?: number;
}

export function nearestNeighbors(
  wte: Float32Array,
  hidden: number,
  tokenId: number,
  k: number,
  opts: NeighborOptions = {}
): Neighbor[] {
  const vocab = Math.floor(wte.length / hidden);
  if (tokenId < 0 || tokenId >= vocab) return [];
  const minId = opts.minId ?? 0;
  const maxId = Math.min(opts.maxId ?? vocab, vocab);
  const minNormRatio = opts.minNormRatio ?? 0.35;

  const qOff = tokenId * hidden;
  let qNorm = 0;
  for (let d = 0; d < hidden; d++) qNorm += wte[qOff + d] * wte[qOff + d];
  qNorm = Math.sqrt(qNorm);
  if (qNorm === 0) return [];
  const normFloor = qNorm * minNormRatio;

  const best: Neighbor[] = [];
  for (let id = minId; id < maxId; id++) {
    if (id === tokenId) continue;
    const off = id * hidden;
    let dot = 0;
    let norm = 0;
    for (let d = 0; d < hidden; d++) {
      dot += wte[qOff + d] * wte[off + d];
      norm += wte[off + d] * wte[off + d];
    }
    norm = Math.sqrt(norm);
    if (norm < normFloor) continue;
    const sim = dot / (qNorm * norm);
    if (best.length < k) {
      best.push({ id, sim });
      best.sort((a, b) => b.sim - a.sim);
    } else if (sim > best[k - 1].sim) {
      best[k - 1] = { id, sim };
      best.sort((a, b) => b.sim - a.sim);
    }
  }
  return best;
}
