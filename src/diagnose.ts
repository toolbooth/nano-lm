/** Auto-discover "character heads" from real attention matrices — so every
 *  claim the UI makes is computed from the user's actual input, never scripted. */

export interface HeadPick {
  layer: number;
  head: number;
  score: number;
}

export interface HeadDiagnostics {
  /** Head that most strongly attends to the previous token. */
  prevToken: HeadPick;
  /** Head that most strongly attends back to the first token ("anchor"). */
  firstToken: HeadPick;
  /** Head with the most diffuse (spread-out) attention. */
  diffuse: HeadPick;
}

function entropyOfRow(attn: Float32Array, seq: number, q: number): number {
  let h = 0;
  for (let k = 0; k <= q; k++) {
    const p = attn[q * seq + k];
    if (p > 1e-9) h -= p * Math.log(p);
  }
  return h;
}

export function diagnoseHeads(attentions: Float32Array[][], seq: number): HeadDiagnostics | null {
  if (seq < 3) return null;
  let prev: HeadPick = { layer: 0, head: 0, score: -1 };
  let first: HeadPick = { layer: 0, head: 0, score: -1 };
  let diffuse: HeadPick = { layer: 0, head: 0, score: -1 };

  for (let l = 0; l < attentions.length; l++) {
    for (let h = 0; h < attentions[l].length; h++) {
      const attn = attentions[l][h];
      let prevSum = 0;
      let firstSum = 0;
      let entSum = 0;
      for (let q = 1; q < seq; q++) {
        prevSum += attn[q * seq + (q - 1)];
        firstSum += attn[q * seq + 0];
        entSum += entropyOfRow(attn, seq, q);
      }
      const n = seq - 1;
      if (prevSum / n > prev.score) prev = { layer: l, head: h, score: prevSum / n };
      if (firstSum / n > first.score) first = { layer: l, head: h, score: firstSum / n };
      if (entSum / n > diffuse.score) diffuse = { layer: l, head: h, score: entSum / n };
    }
  }
  return { prevToken: prev, firstToken: first, diffuse };
}
