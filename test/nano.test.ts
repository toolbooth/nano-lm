/**
 * Token-exact correctness against the official GPT-Neo implementation.
 * tools/reference.json was dumped by tools/export_reference.py from
 * transformers' GPTNeoForCausalLM on the same checkpoint (fp32); the fp16
 * weights here must reproduce its greedy continuation exactly and its
 * logits/attention maps within fp16 tolerance.
 */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { describe, expect, it } from "vitest";
import { parseSafetensors } from "../src/safetensors.js";
import { NanoGPT, type NanoMeta } from "../src/model.js";

const ROOT = join(__dirname, "..");
const WEIGHTS = join(ROOT, "weights/tinystories-1m.safetensors");
const META = join(ROOT, "weights/meta.json");
const REF = join(ROOT, "tools/reference.json");

function toArrayBuffer(buf: Buffer): ArrayBuffer {
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
}

const meta = JSON.parse(readFileSync(META, "utf8")) as NanoMeta;
const ref = JSON.parse(readFileSync(REF, "utf8"));
const tensors = parseSafetensors(toArrayBuffer(readFileSync(WEIGHTS)));
const model = new NanoGPT(tensors, meta);

describe("weights", () => {
  it("loads all 108 tensors with expected shapes", () => {
    expect(tensors.size).toBe(108);
    expect(tensors.get("transformer.wte.weight")!.shape).toEqual([50257, 64]);
    expect(tensors.get("transformer.h.7.mlp.c_proj.weight")!.shape).toEqual([64, 256]);
  });

  it("exposes the same tensors through the model", () => {
    expect(model.tensorNames()).toHaveLength(108);
    expect(model.tensor("transformer.wte.weight")!.shape).toEqual([50257, 64]);
    expect(model.tensor("no.such.tensor")).toBeUndefined();
    expect(model.wte).toBe(tensors.get("transformer.wte.weight")!.data);
    expect(model.wpe.length).toBe(meta.maxPos * meta.hidden);
  });
});

describe("forward pass vs official implementation", () => {
  const result = model.forward(ref.input_ids);

  it("reproduces the last-position logits (fp16 tolerance)", () => {
    for (let i = 0; i < 8; i++) {
      expect(result.logits[i]).toBeCloseTo(ref.logits_first8[i], 1);
    }
  });

  it("agrees on the top-3 next tokens, in order", () => {
    const ids = [...result.logits.keys()]
      .sort((a, b) => result.logits[b] - result.logits[a])
      .slice(0, 3);
    expect(ids).toEqual(ref.top10.slice(0, 3).map((t: { id: number }) => t.id));
  });

  it("reproduces attention matrices layer 0 head 0 and layer 7 head 3", () => {
    const seq = ref.input_ids.length;
    for (const [attn, refM] of [
      [result.attentions[0][0], ref.attn_l0_h0],
      [result.attentions[7][3], ref.attn_l7_h3],
    ] as const) {
      for (let q = 0; q < seq; q++) {
        for (let k = 0; k < seq; k++) {
          expect(Math.abs(attn[q * seq + k] - refM[q][k])).toBeLessThan(0.02);
        }
      }
    }
  });

  it("attention rows are valid distributions", () => {
    const seq = ref.input_ids.length;
    const attn = result.attentions[3][7];
    for (let q = 0; q < seq; q++) {
      let sum = 0;
      for (let k = 0; k < seq; k++) sum += attn[q * seq + k];
      expect(sum).toBeCloseTo(1, 4);
    }
  });

  it("returns layers + 1 hidden states: embeddings first, residual stream after each block", () => {
    const { hidden: H } = meta;
    const seq = ref.input_ids.length;
    expect(result.hiddenStates).toHaveLength(meta.layers + 1);
    for (const hs of result.hiddenStates) expect(hs.length).toBe(seq * H);
    // hiddenStates[0] is wte[id] + wpe[pos], recomputed here from the raw tables
    const wte = model.wte;
    const wpe = model.wpe;
    for (let s = 0; s < seq; s++) {
      for (let d = 0; d < H; d++) {
        expect(result.hiddenStates[0][s * H + d]).toBe(wte[ref.input_ids[s] * H + d] + wpe[s * H + d]);
      }
    }
    // snapshots are independent copies, not views of one mutated buffer
    expect(result.hiddenStates[0]).not.toEqual(result.hiddenStates[1]);
    expect(result.hiddenStates[meta.layers]).not.toEqual(result.hiddenStates[meta.layers - 1]);
  });

  it("rejects empty and over-long inputs", () => {
    expect(() => model.forward([])).toThrow("empty input");
    expect(() => model.forward(new Array(meta.maxPos + 1).fill(0))).toThrow("input too long");
  });
});

describe("greedy generation vs official implementation", () => {
  it("reproduces the exact 12-token continuation", () => {
    const out = model.generate(ref.input_ids, 12);
    expect(out).toEqual(ref.greedy_ids);
  });
});
