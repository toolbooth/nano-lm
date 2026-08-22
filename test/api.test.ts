/** Public API surface: what `import ... from "nano-lm"` must provide, and the loaders. */
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import * as api from "../src/index.js";
import {
  bundledMetaURL,
  bundledWeightsURL,
  createModel,
  fetchWithProgress,
  loadModel,
  NanoGPT,
  TINYSTORIES_1M,
  TINYSTORIES_1M_META,
} from "../src/index.js";

const ROOT = join(__dirname, "..");
const weightsBytes = readFileSync(join(ROOT, "weights/tinystories-1m.safetensors"));
const ref = JSON.parse(readFileSync(join(ROOT, "tools/reference.json"), "utf8"));

describe("exports", () => {
  it("exposes the documented surface", () => {
    const fns = [
      "NanoGPT",
      "parseSafetensors",
      "f16ToF32",
      "toArrayBuffer",
      "loadModel",
      "createModel",
      "fetchWithProgress",
      "bundledWeightsURL",
      "bundledMetaURL",
      "nearestNeighbors",
      "diagnoseHeads",
      "softmaxTopK",
      "sampleFrom",
      "sampleNext",
      "softmax",
      "topK",
      "argmax",
      "layerNorm",
      "linear",
      "geluNew",
      "softmaxRow",
    ] as const;
    for (const name of fns) expect(typeof (api as Record<string, unknown>)[name], name).toBe("function");
    expect(api.TINYSTORIES_1M.model).toBe("roneneldan/TinyStories-1M");
  });
});

describe("bundled weights", () => {
  it("TINYSTORIES_1M_META matches weights/meta.json", () => {
    const meta = JSON.parse(readFileSync(fileURLToPath(bundledMetaURL()), "utf8"));
    for (const [k, v] of Object.entries(TINYSTORIES_1M_META)) expect(meta[k], k).toBe(v);
    expect(meta.model).toBe(TINYSTORIES_1M.model);
    expect(meta.arch).toBe(TINYSTORIES_1M.arch);
  });

  it("bundledWeightsURL points at the shipped 7.5 MB file", () => {
    const p = fileURLToPath(bundledWeightsURL());
    expect(p.endsWith(`weights/${TINYSTORIES_1M.weightsFile}`)).toBe(true);
    expect(readFileSync(p).byteLength).toBe(7502858);
  });
});

describe("createModel / loadModel", () => {
  it("createModel accepts a Node Buffer (offset view) and reproduces the reference", () => {
    const model = createModel(weightsBytes, TINYSTORIES_1M_META);
    expect(model).toBeInstanceOf(NanoGPT);
    expect(model.generate(ref.input_ids, 12)).toEqual(ref.greedy_ids);
  });

  it("createModel accepts pre-parsed tensors", () => {
    const tensors = api.parseSafetensors(weightsBytes);
    const model = createModel(tensors, TINYSTORIES_1M_META);
    expect(model.forward(ref.input_ids).logits[0]).toBeCloseTo(ref.logits_first8[0], 1);
  });

  it("loadModel with bytes defaults to the bundled meta", async () => {
    const model = await loadModel(weightsBytes);
    expect(model.meta).toEqual(TINYSTORIES_1M_META);
    expect(model.generate(ref.input_ids, 3)).toEqual(ref.greedy_ids.slice(0, ref.input_ids.length + 3));
  });

  it("loadModel fetches string/URL sources through the injected fetch and reports progress", async () => {
    const calls: string[] = [];
    const fakeFetch: typeof fetch = async (input) => {
      const url = String(input);
      calls.push(url);
      if (url.endsWith("meta.json")) {
        return new Response(JSON.stringify({ ...TINYSTORIES_1M_META, lnEps: 1e-5 }), {
          headers: { "content-type": "application/json" },
        });
      }
      if (url.endsWith(".safetensors")) {
        return new Response(weightsBytes, { headers: { "content-length": String(weightsBytes.byteLength) } });
      }
      return new Response(null, { status: 404 });
    };
    const pcts: number[] = [];
    const model = await loadModel(
      "https://example.test/weights/tinystories-1m.safetensors",
      new URL("https://example.test/weights/meta.json"),
      { fetch: fakeFetch, onProgress: (p) => pcts.push(p) }
    );
    expect(calls.sort()).toEqual([
      "https://example.test/weights/meta.json",
      "https://example.test/weights/tinystories-1m.safetensors",
    ]);
    expect(pcts.length).toBeGreaterThan(0);
    expect(pcts[pcts.length - 1]).toBe(100);
    for (let i = 1; i < pcts.length; i++) expect(pcts[i]).toBeGreaterThanOrEqual(pcts[i - 1]);
    expect(model.generate(ref.input_ids, 12)).toEqual(ref.greedy_ids);
  });

  it("loadModel surfaces HTTP failures", async () => {
    const fakeFetch: typeof fetch = async () => new Response(null, { status: 404 });
    await expect(loadModel("https://example.test/missing.safetensors", undefined, { fetch: fakeFetch })).rejects.toThrow(
      "404"
    );
  });

  it("fetchWithProgress falls back to arrayBuffer() without Content-Length", async () => {
    const fakeFetch: typeof fetch = async () => new Response(new Uint8Array([1, 2, 3]));
    const pcts: number[] = [];
    const buf = await fetchWithProgress("x", (p) => pcts.push(p), fakeFetch);
    expect(new Uint8Array(buf)).toEqual(new Uint8Array([1, 2, 3]));
    expect(pcts).toEqual([]);
  });
});

describe("analyses on the real weights", () => {
  const model = createModel(weightsBytes, TINYSTORIES_1M_META);

  it("nearestNeighbors reproduces the essay Act 2 map for [ dragon]: knight 0.76, monster 0.75", () => {
    // GPT-2 BPE ids: " dragon" 10441, " knight" 22062, " monster" 9234 (from the checkpoint's vocab.json).
    const n = api.nearestNeighbors(model.wte, model.meta.hidden, 10441, 3, { minId: 256, maxId: 30000 });
    expect(n).toHaveLength(3);
    expect(n[0]).toMatchObject({ id: 22062 });
    expect(n[0].sim).toBeCloseTo(0.759, 2);
    expect(n[1]).toMatchObject({ id: 9234 });
    expect(n[1].sim).toBeCloseTo(0.75, 2);
    expect(n.map((x) => x.id)).not.toContain(10441);
  });

  it("diagnoseHeads runs on a real forward pass", () => {
    const r = model.forward(ref.greedy_ids);
    const d = api.diagnoseHeads(r.attentions, r.seq)!;
    expect(d).not.toBeNull();
    for (const pick of [d.prevToken, d.firstToken, d.diffuse]) {
      expect(pick.layer).toBeGreaterThanOrEqual(0);
      expect(pick.layer).toBeLessThan(model.meta.layers);
      expect(pick.head).toBeLessThan(model.meta.heads);
    }
    expect(d.prevToken.score).toBeGreaterThan(0);
  });
});
