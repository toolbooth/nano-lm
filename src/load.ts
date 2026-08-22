/** Loading weights — from bytes you already have, or from any fetch-able URL. */
import { NanoGPT, type NanoMeta } from "./model.js";
import { parseSafetensors, type ByteSource, type NanoTensor } from "./safetensors.js";

/** Architecture constants of the bundled checkpoint (= `weights/meta.json`). */
export const TINYSTORIES_1M_META: NanoMeta = {
  hidden: 64,
  layers: 8,
  heads: 16,
  vocab: 50257,
  maxPos: 2048,
  lnEps: 1e-5,
};

/** Provenance of the bundled checkpoint. */
export const TINYSTORIES_1M = {
  /** Hugging Face model id the weights were converted from. */
  model: "roneneldan/TinyStories-1M",
  arch: "gpt_neo",
  /** File names under the package's `weights/` directory. */
  weightsFile: "tinystories-1m.safetensors",
  metaFile: "meta.json",
  meta: TINYSTORIES_1M_META,
} as const;

// Kept as variables (not string literals) so bundlers such as Vite do not
// rewrite these `new URL(..., import.meta.url)` calls into emitted assets —
// the essay serves the weights from its own public/ directory instead.
const WEIGHTS_REL = `../weights/${TINYSTORIES_1M.weightsFile}`;
const META_REL = `../weights/${TINYSTORIES_1M.metaFile}`;

/**
 * `file:` URL of the bundled safetensors file, resolved relative to this
 * module — for Node (`readFile(fileURLToPath(bundledWeightsURL()))`) and for
 * bundlers that can serve package files. Browsers cannot fetch `file:` URLs:
 * self-host the file and pass its http(s) URL to {@link loadModel} instead.
 */
export function bundledWeightsURL(): URL {
  return new URL(WEIGHTS_REL, import.meta.url);
}

/** `file:` URL of the bundled `meta.json` (same contents as {@link TINYSTORIES_1M_META}). */
export function bundledMetaURL(): URL {
  return new URL(META_REL, import.meta.url);
}

/** Download a URL as an ArrayBuffer, reporting integer percentages when the server sends Content-Length. */
export async function fetchWithProgress(
  url: string | URL,
  onPct: (pct: number) => void,
  fetchImpl: typeof fetch = fetch
): Promise<ArrayBuffer> {
  const res = await fetchImpl(url);
  if (!res.ok) throw new Error(`fetch ${url}: ${res.status}`);
  const total = Number(res.headers.get("content-length") ?? 0);
  if (!res.body || !total) return res.arrayBuffer();
  const reader = res.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onPct(Math.round((received / total) * 100));
  }
  const buf = new Uint8Array(received);
  let off = 0;
  for (const c of chunks) {
    buf.set(c, off);
    off += c.length;
  }
  return buf.buffer;
}

/** Build a model from bytes (or already-parsed tensors) you hold in memory. Synchronous. */
export function createModel(weights: ByteSource | Map<string, NanoTensor>, meta: NanoMeta): NanoGPT {
  const tensors = weights instanceof Map ? weights : parseSafetensors(weights);
  return new NanoGPT(tensors, meta);
}

export interface LoadOptions {
  /** Download progress, 0–100, when `weights` is a URL and the server reports a length. */
  onProgress?: (pct: number) => void;
  /** Custom fetch (tests, Node polyfills, auth headers). Defaults to the global `fetch`. */
  fetch?: typeof fetch;
}

export type WeightsSource = string | URL | ByteSource | Map<string, NanoTensor>;
export type MetaSource = string | URL | NanoMeta;

/**
 * Load a model from a URL or from bytes. Strings and `URL`s are fetched (a
 * `meta.json` URL likewise); buffers are parsed directly. `meta` defaults to
 * the bundled TinyStories-1M architecture, so self-hosting only the
 * `.safetensors` file is enough:
 *
 * ```ts
 * const model = await loadModel("/weights/tinystories-1m.safetensors", undefined, { onProgress });
 * ```
 */
export async function loadModel(
  weights: WeightsSource,
  meta: MetaSource = TINYSTORIES_1M_META,
  opts: LoadOptions = {}
): Promise<NanoGPT> {
  const fetchImpl = opts.fetch ?? fetch;
  const onPct = opts.onProgress ?? (() => {});
  const [resolvedMeta, bytes] = await Promise.all([
    typeof meta === "string" || meta instanceof URL
      ? fetchImpl(meta).then((r) => {
          if (!r.ok) throw new Error(`fetch ${meta}: ${r.status}`);
          return r.json() as Promise<NanoMeta>;
        })
      : Promise.resolve(meta),
    typeof weights === "string" || weights instanceof URL
      ? fetchWithProgress(weights, onPct, fetchImpl)
      : Promise.resolve(weights),
  ]);
  return createModel(bytes, resolvedMeta);
}
