# nano-lm

An **auditable GPT-Neo forward pass in dependency-free TypeScript**, bundled
with the 7.5 MB TinyStories-1M checkpoint. Feed it token ids; get back the
next-token logits, every layer's per-head attention weights, and the residual
stream after every block — all computed by ~150 lines of plain loops you can
read top to bottom, and verified **token-exact** against the reference
Hugging Face implementation.

Extracted from [Inside the Machine](../llm-explainer) by Shangyan Shen, the
interactive essay in which readers type a sentence and watch a real language
model — this one — tokenize, embed, attend and gamble on the next word, in
the browser.

> **Status: pre-release.** This package is `private` and not yet published.
> `nano-lm` is the working name — see [NAMING.md](./NAMING.md) for the
> candidates and their npm availability. The license will be chosen before
> the open-source release.

## Why

Most in-browser inference libraries are black boxes by design: the numerics
live in WebAssembly or WebGPU kernels, and what comes out is a token. That is
the right trade-off for speed and the wrong one for *understanding*. This
library makes the opposite trade-off:

- **Auditable.** No WASM, no WebGPU, no dependencies. `src/model.ts` is the
  whole architecture; `src/ops.ts` is the whole math (LayerNorm, linear,
  gelu_new, softmax). A reader can check every line against the paper or the
  PyTorch source.
- **Token-exact.** `tools/reference.json` is dumped from transformers'
  `GPTNeoForCausalLM` on the same checkpoint; the test suite holds the
  TypeScript pass to it — identical 12-token greedy continuation, logits and
  attention maps within fp16 tolerance.
- **Everything is exposed.** Embeddings, hidden states, attention matrices
  and logits are first-class return values, not debug hooks.
- **Small enough to ship.** 7.5 MB of fp16 weights; a forward pass over a
  sentence takes milliseconds on the main thread.

It is *not* fast, and it is *not* a general inference runtime. It runs one
model family (GPT-Neo) and ships one tiny checkpoint. Use it to teach, to
probe, to illustrate — or as the reference against which you check a faster
implementation.

## Install

```sh
npm install nano-lm   # once published — see status note above
```

Until then, use it locally: `"nano-lm": "file:../nano-lm"` in your
`package.json` (run `npm run build` here first; the package resolves to
`dist/`).

## Quickstart — browser

The library takes **token ids**. Bring your own tokenizer (see
[Tokenizers](#tokenizers)); the bundled weights expect GPT-2 BPE ids.

```ts
import { loadModel, softmaxTopK, diagnoseHeads } from "nano-lm";

// Self-host weights/tinystories-1m.safetensors (see "Weights" below) and
// point at it. meta defaults to the bundled checkpoint's architecture.
const model = await loadModel("/weights/tinystories-1m.safetensors", undefined, {
  onProgress: (pct) => console.log(`${pct}%`),
});

const ids = [7454, 2402, 257, 640]; // "Once upon a time" in GPT-2 BPE
const { logits, attentions, hiddenStates, seq } = model.forward(ids);

softmaxTopK(logits, 5, 1.0);          // [{ id: 11, p: 0.70 }, { id: 612, p: 0.29 }, …]  ("," / " there")
attentions[0][0][1 * seq + 0];        // layer 0, head 0: how much token 1 attends to token 0
hiddenStates[3];                      // residual stream after block 2, row-major [seq, hidden]
diagnoseHeads(attentions, seq);       // { prevToken: {layer, head, score}, firstToken, diffuse }
model.generate(ids, 12);              // greedy: […, 11, 612, 373, 257, 1310, 2576, 3706, 20037, 13, 1375, 6151, 284]
```

## Quickstart — Node

Node's `fetch` does not serve `file:` URLs, so read the bundled file yourself
and hand the bytes over:

```ts
import { readFile } from "node:fs/promises";
import { fileURLToPath } from "node:url";
import { bundledWeightsURL, createModel, TINYSTORIES_1M_META, nearestNeighbors } from "nano-lm";

const bytes = await readFile(fileURLToPath(bundledWeightsURL()));
const model = createModel(bytes, TINYSTORIES_1M_META); // synchronous

// Nearest neighbours of " dragon" (GPT-2 id 10441): " knight" 0.76, " monster" 0.75, …
nearestNeighbors(model.wte, model.meta.hidden, 10441, 8, { minId: 256, maxId: 30000 });
```

## API

Everything is exported from the package root. Arrays are `Float32Array`,
row-major.

| Export | Signature | What it does |
| --- | --- | --- |
| `loadModel` | `(weights: string \| URL \| ArrayBuffer \| Uint8Array \| Map<string, NanoTensor>, meta?: string \| URL \| NanoMeta, opts?: { onProgress?, fetch? }) => Promise<NanoGPT>` | Fetches URLs (with download progress), parses bytes. `meta` defaults to `TINYSTORIES_1M_META`. |
| `createModel` | `(weights: ArrayBuffer \| Uint8Array \| Map<string, NanoTensor>, meta: NanoMeta) => NanoGPT` | Synchronous constructor from bytes you already hold. |
| `NanoGPT#forward` | `(ids: number[]) => ForwardResult` | Full forward pass. `logits` for the **last position** `[vocab]`; `attentions[layer][head]` as `[seq, seq]`; `hiddenStates[0..layers]` as `[seq, hidden]` (embeddings first, residual stream after each block, none final-normed); `seq`. |
| `NanoGPT#generate` | `(ids: number[], steps: number) => number[]` | Greedy decoding; returns prompt + continuation. |
| `NanoGPT#wte` / `#wpe` | `Float32Array` | Token `[vocab, hidden]` / position `[maxPos, hidden]` embedding tables. |
| `NanoGPT#tensor` / `#tensorNames` | `(name) => NanoTensor \| undefined` / `() => string[]` | Any raw tensor by its HF state-dict name. |
| `NanoGPT#meta` | `NanoMeta` | `{ hidden, layers, heads, vocab, maxPos, lnEps }`. |
| `parseSafetensors` | `(bytes: ArrayBuffer \| Uint8Array) => Map<string, NanoTensor>` | Minimal safetensors reader, F16 → Float32. |
| `f16ToF32`, `toArrayBuffer` | | Half-precision decode; whole-buffer normalisation of byte views. |
| `fetchWithProgress` | `(url, onPct, fetch?) => Promise<ArrayBuffer>` | Streaming download with integer-percent callbacks. |
| `bundledWeightsURL`, `bundledMetaURL` | `() => URL` | `file:` URLs of the shipped weights, resolved from the package. |
| `TINYSTORIES_1M`, `TINYSTORIES_1M_META` | | Provenance record / architecture constants of the bundled checkpoint. |
| `nearestNeighbors` | `(wte, hidden, tokenId, k, { minId?, maxId?, minNormRatio? }) => Neighbor[]` | Cosine nearest neighbours in the embedding table, self excluded; skips near-zero-norm (untrained) rows. |
| `diagnoseHeads` | `(attentions, seq) => HeadDiagnostics \| null` | Finds the head that most attends to the previous token, to the first token, and the most diffuse (highest mean entropy) head. `null` for `seq < 3`. |
| `softmaxTopK` | `(logits, k, temperature) => TokenProb[]` | Temperature softmax restricted to the top-k logits. |
| `sampleFrom` | `(dist: TokenProb[], rand: number) => TokenProb` | Inverse-CDF pick with an injected uniform `rand ∈ [0,1)`. |
| `sampleNext` | `(logits, { k?, temperature?, rand? }) => TokenProb` | `sampleFrom(softmaxTopK(…))` in one call. |
| `softmax`, `topK`, `argmax` | | Full-vocab softmax; top-k ids; greedy pick. |
| `layerNorm`, `linear`, `geluNew`, `softmaxRow` | | The tensor primitives, for writing new probes. |
| `Tokenizer` (type) | `{ encode(text) => ids; decode(ids) => string }` | The contract for whatever tokenizer you pair with the model (sync or async). |

## Weights

**Bundled:** `weights/tinystories-1m.safetensors` (7,502,858 bytes, 108
tensors, fp16) and `weights/meta.json`, converted from
[`roneneldan/TinyStories-1M`](https://huggingface.co/roneneldan/TinyStories-1M)
— a GPT-Neo model (8 layers, 16 heads, hidden 64, 50257-token GPT-2
vocabulary, 2048 positions) trained on the TinyStories dataset of short
children's stories. The checkpoint and dataset are research releases by Ronen
Eldan and Yuanzhi Li ([Eldan & Li, 2023, "TinyStories: How Small Can
Language Models Be and Still Speak Coherent English?"](https://arxiv.org/abs/2305.07759));
`meta.json` records the license as stated on the model card. Check the
upstream card before redistributing under different terms.

The files are committed directly in git (no LFS): 7.5 MB is small enough, and
it keeps `git clone` + `npm test` self-contained.

**Self-hosting (browser):** copy the two files to a static directory of your
app (the essay uses `public/weights/`) and pass the URL to `loadModel`. Only
the `.safetensors` file is required — `meta` defaults to the bundled
architecture. With Vite you can also import the file URL straight from the
package: `import url from "nano-lm/weights/tinystories-1m.safetensors?url"`.
Serve it with `Content-Length` to get download progress.

**Other checkpoints:** any GPT-Neo state dict converted with
`tools/convert_weights.py` works — pass its own `meta`. See
[tools/README.md](./tools/README.md) for the conversion pipeline and how the
reference fixture is regenerated.

## Tokenizers

The library never sees text. The bundled weights index their embedding table
by the **GPT-2 byte-level BPE** vocabulary; ids from any other vocabulary are
meaningless to them. Inside the Machine pairs them with `Xenova/gpt2` via
[transformers.js](https://github.com/huggingface/transformers.js):

```ts
import { AutoTokenizer } from "@huggingface/transformers";
const tok = await AutoTokenizer.from_pretrained("Xenova/gpt2");
const enc = await tok("Once upon a time", { add_special_tokens: false });
const ids = Array.from(enc.input_ids.data as BigInt64Array, Number); // [7454, 2402, 257, 640]
```

That dependency belongs to your app, not to this package, which has **zero
runtime dependencies**. (`tools/` keeps the checkpoint's `tokenizer.json` etc.
out of git; they are only needed to regenerate the fixture.)

## Known semantics

Deliberate behaviours you should know about before building on this package:

- **No 1/√d attention scaling — on purpose.** GPT-Neo (unlike GPT-2) does not
  divide attention scores by `sqrt(head_dim)`. The implementation reproduces
  this quirk because the checkpoint was trained with it; "fixing" it changes
  every output. `src/model.ts` marks the line.
- **Local-attention layers are treated as global.** GPT-Neo alternates global
  and local (window 256) attention layers. For inputs of ≤ 256 tokens the two
  are identical and this implementation is faithful; beyond 256 tokens the
  odd layers diverge from the reference. `forward` accepts up to `maxPos`
  (2048) ids but is only verified for short inputs.
- **fp16 weights, fp32 arithmetic.** Weights are stored in half precision and
  widened on load; all math is JavaScript doubles written into
  `Float32Array`s. Against the fp32 reference this gives token-exact greedy
  decoding on the fixture and logits within ~0.1 — not bit-exact floats.
- **Logits are returned for the last position only**, matching what next-token
  prediction needs. Hidden states and attention maps cover every position.
- **`hiddenStates` are the raw residual stream.** `hiddenStates[0]` is
  `wte[id] + wpe[pos]`; `hiddenStates[l + 1]` is after block `l`. None has
  `ln_f` applied — unlike Hugging Face's `output_hidden_states`, whose last
  entry is final-normed.
- **`softmaxTopK`/`softmax` floor the temperature at 0.05** to avoid
  division blow-ups; `sampleFrom` returns the last entry if `rand` rounds past
  the cumulative mass.
- **`nearestNeighbors` skips "untrained" rows** whose norm is below
  `minNormRatio` (default 0.35) × the query norm — in a 1M-parameter model
  most of the 50257 rows never moved from their random init. The essay also
  passes `minId: 256, maxId: 30000` to skip byte tokens and rare tails.
- **`diagnoseHeads` uses natural-log entropy** averaged over query positions
  1..seq−1 and returns `null` for sequences shorter than 3 tokens.
- **Safetensors reader: F16 only**, and it tolerates an unpadded JSON header
  (the conversion script does not 8-byte-align it) by copying misaligned
  tensor data into an aligned buffer.

## Development

```sh
npm install
npm test          # vitest — includes the token-exact reference tests
npm run typecheck
npm run build     # emits ESM + .d.ts to dist/
```

Tests are Node-only (no DOM): `test/nano.test.ts` is the reference-fixture
suite, `test/api.test.ts` covers the public surface and the loaders, and the
rest unit-test the ops, safetensors reader, neighbour search, head
diagnostics and sampling helpers.

## Citation

See [CITATION.cff](./CITATION.cff). Extracted from *Inside the Machine: An
Interactive Guide to How LLMs Actually Think* by Shangyan Shen; when citing
the essay's in-browser model, cite this package.
