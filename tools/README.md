# tools/ — weight conversion and reference fixtures

The package ships `weights/tinystories-1m.safetensors` + `weights/meta.json`
and the test fixture `tools/reference.json`. These two scripts produced them
from the Hugging Face checkpoint; re-run them only if you change the
checkpoint or want to regenerate the fixture.

## Inputs (not committed)

Download from https://huggingface.co/roneneldan/TinyStories-1M into this
directory: `config.json` (committed for reference), `pytorch_model.bin`
(~48 MB), and the tokenizer files `tokenizer.json`, `tokenizer_config.json`,
`special_tokens_map.json`, `vocab.json`, `merges.txt`. All but `config.json`
are git-ignored.

```sh
cd tools
python3 -m venv .venv && source .venv/bin/activate
pip install torch transformers
```

## 1. `convert_weights.py` → `../weights/`

Writes an fp16 safetensors file **by hand** (no `safetensors` dependency) and a
`meta.json` with the architecture constants the TypeScript side needs
(`hidden`, `layers`, `heads`, `vocab`, `maxPos`, `lnEps`). The causal-mask
buffers (`attn.attention.bias` / `masked_bias`) are skipped — the mask is
recomputed in TypeScript. 108 tensors, 7.5 MB.

Note: the JSON header is not padded to 8 bytes as the reference `safetensors`
writer does. The TypeScript parser handles both (it copies to an aligned
buffer when the header length is odd).

## 2. `export_reference.py` → `reference.json`

Runs transformers' `GPTNeoForCausalLM` (fp32) on `"Once upon a time"` and
dumps: input ids, top-10 logits, first 8 logits, attention matrices for
layer 0 head 0 and layer 7 head 3, and the 12-token greedy continuation
(`"Once upon a time, there was a little girl named Lily. She loved to"`).
`test/nano.test.ts` and `test/api.test.ts` hold the TypeScript forward pass
to this fixture: greedy ids must match exactly; logits and attention within
fp16 tolerance.

```sh
python convert_weights.py    # tensors: 108, file: 7.5 MB
python export_reference.py   # greedy: Once upon a time, there was a little girl named Lily. She loved to
```
