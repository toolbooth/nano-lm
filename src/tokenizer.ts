/**
 * The library is tokenizer-agnostic: every model entry point takes and returns
 * token ids. Bring any tokenizer that satisfies this shape — sync or async.
 *
 * The bundled TinyStories-1M weights index their embedding table by the GPT-2
 * byte-level BPE vocabulary (50257 ids, `<|endoftext|>` = 50256). Ids from any
 * other vocabulary are meaningless to them. The Inside the Machine essay pairs
 * the weights with `Xenova/gpt2` via transformers.js:
 *
 * ```ts
 * import { AutoTokenizer } from "@huggingface/transformers";
 * const tok = await AutoTokenizer.from_pretrained("Xenova/gpt2");
 * const gpt2: Tokenizer = {
 *   encode: async (text) =>
 *     Array.from((await tok(text, { add_special_tokens: false })).input_ids.data as BigInt64Array, Number),
 *   decode: (ids) => tok.decode(ids),
 * };
 * ```
 */
export interface Tokenizer {
  encode(text: string): number[] | Promise<number[]>;
  decode(ids: readonly number[]): string | Promise<string>;
}
