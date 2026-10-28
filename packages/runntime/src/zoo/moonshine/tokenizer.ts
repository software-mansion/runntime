/** Moonshine tokenizer, decode-only: the decoder starts from BOS and emits
 *  ids, so id → text is the whole job. tokenizers.js runs the Hub's Llama-style
 *  BPE tokenizer.json with its special ids skipped and the text left as is. */

import { Tokenizer } from '@huggingface/tokenizers';
import { checkpointMismatch } from '../errors.ts';

export interface MoonshineTokenizer {
  /** Text of the ids, special ids skipped. `ids` is never empty. */
  decode(ids: number[]): string;
  readonly eosId: number;
}

const isBpeTokenizerJson = (json: unknown): json is object =>
  typeof json === 'object' &&
  json !== null &&
  'model' in json &&
  typeof json.model === 'object' &&
  json.model !== null &&
  'type' in json.model &&
  json.model.type === 'BPE';

/** Builds the tokenizer from a Hugging Face tokenizer.json of the BPE kind.
 *  Throws CHECKPOINT_MISMATCH for any other file. */
export function createMoonshineTokenizer(json: unknown): MoonshineTokenizer {
  const bad = (why: string) => checkpointMismatch('moonshine tokenizer', why);
  if (!isBpeTokenizerJson(json)) throw bad('expected a BPE tokenizer.json');
  const tokenizer = new Tokenizer(json, {});
  const eosId = tokenizer.token_to_id('</s>');
  if (eosId === undefined) throw bad('vocab has no </s>');
  return {
    eosId,
    decode: (ids) =>
      tokenizer.decode(ids, { skip_special_tokens: true, clean_up_tokenization_spaces: false }),
  };
}
