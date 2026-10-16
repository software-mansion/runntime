/** Gemma tokenizer, encode-only: text → <bos> pieces <eos> ids, truncated
 *  to maxTokens. tokenizers.js runs the Hub's tokenizer.json. */

import { Tokenizer } from '@huggingface/tokenizers';
import { checkpointMismatch } from '../errors.ts';
import { EMBEDDINGGEMMA2_MAX_TOKENS } from './config.ts';

export interface EmbeddingGemma2Tokenizer {
  encode(text: string): number[];
}

const isBpeTokenizerJson = (json: unknown): json is object =>
  typeof json === 'object' &&
  json !== null &&
  'model' in json &&
  typeof json.model === 'object' &&
  json.model !== null &&
  'type' in json.model &&
  json.model.type === 'BPE';

/** Builds the tokenizer from a Hugging Face tokenizer.json of the Gemma
 *  BPE kind. `maxTokens` defaults to EMBEDDINGGEMMA2_MAX_TOKENS. Throws
 *  CHECKPOINT_MISMATCH for any other file. */
export function createEmbeddingGemma2Tokenizer(
  json: unknown,
  maxTokens = EMBEDDINGGEMMA2_MAX_TOKENS,
): EmbeddingGemma2Tokenizer {
  const bad = (why: string) => checkpointMismatch('embeddinggemma2 tokenizer', why);
  if (!isBpeTokenizerJson(json)) throw bad('expected a BPE tokenizer.json');
  const tokenizer = new Tokenizer(json, {});
  const idOf = (token: string): number => {
    const id = tokenizer.token_to_id(token);
    if (id === undefined) throw bad(`vocab has no ${token}`);
    return id;
  };
  const bosId = idOf('<bos>');
  const eosId = idOf('<eos>');
  return {
    encode(text) {
      const { ids } = tokenizer.encode(text, { add_special_tokens: false });
      return [bosId, ...ids.slice(0, maxTokens - 2), eosId];
    },
  };
}
