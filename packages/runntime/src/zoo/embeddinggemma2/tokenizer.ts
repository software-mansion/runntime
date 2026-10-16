/** Gemma tokenizer, encode-only: text → piece ids, without <bos> and <eos>,
 *  which the caller places. tokenizers.js runs the Hub's tokenizer.json. */

import { Tokenizer } from '@huggingface/tokenizers';
import { checkpointMismatch } from '../errors.ts';

export interface EmbeddingGemma2Tokenizer {
  encode(text: string): number[];
  readonly bosId: number;
  readonly eosId: number;
  readonly padId: number;
  /** Start and end of an image or of a video frame. */
  readonly boiId: number;
  readonly eoiId: number;
  /** Start and end of an audio clip. */
  readonly boaId: number;
  readonly eoaId: number;
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
 *  BPE kind. Throws CHECKPOINT_MISMATCH for any other file. */
export function createEmbeddingGemma2Tokenizer(json: unknown): EmbeddingGemma2Tokenizer {
  const bad = (why: string) => checkpointMismatch('embeddinggemma2 tokenizer', why);
  if (!isBpeTokenizerJson(json)) throw bad('expected a BPE tokenizer.json');
  const tokenizer = new Tokenizer(json, {});
  const idOf = (token: string): number => {
    const id = tokenizer.token_to_id(token);
    if (id === undefined) throw bad(`vocab has no ${token}`);
    return id;
  };
  return {
    bosId: idOf('<bos>'),
    eosId: idOf('<eos>'),
    padId: idOf('<pad>'),
    boiId: idOf('<|image>'),
    eoiId: idOf('<image|>'),
    boaId: idOf('<|audio>'),
    eoaId: idOf('<audio|>'),
    encode: (text) => tokenizer.encode(text, { add_special_tokens: false }).ids,
  };
}
