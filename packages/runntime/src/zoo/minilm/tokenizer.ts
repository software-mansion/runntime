/** BERT WordPiece tokenizer, encode-only: text → [CLS] pieces [SEP] ids,
 *  truncated to maxTokens. tokenizers.js runs the Hub's tokenizer.json. */

import { Tokenizer } from '@huggingface/tokenizers';
import { checkpointMismatch } from '../errors.ts';

export interface MinilmTokenizer {
  encode(text: string): number[];
  readonly clsId: number;
  readonly sepId: number;
}

/** Tokens per text, [CLS] and [SEP] included, when the caller sets no limit. */
export const MINILM_MAX_WORDPIECES = 256;

const isWordPieceTokenizerJson = (json: unknown): json is object =>
  typeof json === 'object' &&
  json !== null &&
  'model' in json &&
  typeof json.model === 'object' &&
  json.model !== null &&
  'type' in json.model &&
  json.model.type === 'WordPiece';

/** Builds the tokenizer from a Hugging Face tokenizer.json of the WordPiece
 *  kind. Throws CHECKPOINT_MISMATCH for any other file. */
export function createMinilmTokenizer(
  json: unknown,
  maxTokens = MINILM_MAX_WORDPIECES,
): MinilmTokenizer {
  const bad = (why: string) => checkpointMismatch('minilm tokenizer', why);
  if (!isWordPieceTokenizerJson(json)) throw bad('expected a WordPiece tokenizer.json');
  const tokenizer = new Tokenizer(json, {});
  const idOf = (token: string): number => {
    const id = tokenizer.token_to_id(token);
    if (id === undefined) throw bad(`vocab has no ${token}`);
    return id;
  };
  const clsId = idOf('[CLS]');
  const sepId = idOf('[SEP]');
  return {
    clsId,
    sepId,
    encode(text) {
      const pieces = tokenizer.encode(text, { add_special_tokens: false }).ids;
      return [clsId, ...pieces.slice(0, maxTokens - 2), sepId];
    },
  };
}
