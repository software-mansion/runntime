/** Reverb's text tokenizer, encode-only: Qwen2's byte-level BPE cut to Reverb's
 *  vocabulary, with its prompt markers as added tokens. tokenizers.js runs the
 *  release's tokenizer.json. */

import { Tokenizer } from '@huggingface/tokenizers';
import { checkpointMismatch } from '../errors.ts';

export interface ReverbTokenizer {
  /** Text to ids, prompt markers included, no special tokens added. */
  encode(text: string): number[];
}

const MARKERS = [
  '<|denoise|>',
  '<|lang_start|>',
  '<|lang_end|>',
  '<|instruct_start|>',
  '<|instruct_end|>',
  '<|text_start|>',
  '<|text_end|>',
];

/** Builds the tokenizer from the release's tokenizer.json. Throws
 *  CHECKPOINT_MISMATCH for a file without Reverb's prompt markers. */
export function createReverbTokenizer(json: unknown): ReverbTokenizer {
  const bad = (why: string) => checkpointMismatch('reverb tokenizer', why);
  if (typeof json !== 'object' || json === null || !('model' in json)) {
    throw bad('expected a tokenizer.json');
  }
  const tokenizer = new Tokenizer(json as object, {});
  for (const m of MARKERS) {
    if (tokenizer.token_to_id(m) === undefined) throw bad(`vocab has no ${m}`);
  }
  return { encode: (text) => tokenizer.encode(text, { add_special_tokens: false }).ids };
}
