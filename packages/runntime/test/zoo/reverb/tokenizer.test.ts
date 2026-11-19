import { describe, expect, it } from 'vitest';
import { createReverbTokenizer } from '../../../src/zoo/reverb/tokenizer.ts';

const MARKERS = [
  '<|denoise|>',
  '<|lang_start|>',
  '<|lang_end|>',
  '<|instruct_start|>',
  '<|instruct_end|>',
  '<|text_start|>',
  '<|text_end|>',
];

/** A character-level BPE tokenizer.json with Reverb's markers as added tokens
 *  from id 10. */
const json = (markers = MARKERS) => ({
  added_tokens: markers.map((content, i) => ({ id: 10 + i, content, special: true })),
  normalizer: null,
  pre_tokenizer: null,
  post_processor: null,
  decoder: null,
  model: {
    type: 'BPE',
    vocab: {
      a: 0,
      b: 1,
      ...Object.fromEntries(markers.map((m, i) => [m, 10 + i])),
    },
    merges: [],
  },
});

describe('createReverbTokenizer', () => {
  it('reads the markers as single tokens and adds no special tokens', () => {
    const tokenizer = createReverbTokenizer(json());
    expect(tokenizer.encode('<|text_start|>ab<|text_end|>')).toEqual([15, 0, 1, 16]);
  });

  it.each([
    ['a tokenizer.json without a marker', json(MARKERS.slice(1))],
    ['something else', { vocab: {} }],
  ])('throws CHECKPOINT_MISMATCH for %s', (_, file) => {
    expect(() => createReverbTokenizer(file)).toThrow(
      expect.objectContaining({ code: 'CHECKPOINT_MISMATCH' }),
    );
  });
});
