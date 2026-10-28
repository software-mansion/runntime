import { describe, expect, it } from 'vitest';
import { isRunntimeError } from '../../../src/core/index.ts';
import { createMoonshineTokenizer } from '../../../src/zoo/moonshine/tokenizer.ts';

/** The smallest BPE tokenizer.json tokenizers.js loads, with Moonshine's
 *  decoder chain: `▁` → space, pieces fused, one leading space stripped. */
const json = {
  added_tokens: [
    { id: 1, content: '<s>', special: true },
    { id: 2, content: '</s>', special: true },
  ],
  normalizer: null,
  pre_tokenizer: null,
  post_processor: null,
  model: {
    type: 'BPE',
    vocab: { '<s>': 1, '</s>': 2, '▁hello': 3, '▁world': 4, '▁.': 5 },
    merges: [],
  },
  decoder: {
    type: 'Sequence',
    decoders: [
      { type: 'Replace', pattern: { String: '▁' }, content: ' ' },
      { type: 'Fuse' },
      { type: 'Strip', content: ' ', start: 1, stop: 0 },
    ],
  },
};

describe('createMoonshineTokenizer', () => {
  const tokenizer = createMoonshineTokenizer(json);

  it('reads </s> from the vocab', () => {
    expect(tokenizer.eosId).toBe(2);
  });

  it('skips special ids and leaves the text as is', () => {
    // With tokenizers.js's default clean-up the space before '.' would go.
    expect(tokenizer.decode([1, 3, 4, 5, 2])).toBe('hello world .');
  });

  it.each([42, {}, { model: { type: 'WordPiece' } }])(
    'throws CHECKPOINT_MISMATCH for %j',
    (bad) => {
      expect(() => createMoonshineTokenizer(bad)).toThrow(
        expect.toSatisfy(
          (err: unknown) => isRunntimeError(err) && err.code === 'CHECKPOINT_MISMATCH',
        ),
      );
    },
  );
});
