import { describe, expect, it } from 'vitest';
import { isRunntimeError } from '../../../src/core/index.ts';
import { createMinilmTokenizer, MINILM_MAX_WORDPIECES } from '../../../src/zoo/minilm/tokenizer.ts';

/** The smallest WordPiece tokenizer.json tokenizers.js loads. */
const json = {
  added_tokens: [
    { id: 1, content: '[UNK]', special: true },
    { id: 2, content: '[CLS]', special: true },
    { id: 3, content: '[SEP]', special: true },
  ],
  normalizer: { type: 'BertNormalizer', lowercase: true },
  pre_tokenizer: { type: 'BertPreTokenizer' },
  post_processor: null,
  decoder: null,
  model: {
    type: 'WordPiece',
    unk_token: '[UNK]',
    continuing_subword_prefix: '##',
    vocab: { '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, hello: 4, world: 5 },
  },
};

describe('createMinilmTokenizer', () => {
  const tokenizer = createMinilmTokenizer(json);

  it('reads [CLS] and [SEP] from the vocab', () => {
    expect(tokenizer.clsId).toBe(2);
    expect(tokenizer.sepId).toBe(3);
  });

  it('wraps the pieces in [CLS] and [SEP]', () => {
    expect(tokenizer.encode('hello world')).toEqual([2, 4, 5, 3]);
  });

  it('keeps maxTokens − 2 pieces', () => {
    expect(createMinilmTokenizer(json, 4).encode('hello world hello')).toEqual([2, 4, 5, 3]);
    const ids = tokenizer.encode('hello '.repeat(MINILM_MAX_WORDPIECES));
    expect(ids).toHaveLength(MINILM_MAX_WORDPIECES);
  });

  it.each([42, {}, { model: { type: 'BPE' } }])('throws CHECKPOINT_MISMATCH for %j', (bad) => {
    expect(() => createMinilmTokenizer(bad)).toThrow(
      expect.toSatisfy(
        (err: unknown) => isRunntimeError(err) && err.code === 'CHECKPOINT_MISMATCH',
      ),
    );
  });
});
