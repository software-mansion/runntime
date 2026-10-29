import { describe, expect, it } from 'vitest';
import { createMinilmTokenizer } from '../../../src/zoo/minilm/tokenizer.ts';

/** The smallest WordPiece tokenizer.json tokenizers.js loads. */
const json = {
  added_tokens: [
    { id: 1, content: '[UNK]', special: true },
    { id: 2, content: '[CLS]', special: true },
    { id: 3, content: '[SEP]', special: true },
  ],
  normalizer: { type: 'BertNormalizer', lowercase: true },
  pre_tokenizer: { type: 'BertPreTokenizer' },
  post_processor: { type: 'BertProcessing', sep: ['[SEP]', 3], cls: ['[CLS]', 2] },
  decoder: null,
  model: {
    type: 'WordPiece',
    unk_token: '[UNK]',
    continuing_subword_prefix: '##',
    vocab: { '[PAD]': 0, '[UNK]': 1, '[CLS]': 2, '[SEP]': 3, hello: 4, world: 5 },
  },
};

describe('createMinilmTokenizer', () => {
  it('wraps the pieces in [CLS] and [SEP] and keeps maxTokens − 2 of them', () => {
    expect(createMinilmTokenizer(json, 4).encode('hello world hello')).toEqual([2, 4, 5, 3]);
  });
});
