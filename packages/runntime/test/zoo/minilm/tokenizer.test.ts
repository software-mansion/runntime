import { beforeAll, describe, expect, it } from 'vitest';
import { isRunntimeError } from '../../../src/core/index.ts';
import { createMinilmTokenizer, type MinilmTokenizer } from '../../../src/zoo/minilm/tokenizer.ts';
import { models } from '../../../src/zoo/models.ts';

const CLS = 101;
const SEP = 102;
const UNK = 100;

/** Ids of HF BertTokenizer for all-MiniLM-L6-v2 on the same text. */
const cases: [string, number[]][] = [
  ['Hello world', [7592, 2088]],
  ['Café', [7668]],
  ['résumé niño Crème brûlée', [13746, 20801, 13675, 21382, 7987, 9307, 2063]],
  ['ÀÉÎÕÜ ß Straße', [29347, 3695, 2226, 1096, 2358, 27807]],
  // Every CJK ideograph is its own word.
  ['我爱北京天安门', [1855, UNK, 1781, 1755, 1811, 1820, UNK]],
  ['こんにちは世界', [1655, 30217, 30194, 30188, 30198, 1745, UNK]],
  [
    '안녕하세요',
    [1463, 30006, 30021, 29992, 30010, 30025, 30005, 30006, 29997, 30009, 29999, 30013],
  ],
  ['I love it 😀👍', [1045, 2293, 2009, UNK]],
  ['🇵🇱 flag', [UNK, 5210]],
  [
    "Don't stop—it's 3:30 p.m.!? (really)...",
    [
      2123, 1005, 1056, 2644, 1517, 2009, 1005, 1055, 1017, 1024, 2382, 1052, 1012, 1049, 1012, 999,
      1029, 1006, 2428, 1007, 1012, 1012, 1012,
    ],
  ],
  [
    'e-mail: foo@bar.com, "quoted" [x] {y} <z>',
    [
      1041, 1011, 5653, 1024, 29379, 1030, 3347, 1012, 4012, 1010, 1000, 9339, 1000, 1031, 1060,
      1033, 1063, 1061, 1065, 1026, 1062, 1028,
    ],
  ],
  ['COVID-19 vaccines 2021!', [2522, 17258, 1011, 2539, 28896, 25682, 999]],
  ['  a   b\n\nc\t\td  ', [1037, 1038, 1039, 1040]],
  ['line1\r\nline2', [2240, 2487, 2240, 2475]],
  ['a b c', [1037, 1038, 1039]],
  ['', []],
  ['   ', []],
  // NUL, control characters and U+FFFD are dropped.
  ['a\u0000b\u0007c�', [5925]],
  ['Ⓐ ✓ ∑ ‰ ①', [UNK, UNK, UNK, 1530, UNK]],
  ['ﬁ ﬂ ①② Ⅳ', [1984, 1985, UNK, UNK]],
  // A word past 100 characters is one [UNK]; 100 still splits.
  ['x'.repeat(101), [UNK]],
  ['x'.repeat(100), [22038, ...Array<number>(49).fill(20348)]],
];

describe('createMinilmTokenizer', () => {
  let json: unknown;
  beforeAll(async () => {
    const res = await fetch(models.embedding.ALL_MINILM_L6_V2.DEFAULT.tokenizerPath);
    json = await res.json();
  });

  it('reads [CLS] and [SEP] from the vocab', () => {
    const tokenizer = createMinilmTokenizer(json);
    expect(tokenizer.clsId).toBe(CLS);
    expect(tokenizer.sepId).toBe(SEP);
  });

  describe('encode', () => {
    let tokenizer: MinilmTokenizer;
    beforeAll(() => {
      tokenizer = createMinilmTokenizer(json);
    });

    it.each(cases)('%j', (text, pieces) => {
      expect(tokenizer.encode(text)).toEqual([CLS, ...pieces, SEP]);
    });

    it('keeps the first 254 pieces of a long text', () => {
      const sentence = [1996, 4248, 2829, 4419, 14523, 2058, 1996, 13971, 3899, 1012];
      const ids = tokenizer.encode('The quick brown fox jumps over the lazy dog. '.repeat(40));
      expect(ids).toEqual([CLS, ...Array<number[]>(26).fill(sentence).flat().slice(0, 254), SEP]);
    });

    it('cuts at maxTokens', () => {
      const ids = createMinilmTokenizer(json, 8).encode('word '.repeat(300));
      expect(ids).toEqual([CLS, ...Array<number>(6).fill(2773), SEP]);
    });
  });

  it.each([
    ['not an object', 42],
    ['no model', {}],
    ['a BPE model', { model: { type: 'BPE' } }],
  ])('throws CHECKPOINT_MISMATCH for %s', (_what, bad) => {
    expect(() => createMinilmTokenizer(bad)).toThrow(
      expect.toSatisfy(
        (err: unknown) => isRunntimeError(err) && err.code === 'CHECKPOINT_MISMATCH',
      ),
    );
  });
});
