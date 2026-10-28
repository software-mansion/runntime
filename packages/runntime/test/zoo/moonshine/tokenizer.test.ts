import { beforeAll, describe, expect, it } from 'vitest';
import { isRunntimeError } from '../../../src/core/index.ts';
import { models } from '../../../src/zoo/models.ts';
import {
  createMoonshineTokenizer,
  type MoonshineTokenizer,
} from '../../../src/zoo/moonshine/tokenizer.ts';

/** Text of HF's Moonshine tokenizer with skip_special_tokens on the same ids.
 *  1 is <s>, 2 is </s>. */
const cases: [string, number[]][] = [
  ['Hello world.', [1, 15043, 3186, 29889, 2]],
  [
    "It's 3:30 p.m., don't be late!",
    [
      1, 739, 29915, 29879, 29871, 29941, 29901, 29941, 29900, 282, 29889, 29885, 1696, 1016, 29915,
      29873, 367, 5683, 29991, 2,
    ],
  ],
  ['café résumé niño', [1, 274, 28059, 6896, 398, 29948, 6836, 8266, 2]],
  ['北京 东京', [1, 29871, 30662, 30675, 29871, 30979, 30675, 2]],
  // Each emoji is four byte-fallback tokens.
  ['I love it 😀👍', [1, 306, 5360, 372, 29871, 243, 162, 155, 131, 243, 162, 148, 144, 2]],
  ['a  b   c', [1, 263, 29871, 289, 259, 274, 2]],
  ['line1\nline2\ttab', [1, 1196, 29896, 13, 1220, 29906, 12, 3891, 2]],
  [
    '12345 $6.78 100%',
    [
      1, 29871, 29896, 29906, 29941, 29946, 29945, 395, 29953, 29889, 29955, 29947, 29871, 29896,
      29900, 29900, 29995, 2,
    ],
  ],
  ['ĐŽ ß ÿ', [1, 29871, 30250, 30144, 29871, 30034, 29871, 30370, 2]],
  // Only the sentencepiece prefix of the first word is stripped.
  [' leading space', [1, 29871, 8236, 2913, 2]],
  ['trailing space ', [1, 25053, 2913, 29871, 2]],
  ['Hello world', [15043, 3186]],
  ['Hello world', [1, 15043, 3186]],
  ['', [1, 2]],
  ['', [1]],
  // The <<ST_n>> ids are special too.
  ['', [1, 32000, 32005, 2]],
  ['', [1, 29871, 2]],
  // A byte token that is no UTF-8 sequence on its own.
  ['�', [1, 234, 2]],
];

describe.each([
  ['moonshine', models.speechToText.MOONSHINE.TINY.DEFAULT.tokenizerPath],
  ['moonshine-streaming', models.speechToText.MOONSHINE_STREAMING.TINY.DEFAULT.tokenizerPath],
])('createMoonshineTokenizer for %s', (_repo, tokenizerPath) => {
  let tokenizer: MoonshineTokenizer;
  beforeAll(async () => {
    const res = await fetch(tokenizerPath);
    tokenizer = createMoonshineTokenizer(await res.json());
  });

  it('reads </s> from the vocab', () => {
    expect(tokenizer.eosId).toBe(2);
  });

  it.each(cases)('decodes %j', (text, ids) => {
    expect(tokenizer.decode(ids)).toBe(text);
  });
});

it.each([
  ['not an object', 42],
  ['no model', {}],
  ['a WordPiece model', { model: { type: 'WordPiece' } }],
])('createMoonshineTokenizer throws CHECKPOINT_MISMATCH for %s', (_what, bad) => {
  expect(() => createMoonshineTokenizer(bad)).toThrow(
    expect.toSatisfy((err: unknown) => isRunntimeError(err) && err.code === 'CHECKPOINT_MISMATCH'),
  );
});
