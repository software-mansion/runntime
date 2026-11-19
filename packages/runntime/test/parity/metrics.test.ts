/** The parity metrics on numbers worked out by hand. */
import { describe, expect, it } from 'vitest';
import {
  cosine,
  editDistance,
  maxAbsError,
  missing,
  rankingDifferences,
  rankMismatches,
  summarize,
  topK,
  wer,
  words,
} from './metrics.ts';

describe('cosine', () => {
  it.each([
    { name: 'the same direction', a: [1, 2, 3], b: [2, 4, 6], want: 1 },
    { name: 'a right angle', a: [1, 0], b: [0, 5], want: 0 },
    { name: 'opposite directions', a: [1, 1], b: [-3, -3], want: -1 },
    // 3·4 + 4·3 = 24 over lengths 5 and 5.
    { name: 'an angle in between', a: [3, 4], b: [4, 3], want: 24 / 25 },
  ])('is $want for $name', ({ a, b, want }) => {
    expect(cosine(a, b)).toBeCloseTo(want, 12);
  });

  it('is NaN for a zero vector', () => {
    expect(cosine([0, 0], [1, 2])).toBeNaN();
  });
});

describe('maxAbsError', () => {
  it('is the largest difference of either sign', () => {
    expect(maxAbsError([1, 2, 3], [1.5, 2, 1])).toBe(2);
  });
});

describe('summarize', () => {
  it('gives the mean, the 99th percentile and the max', () => {
    // 1..200: the 99th percentile is the 198th value.
    const values = Array.from({ length: 200 }, (_, i) => 200 - i);
    expect(summarize(values)).toEqual({ mean: 100.5, p99: 198, max: 200 });
  });

  it('takes the max as the 99th percentile of a short list', () => {
    expect(summarize([3, 1, 2])).toEqual({ mean: 2, p99: 3, max: 3 });
  });
});

describe('topK', () => {
  it('ranks the largest scores first', () => {
    expect(topK([0.1, 0.9, 0.5, 0.7], 3)).toEqual([1, 3, 2]);
  });

  it('keeps equal scores in index order', () => {
    expect(topK([0.5, 0.9, 0.5], 3)).toEqual([1, 0, 2]);
  });
});

describe('ranking differences', () => {
  // Reference scores by item: 0 and 1 are 0.25 apart, 2 and 3 are 0.125 apart.
  const scores = [1, 0.75, 0.5, 0.375];
  const want = [0, 1, 2];
  const none = { count: 0, nearTies: 0 };

  it.each([
    { name: 'the same ranking', got: [0, 1, 2], margin: 0, mismatches: none, missed: none },
    {
      name: 'two items swapped, further apart than the margin',
      got: [1, 0, 2],
      margin: 0.125,
      mismatches: { count: 2, nearTies: 0 },
      missed: none,
    },
    {
      name: 'two items swapped, as far apart as the margin',
      got: [1, 0, 2],
      margin: 0.25,
      mismatches: { count: 0, nearTies: 2 },
      missed: none,
    },
    {
      name: 'one item replaced, further apart than the margin',
      got: [0, 1, 3],
      margin: 0.0625,
      mismatches: { count: 1, nearTies: 0 },
      missed: { count: 1, nearTies: 0 },
    },
    {
      name: 'one item replaced, as far apart as the margin',
      got: [0, 1, 3],
      margin: 0.125,
      mismatches: { count: 0, nearTies: 1 },
      missed: { count: 0, nearTies: 1 },
    },
    {
      name: 'a NaN margin',
      got: [0, 1, 3],
      margin: NaN,
      mismatches: { count: 1, nearTies: 0 },
      missed: { count: 1, nearTies: 0 },
    },
  ])('$name', ({ got, margin, mismatches, missed }) => {
    expect(rankMismatches(got, want, scores, margin)).toEqual(mismatches);
    expect(missing(got, want, scores, margin)).toEqual(missed);
  });

  it('counts a NaN reference score as a difference', () => {
    const broken = [1, 0.75, 0.5, NaN];
    expect(rankMismatches([0, 1, 3], want, broken, 1)).toEqual({ count: 1, nearTies: 0 });
    expect(missing([0, 1, 3], want, broken, 1)).toEqual({ count: 1, nearTies: 0 });
  });
});

describe('rankingDifferences', () => {
  // Two asking vectors, each along one axis, so an item's score is its
  // value on that axis. Reference scores: 1, 0.75, 0.5, 0.375 for the first,
  // 0.375, 0.5, 0.75, 1 for the second.
  const want = {
    asking: [
      [1, 0],
      [0, 1],
    ],
    ranked: [
      [1, 0.375],
      [0.75, 0.5],
      [0.5, 0.75],
      [0.375, 1],
    ],
  };
  const ranking = [0, 1, 2, 3, 2, 1];
  const none = { count: 0, nearTies: 0 };

  it('finds none between the same vectors', () => {
    expect(rankingDifferences(want, want, ranking, 3, 0)).toEqual({ positions: none, items: none });
  });

  // The first asking vector now ranks 0, 1, 3: item 3 replaces item 2, which
  // the reference scores 0.125 higher. The second ranks 3, 1, 2: items 1 and
  // 2, 0.25 apart, swap places.
  const got = {
    asking: want.asking,
    ranked: [
      [1, 0.375],
      [0.75, 0.75],
      [0.25, 0.5],
      [0.375, 1],
    ],
  };

  it.each([
    {
      name: 'a margin below every gap',
      margin: 0.0625,
      positions: { count: 3, nearTies: 0 },
      items: { count: 1, nearTies: 0 },
    },
    {
      name: 'a margin at the smaller gap',
      margin: 0.125,
      positions: { count: 2, nearTies: 1 },
      items: { count: 0, nearTies: 1 },
    },
    {
      name: 'a margin at the larger gap',
      margin: 0.25,
      positions: { count: 0, nearTies: 3 },
      items: { count: 0, nearTies: 1 },
    },
  ])('sums the differences of every asking vector with $name', ({ margin, positions, items }) => {
    expect(rankingDifferences(got, want, ranking, 3, margin)).toEqual({ positions, items });
  });

  it('takes the gaps from the reference asking vectors', () => {
    // Each asking vector reads the other axis, so its top 3 is the
    // reference's reversed and misses the reference's best item. That item
    // is 0.625 above the one in its place by the reference scores, and
    // 0.625 below it by the scores of `got`.
    const swapped = { asking: [want.asking[1]!, want.asking[0]!], ranked: want.ranked };
    expect(rankingDifferences(swapped, want, ranking, 3, 0.5)).toEqual({
      positions: { count: 2, nearTies: 4 },
      items: { count: 2, nearTies: 0 },
    });
  });
});

describe('words', () => {
  it('lowers the case, drops punctuation and keeps apostrophes', () => {
    expect(words("Eleven o'clock had struck. It was fine, clear!")).toEqual([
      'eleven',
      "o'clock",
      'had',
      'struck',
      'it',
      'was',
      'fine',
      'clear',
    ]);
  });

  it('splits a hyphenated word', () => {
    expect(words('thirty-six  members')).toEqual(['thirty', 'six', 'members']);
  });
});

describe('editDistance', () => {
  it.each([
    { name: 'equal lists', a: ['a', 'b', 'c'], b: ['a', 'b', 'c'], want: 0 },
    { name: 'one substitution', a: ['a', 'x', 'c'], b: ['a', 'b', 'c'], want: 1 },
    { name: 'one insertion', a: ['a', 'b', 'x', 'c'], b: ['a', 'b', 'c'], want: 1 },
    { name: 'one deletion', a: ['a', 'c'], b: ['a', 'b', 'c'], want: 1 },
    { name: 'an empty list', a: [], b: ['a', 'b', 'c'], want: 3 },
    // kitten to sitting: two substitutions and one insertion.
    { name: 'a mix', a: [...'kitten'], b: [...'sitting'], want: 3 },
  ])('is $want for $name', ({ a, b, want }) => {
    expect(editDistance(a, b)).toBe(want);
  });
});

describe('wer', () => {
  it('is 0 for the same words in another case and punctuation', () => {
    expect(wer(['The cat sat.'], ['THE CAT SAT'])).toBe(0);
  });

  it('sums the edits and the reference words over the texts', () => {
    // One substitution in four words, one deletion in six: 2 / 10.
    expect(
      wer(['the cat sat here', 'a dog ran to the'], ['the cat sat there', 'a dog ran to the park']),
    ).toBe(0.2);
  });

  it('counts every reference word for an empty hypothesis', () => {
    expect(wer([''], ['one two three'])).toBe(1);
  });

  it('throws for another number of hypotheses than references', () => {
    expect(() => wer(['one'], ['one', 'two'])).toThrow('1 hypotheses for 2 references');
  });

  it('is NaN when neither side holds a word', () => {
    expect(wer([''], [''])).toBeNaN();
  });

  it('is Infinity for words against references that hold none', () => {
    expect(wer(['one'], [''])).toBe(Infinity);
  });
});
