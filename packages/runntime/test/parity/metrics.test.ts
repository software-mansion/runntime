/** The parity metrics on numbers worked out by hand. */
import { describe, expect, it } from 'vitest';
import { cosine, maxAbsError, missing, rankMismatches, summarize, topK } from './metrics.ts';

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
