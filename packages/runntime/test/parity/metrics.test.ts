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
  it.each([
    { name: 'the same ranking', got: [4, 2, 7], mismatches: 0, missed: 0 },
    { name: 'two items swapped', got: [2, 4, 7], mismatches: 2, missed: 0 },
    { name: 'one item replaced', got: [4, 2, 9], mismatches: 1, missed: 1 },
  ])('$name: $mismatches positions differ, $missed missing', ({ got, mismatches, missed }) => {
    const want = [4, 2, 7];
    expect(rankMismatches(got, want)).toBe(mismatches);
    expect(missing(got, want)).toBe(missed);
  });
});
