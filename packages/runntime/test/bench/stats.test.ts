import { describe, expect, it } from 'vitest';
import {
  callsPerRound,
  compare,
  formatTable,
  maxAbsDiff,
  median,
  spread,
} from '../../bench/stats.ts';

describe('callsPerRound', () => {
  it.each([
    { name: 'a 100 µs call fills 10 ms', oneCallUs: 100, outBytes: 4, want: 100 },
    { name: 'a 1 µs call is capped at 256', oneCallUs: 1, outBytes: 4, want: 256 },
    { name: 'a 0 µs reading is capped at 256', oneCallUs: 0, outBytes: 4, want: 256 },
    { name: 'a 50 ms call still runs twice', oneCallUs: 50_000, outBytes: 4, want: 2 },
    { name: '64 MB outputs fit 4 in 256 MB', oneCallUs: 10, outBytes: 64 * 2 ** 20, want: 4 },
    { name: 'a 200 MB output still runs twice', oneCallUs: 10, outBytes: 200 * 2 ** 20, want: 2 },
  ])('$name', ({ oneCallUs, outBytes, want }) => {
    expect(callsPerRound(oneCallUs, outBytes)).toBe(want);
  });
});

describe('median and spread', () => {
  it.each([
    { xs: [3, 1, 2], median: 2, spread: 0.5 },
    { xs: [4, 1, 3, 2], median: 2.5, spread: 0.6 },
    { xs: [5, 5, 5], median: 5, spread: 0 },
  ])('$xs', ({ xs, median: m, spread: s }) => {
    expect(median(xs)).toBe(m);
    expect(spread(xs)).toBeCloseTo(s, 12);
  });
});

describe('compare', () => {
  it.each([
    { name: 'identical rounds', a: [100, 100, 100], b: [100, 100, 100], verdict: 'same', delta: 0 },
    {
      name: '2% slower, under the 3% floor',
      a: [100, 100, 100],
      b: [102, 102, 102],
      verdict: 'same',
      delta: 0.02,
    },
    {
      name: '10% slower, quiet rounds',
      a: [100, 100, 100],
      b: [110, 110, 110],
      verdict: 'slower',
      delta: 0.1,
    },
    {
      name: '25% faster, quiet rounds',
      a: [100, 100, 100],
      b: [75, 75, 75],
      verdict: 'faster',
      delta: -0.25,
    },
    // A's rounds spread ±10% (90..110), so a 8% change is inside the noise.
    {
      name: '8% faster, noisy A',
      a: [90, 100, 110],
      b: [92, 92, 92],
      verdict: 'same',
      delta: -0.08,
    },
  ] as const)('$name', ({ a, b, verdict, delta }) => {
    const c = compare(a, b);
    expect(c.verdict).toBe(verdict);
    expect(c.delta).toBeCloseTo(delta, 12);
  });

  it('reports the larger of the floor and either side’s spread as noise', () => {
    expect(compare([100, 100, 100], [100, 100, 100]).noise).toBe(0.03);
    expect(compare([90, 100, 110], [100, 100, 100]).noise).toBeCloseTo(0.1, 12);
  });
});

describe('maxAbsDiff', () => {
  it.each([
    { name: 'equal', a: [1, 2, 3], b: [1, 2, 3], want: 0 },
    { name: 'one element off', a: [1, 2, 3], b: [1, 2.5, 3], want: 0.5 },
    { name: 'NaN in A', a: [1, NaN], b: [1, 2], want: Infinity },
    { name: 'NaN in B after a real difference', a: [0, 2], b: [1, NaN], want: Infinity },
    { name: 'different lengths', a: [1, 2], b: [1], want: Infinity },
  ])('$name', ({ a, b, want }) => {
    expect(maxAbsDiff(a, b)).toBe(want);
  });
});

describe('formatTable', () => {
  it('writes one row per case, skipped ones included', () => {
    const table = formatTable([
      {
        kernel: 'matmul',
        case: 'tiled 64×64×64',
        comparison: { aUs: 412, bUs: 298.3, delta: -0.276, noise: 0.03, verdict: 'faster' },
        maxAbsDiff: 3.1e-4,
        calls: 24,
      },
      {
        kernel: 'softmax',
        case: 'rows 8×1024',
        comparison: { aUs: 88.2, bUs: 87.9, delta: -0.0034, noise: 0.031, verdict: 'same' },
        maxAbsDiff: 0,
        calls: 113,
      },
      { kernel: 'conv2d', case: 'yolo 80×80', skipped: 'threw on A: conv2d is not a function' },
    ]);
    expect(table.split('\n').slice(2)).toEqual([
      '| matmul | tiled 64×64×64 | 412 | 298 | -27.6% | ±3.0% | faster | 3.1e-4 |',
      '| softmax | rows 8×1024 | 88.2 | 87.9 | -0.3% | ±3.1% | same | 0 |',
      '| conv2d | yolo 80×80 | | | | | skipped: threw on A: conv2d is not a function | |',
    ]);
  });
});
