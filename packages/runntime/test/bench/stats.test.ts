import { describe, expect, it } from 'vitest';
import {
  callsPerRound,
  compare,
  confirm,
  formatTable,
  formatTerminal,
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
    // Quartiles are the medians of the lower and upper halves, middle excluded.
    { xs: [3, 1, 2], median: 2, spread: 1 },
    { xs: [4, 1, 3, 2], median: 2.5, spread: 0.8 },
    { xs: [5, 5, 5], median: 5, spread: 0 },
    // One spike in eight leaves the middle half alone: q1 = 100, q3 = 101.
    { xs: [100, 100, 101, 100, 300, 101, 100, 101], median: 100.5, spread: 1 / 100.5 },
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
    // A's middle half spans 90..110, 20% of its median, so 8% is noise.
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
    expect(compare([90, 100, 110], [100, 100, 100]).noise).toBeCloseTo(0.2, 12);
  });
});

describe('confirm', () => {
  const slower = { aUs: 100, bUs: 110, delta: 0.1, noise: 0.03, verdict: 'slower' } as const;
  it.each([
    {
      name: 'two slower runs stay slower',
      second: { aUs: 100, bUs: 120, delta: 0.2, noise: 0.05, verdict: 'slower' },
      want: { aUs: 100, bUs: 115, delta: 0.15, noise: 0.05, verdict: 'slower' },
    },
    {
      name: 'a slower run then a same run is same',
      second: { aUs: 100, bUs: 101, delta: 0.01, noise: 0.03, verdict: 'same' },
      want: { aUs: 100, bUs: 105.5, delta: 0.055, noise: 0.03, verdict: 'same' },
    },
    {
      name: 'opposite directions are same',
      second: { aUs: 100, bUs: 90, delta: -0.1, noise: 0.03, verdict: 'faster' },
      want: { aUs: 100, bUs: 100, delta: 0, noise: 0.03, verdict: 'same' },
    },
  ] as const)('$name', ({ second, want }) => {
    const got = confirm(slower, second);
    expect(got.verdict).toBe(want.verdict);
    expect(got.aUs).toBeCloseTo(want.aUs, 12);
    expect(got.bUs).toBeCloseTo(want.bUs, 12);
    expect(got.delta).toBeCloseTo(want.delta, 12);
    expect(got.noise).toBe(want.noise);
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
      {
        kernel: 'conv2d',
        case: 'yolo 80×80',
        note: 'skipped, threw on A: conv2d is not a function',
      },
    ]);
    expect(table.split('\n').slice(2)).toEqual([
      '| matmul | tiled 64×64×64 | 412 | 298 | -27.6% | ±3.0% | faster | 3.1e-4 |',
      '| softmax | rows 8×1024 | 88.2 | 87.9 | -0.3% | ±3.1% | same | 0 |',
      '| conv2d | yolo 80×80 | | | | | skipped, threw on A: conv2d is not a function | |',
    ]);
  });
});

describe('formatTerminal', () => {
  const results = [
    {
      kernel: 'matmul',
      case: 'gemv',
      comparison: { aUs: 40, bUs: 40.4, delta: 0.01, noise: 0.03, verdict: 'same' },
      maxAbsDiff: 0,
    },
    {
      kernel: 'matmul',
      case: 'tiled 64×64×64',
      comparison: { aUs: 412, bUs: 298.3, delta: -0.276, noise: 0.03, verdict: 'faster' },
      maxAbsDiff: 3.1e-4,
    },
    {
      kernel: 'softmax',
      case: 'rows 8×1024',
      comparison: { aUs: 88.2, bUs: 87.9, delta: -0.0034, noise: 0.031, verdict: 'same' },
      maxAbsDiff: 0,
    },
    { kernel: 'conv2d', case: 'yolo 80×80', note: 'skipped, device has no shader-f16' },
  ] as const;

  it('boxes the measured cases, a rule between kernels, skipped ones listed below', () => {
    expect(formatTerminal(results).split('\n')).toEqual([
      '┌─────────┬────────────────┬────────┬────────┬────────┬───────┬─────────┬──────────────┐',
      '│ kernel  │ case           │ A (µs) │ B (µs) │      Δ │ noise │ verdict │ max abs(A−B) │',
      '├─────────┼────────────────┼────────┼────────┼────────┼───────┼─────────┼──────────────┤',
      '│ matmul  │ gemv           │   40.0 │   40.4 │  +1.0% │ ±3.0% │ same    │            0 │',
      '│ matmul  │ tiled 64×64×64 │    412 │    298 │ -27.6% │ ±3.0% │ faster  │       3.1e-4 │',
      '├─────────┼────────────────┼────────┼────────┼────────┼───────┼─────────┼──────────────┤',
      '│ softmax │ rows 8×1024    │   88.2 │   87.9 │  -0.3% │ ±3.1% │ same    │            0 │',
      '└─────────┴────────────────┴────────┴────────┴────────┴───────┴─────────┴──────────────┘',
      '',
      'conv2d · yolo 80×80: skipped, device has no shader-f16',
    ]);
  });

  it('paints only the padded verdict cell', () => {
    const table = formatTerminal(results.slice(1, 2), (v, text) => `<${v}>${text}</>`);
    expect(table).toContain('│ <faster>faster </> │');
    expect(table).not.toContain('<same>');
  });
});
