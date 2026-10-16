import { describe, expect, it } from 'vitest';
import { layerNorm, mean, rmsNorm, softmax, topk } from '../../../../src/core/graph/ops.ts';
import { input, meta } from '../values.ts';

describe('reductions over the last dim', () => {
  it('mean gives one value per row, in the input dtype', () => {
    expect(mean(input([4, 8], 'f16')).shape).toEqual(meta([4, 1], 'f16'));
  });

  it('softmax keeps the shape', () => {
    expect(softmax(input([4, 8], 'f16')).shape).toEqual(meta([4, 8], 'f16'));
  });

  it.each([
    ['mean', mean],
    ['softmax', softmax],
  ] as const)('%s rejects anything but 2D', (_, op) => {
    expect(() => op(input([8]))).toThrow(/expected a 2D/);
  });
});

describe('masked softmax', () => {
  // 5×5 attention scores, a [start, end) key range per row
  const x = input([5, 5]);
  const segments = input([5, 2]);

  it('takes segments as a second input', () => {
    const c = softmax(x, { segments, windowLeft: 2, windowRight: 0 });
    expect(c.shape).toEqual(meta([5, 5]));
    expect(c.pending?.inputs).toEqual([x, segments]);
    // attrs: masked, windowLeft, windowRight
    expect(c.pending?.attrs).toEqual([1, 2, 0]);
  });

  it('runs unmasked without segments', () => {
    expect(softmax(x).pending?.attrs).toEqual([0, 0, 0]);
  });

  it('carries unbounded windows as a large finite number', () => {
    const [, left, right] = softmax(x, { segments }).pending!.attrs!;
    expect(Number.isFinite(left)).toBe(true);
    expect(left).toBeGreaterThan(1e9);
    expect(right).toBe(left);
  });

  it.each([
    {
      name: 'scores that are not square',
      x: input([4, 8]),
      segments: input([4, 2]),
      error: /needs square scores, got \[4, 8\]/,
    },
    {
      name: 'segments of the wrong size',
      segments: input([5]),
      error: /segments must be f32 \[5, 2\]/,
    },
    { name: 'f16 segments', segments: input([5, 2], 'f16'), error: /segments must be f32/ },
    {
      name: 'a negative window',
      opts: { windowLeft: -1 },
      error: /windowLeft must be a non-negative integer or Infinity, got -1/,
    },
    {
      name: 'a fractional window',
      opts: { windowRight: 0.5 },
      error: /windowRight must be a non-negative integer or Infinity, got 0.5/,
    },
  ])('rejects $name', ({ x: xIn = x, segments: s = segments, opts, error }) => {
    expect(() => softmax(xIn, { segments: s, ...opts })).toThrow(error);
  });
});

describe('layerNorm', () => {
  const x = input([4, 8]);

  it('keeps the shape and carries eps', () => {
    const weight = input([8]);
    const bias = input([8]);
    const c = layerNorm(x, weight, 1e-6, bias);
    expect(c.shape).toEqual(meta([4, 8]));
    expect(c.pending).toMatchObject({ scalar: 1e-6, inputs: [x, weight, bias] });
  });

  it('works without a bias, with eps 1e-5 by default', () => {
    expect(layerNorm(x, input([8])).pending).toMatchObject({ scalar: 1e-5 });
  });

  it.each([
    {
      name: 'a weight of the wrong length',
      weight: input([4]),
      error: /weight must be f32 \[N\] = 8 elems/,
    },
    { name: 'an f16 weight on f32', weight: input([8], 'f16'), error: /weight must be f32/ },
    {
      name: 'a bias of the wrong length',
      bias: input([4]),
      error: /bias must be f32 \[N\] = 8 elems/,
    },
    { name: 'eps of 0', eps: 0, error: /eps must be positive/ },
    { name: 'a negative eps', eps: -1e-5, error: /eps must be positive/ },
    { name: 'eps NaN', eps: NaN, error: /eps must be positive/ },
  ])('rejects $name', ({ weight = input([8]), bias, eps = 1e-5, error }) => {
    expect(() => layerNorm(x, weight, eps, bias)).toThrow(error);
  });
});

describe('rmsNorm', () => {
  const x = input([4, 8]);

  it('keeps the shape and carries eps', () => {
    const weight = input([8]);
    const c = rmsNorm(x, weight, 1e-5);
    expect(c.shape).toEqual(meta([4, 8]));
    // attrs: group, scale
    expect(c.pending).toMatchObject({
      op: 'rmsNorm',
      scalar: 1e-5,
      inputs: [x, weight],
      attrs: [8, 1],
    });
  });

  it('keeps f16, with eps 1e-6 by default', () => {
    const c = rmsNorm(input([4, 8], 'f16'), input([8], 'f16'));
    expect(c.shape).toEqual(meta([4, 8], 'f16'));
    expect(c.pending).toMatchObject({ scalar: 1e-6 });
  });

  it.each([
    { name: 'one weight shared by every run', weight: [4] },
    { name: 'a weight slice per run', weight: [8] },
  ])('normalizes runs of group columns, with $name', ({ weight }) => {
    const c = rmsNorm(x, input(weight), 1e-6, { group: 4 });
    expect(c.shape).toEqual(meta([4, 8]));
    expect(c.pending?.attrs).toEqual([4, 1]);
  });

  it('adds a residual and scales the sum', () => {
    const weight = input([8]);
    const residual = input([4, 8]);
    const c = rmsNorm(x, weight, 1e-6, { residual, scale: 0.5 });
    expect(c.pending).toMatchObject({ inputs: [x, weight, residual], attrs: [8, 0.5] });
  });

  it.each([
    { name: 'a 1D input', x: input([8]), error: /expected a 2D/ },
    {
      name: 'a weight of the wrong length',
      weight: input([4]),
      error: /weight must be f32 with 8 elems/,
    },
    {
      name: 'a grouped weight of neither length',
      weight: input([2]),
      opts: { group: 4 },
      error: /weight must be f32 with 4 or 8 elems/,
    },
    { name: 'an f16 weight on f32', weight: input([8], 'f16'), error: /weight must be f32/ },
    { name: 'eps of 0', eps: 0, error: /eps must be positive/ },
    { name: 'eps NaN', eps: NaN, error: /eps must be positive/ },
    {
      name: 'a group that does not divide the row',
      opts: { group: 3 },
      error: /group 3 must divide the row width 8/,
    },
    { name: 'a group of 0', opts: { group: 0 }, error: /group 0 must divide/ },
    {
      name: 'a residual of the wrong shape',
      opts: { residual: input([4, 4]) },
      error: /residual must be f32 \[4, 8\]/,
    },
    {
      name: 'an f16 residual on f32',
      opts: { residual: input([4, 8], 'f16') },
      error: /residual must be f32/,
    },
  ])('rejects $name', ({ x: xIn = x, weight = input([8]), eps = 1e-6, opts, error }) => {
    expect(() => rmsNorm(xIn, weight, eps, opts)).toThrow(error);
  });
});

describe('topk', () => {
  it('packs 4 ids then 4 weights per row, as f32', () => {
    expect(topk(input([3, 32], 'f16'), 4).shape).toEqual(meta([3, 8]));
  });

  it('takes exactly 4 columns', () => {
    expect(topk(input([3, 4]), 4).shape).toEqual(meta([3, 8]));
  });

  it.each([
    { name: 'k other than 4', x: input([3, 32]), k: 2, error: /only k=4 is supported, got 2/ },
    {
      name: 'fewer columns than k',
      x: input([3, 3]),
      k: 4,
      error: /need at least 4 columns, got 3/,
    },
  ])('rejects $name', ({ x, k, error }) => {
    expect(() => topk(x, k)).toThrow(error);
  });
});
