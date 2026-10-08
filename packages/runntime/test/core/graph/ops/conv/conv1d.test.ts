import { describe, expect, it } from 'vitest';
import { ACT_CODE, conv1d } from '../../../../../src/core/graph/ops.ts';
import { input, meta } from '../../values.ts';

// x is time-major [T, C_in]; w is tap-major [kernelSize·C_in, C_out]
const T = 10;
const CIN = 4;
const COUT = 6;
const x = input([T, CIN]);
const weight = (k: number) => input([k * CIN, COUT]);

describe('conv1d', () => {
  // out = floor((T + padLeft + padRight − k) / stride) + 1
  it.each([
    { name: "'same' keeps the length", k: 3, opts: { padding: 'same' as const }, out: T },
    { name: "'same' with an even kernel", k: 4, opts: { padding: 'same' as const }, out: T },
    { name: "'valid' shrinks by k − 1", k: 3, opts: { padding: 'valid' as const }, out: 8 },
    { name: 'stride 2 with padding 1', k: 3, opts: { padding: 1, stride: 2 }, out: 5 },
    { name: 'causal (k − 1 on the left)', k: 3, opts: { padLeft: 2 }, out: T },
    { name: 'a kernel of 1', k: 1, opts: {}, out: T },
  ])('$name', ({ k, opts, out }) => {
    const c = conv1d(x, weight(k), { kernelSize: k, stride: 1, ...opts });
    expect(c.shape).toEqual(meta([out, COUT]));
  });

  it('splits an even kernel’s same-padding with the extra pad on the right', () => {
    const c = conv1d(x, weight(4), { kernelSize: 4, stride: 1, padding: 'same' });
    // attrs: kernelSize, stride, padLeft, padRight, hasBias, activation
    expect(c.pending?.attrs?.slice(2, 4)).toEqual([1, 2]);
  });

  it('takes a bias and a fused activation', () => {
    const w = weight(3);
    const bias = input([COUT]);
    const c = conv1d(x, w, { kernelSize: 3, stride: 1, bias, activation: 'gelu' });
    expect(c.pending?.inputs).toEqual([x, w, bias]);
    expect(c.pending?.attrs?.slice(4)).toEqual([1, ACT_CODE.gelu]);
  });

  it('gives one frame when the padded input is exactly one kernel long', () => {
    expect(conv1d(input([3, CIN]), weight(3), { kernelSize: 3, stride: 1 }).shape).toEqual(
      meta([1, COUT]),
    );
  });

  it.each([
    {
      name: 'padding and padLeft together',
      opts: { padding: 1, padLeft: 1 },
      error: /padding excludes padLeft\/padRight/,
    },
    { name: 'a zero stride', opts: { stride: 0 }, error: /stride 0 must be positive integers/ },
    {
      name: "'same' with stride 2",
      opts: { padding: 'same' as const, stride: 2 },
      error: /'same' requires stride 1/,
    },
    {
      name: 'padding as large as the kernel',
      opts: { padding: 3 },
      error: /padding 3 must be an integer in \[0, kernelSize\)/,
    },
    { name: 'a negative pad', opts: { padRight: -1 }, error: /padRight -1 must be an integer/ },
    {
      name: 'a weight of another dtype',
      w: input([12, COUT], 'f16'),
      error: /weight dtype f16 must match the input's f32/,
    },
    {
      name: 'weight rows that are not kernelSize·C_in',
      w: input([10, COUT]),
      error: /w rows 10 != kernelSize·C_in = 3·4/,
    },
    {
      name: 'a bias of the wrong length',
      opts: { bias: input([4]) },
      error: /bias must be f32 \[C_out\] = 6 elems/,
    },
  ])('rejects $name', ({ opts = {}, w = weight(3), error }) => {
    expect(() => conv1d(x, w, { kernelSize: 3, stride: 1, ...opts })).toThrow(error);
  });

  it('rejects an input shorter than the kernel', () => {
    expect(() => conv1d(input([2, CIN]), weight(3), { kernelSize: 3, stride: 1 })).toThrow(
      /padded frames 2\+0\+0 < kernelSize 3/,
    );
  });

  it('rejects a zero kernel', () => {
    expect(() => conv1d(x, input([0, COUT]), { kernelSize: 0, stride: 1 })).toThrow(
      /kernelSize 0 and stride 1 must be positive integers/,
    );
  });
});
