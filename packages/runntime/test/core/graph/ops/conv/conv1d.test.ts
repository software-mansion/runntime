import { describe, expect, it } from 'vitest';
import { ACT_CODE, conv1d, gluConv1d } from '../../../../../src/core/graph/ops.ts';
import { input, meta, quantWeight } from '../../values.ts';

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
    // attrs: kernel, stride, padLeft, padRight, hasBias, act, dilation, bits,
    // groupSize
    expect(c.pending?.attrs?.slice(4)).toEqual([1, ACT_CODE.gelu, 1, 0, 0]);
  });

  it('takes a quantW weight with its scales before the bias', () => {
    const w = quantWeight(3 * 8, 4, 8);
    const scales = input([2, 4]);
    const bias = input([4]);
    const c = conv1d(input([T, 8]), w, {
      kernelSize: 3,
      stride: 1,
      bias,
      scales,
      bits: 8,
      groupSize: 4,
    });
    expect(c.shape).toEqual(meta([8, 4]));
    expect(c.pending?.inputs).toEqual([expect.anything(), w, scales, bias]);
    expect(c.pending?.attrs?.slice(7)).toEqual([8, 4]);
  });

  it.each([
    {
      name: 'a quantW weight without scales',
      opts: { bits: 8 as const, groupSize: 4 },
      err: /quantW weights need scales, bits and groupSize/,
    },
    {
      name: 'C_in not a multiple of groupSize',
      opts: { scales: input([2, 4]), bits: 8 as const, groupSize: 12 },
      err: /C_in a multiple of groupSize/,
    },
    {
      name: 'kernelSize·C_in not a multiple of 8',
      x: input([T, 4]),
      w: quantWeight(3 * 4, 4, 8),
      opts: { scales: input([1, 4]), bits: 8 as const, groupSize: 4 },
      err: /kernelSize·C_in a multiple of 8/,
    },
    {
      name: 'scales of the wrong size',
      opts: { scales: input([1, 4]), bits: 8 as const, groupSize: 4 },
      err: /scales must be f32 \[C_in\/groupSize, C_out\] = \[2, 4\]/,
    },
  ])('refuses $name', ({ x = input([T, 8]), w = quantWeight(3 * 8, 4, 8), opts, err }) => {
    expect(() => conv1d(x, w, { kernelSize: 3, stride: 1, ...opts })).toThrow(err);
  });

  it('refuses scales on a float weight', () => {
    const w = weight(3);
    expect(() =>
      conv1d(x, w, { kernelSize: 3, stride: 1, scales: input([1, COUT]), bits: 8, groupSize: 4 }),
    ).toThrow(/scales\/bits\/groupSize apply to quantW weights only/);
  });

  it('widens the window by the dilation', () => {
    const c = conv1d(input([20, CIN]), weight(3), { kernelSize: 3, stride: 1, dilation: 4 });
    expect(c.shape).toEqual(meta([12, COUT]));
    const same = conv1d(input([20, CIN]), weight(3), {
      kernelSize: 3,
      stride: 1,
      dilation: 4,
      padding: 'same',
    });
    expect(same.shape).toEqual(meta([20, COUT]));
    expect(same.pending?.attrs?.slice(2, 4)).toEqual([4, 4]);
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
    {
      name: 'a zero kernel',
      opts: { kernelSize: 0 },
      error: /kernelSize 0 and stride 1 must be positive integers/,
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
      error: /padding 3 must be an integer in \[0, 3\), the window span/,
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
    {
      name: 'a dilation of 0',
      opts: { dilation: 0 },
      error: /dilation 0 must be a positive integer/,
    },
    {
      name: 'an input shorter than the kernel',
      x: input([2, CIN]),
      error: /padded frames 2\+0\+0 < window span 3/,
    },
  ])('rejects $name', ({ x: xIn = x, opts = {}, w = weight(3), error }) => {
    expect(() => conv1d(xIn, w, { kernelSize: 3, stride: 1, ...opts })).toThrow(error);
  });
});

describe('gluConv1d', () => {
  it('halves the columns and keeps the frames', () => {
    const c = gluConv1d(input([T, 16]), input([5, 8]), { lo: -2, hi: 2 });
    expect(c.shape).toEqual(meta([T, 8]));
    expect(c.pending?.attrs).toEqual([5, -2, 2]);
  });

  it.each([
    { name: 'x not twice the weight width', x: input([T, 12]), error: /x cols 12 must be 2·C/ },
    {
      name: 'channels not a multiple of 4',
      x: input([T, 12]),
      w: input([5, 6]),
      error: /C a multiple of 4/,
    },
    {
      name: 'a weight of another dtype',
      w: input([5, 8], 'f16'),
      error: /weight dtype f16 must match/,
    },
    { name: 'lo above hi', opts: { lo: 3, hi: 2 }, error: /lo \(3\) must be <= hi \(2\)/ },
  ])(
    'rejects $name',
    ({ x: xIn = input([T, 16]), w = input([5, 8]), opts = { lo: -2, hi: 2 }, error }) => {
      expect(() => gluConv1d(xIn, w, opts)).toThrow(error);
    },
  );
});
