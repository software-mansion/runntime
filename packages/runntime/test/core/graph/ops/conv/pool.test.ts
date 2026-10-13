import { describe, expect, it } from 'vitest';
import {
  avgPool2d,
  maxPool2d,
  pad2d,
  resizeBilinear2d,
  upsample2d,
} from '../../../../../src/core/graph/ops.ts';
import { hwc4Input, input } from '../../values.ts';

const map = (h: number, w: number) => hwc4Input(5, h, w);

describe('maxPool2d', () => {
  it.each([
    { name: '3×3, stride 2, padding 1 halves', h: 16, k: 3, stride: 2, padding: 1, out: 8 },
    { name: '2×2, stride 2 on an odd size rounds down', h: 5, k: 2, stride: 2, padding: 0, out: 2 },
    {
      name: '5×5, stride 1, padding 2 keeps the size (SPPF)',
      h: 20,
      k: 5,
      stride: 1,
      padding: 2,
      out: 20,
    },
    { name: 'a window as large as the input', h: 3, k: 3, stride: 1, padding: 0, out: 1 },
  ])('$name', ({ h, k, stride, padding, out }) => {
    expect(maxPool2d(map(h, h), { kernelSize: k, stride, padding }).shape).toEqual(
      map(out, out).shape,
    );
  });

  it.each([
    {
      name: 'padding over half the kernel',
      opts: { kernelSize: 3, stride: 1, padding: 2 },
      error: /bad geometry/,
    },
    {
      name: 'a zero stride',
      opts: { kernelSize: 2, stride: 0, padding: 0 },
      error: /bad geometry/,
    },
    {
      name: 'a negative padding',
      opts: { kernelSize: 2, stride: 1, padding: -1 },
      error: /bad geometry/,
    },
    {
      name: 'a window larger than the input',
      opts: { kernelSize: 5, stride: 1, padding: 0 },
      error: /window k=5 larger than padded input 3×3/,
    },
  ])('rejects $name', ({ opts, error }) => {
    expect(() => maxPool2d(map(3, 3), opts)).toThrow(error);
  });

  it('rejects a row-major input', () => {
    expect(() =>
      maxPool2d(input([5, 4, 4], 'f16'), { kernelSize: 2, stride: 2, padding: 0 }),
    ).toThrow(/must be hwc4-stored/);
  });
});

describe('avgPool2d', () => {
  it('strides by the kernel size by default', () => {
    expect(avgPool2d(map(8, 6), { kernelSize: 2 }).shape).toEqual(map(4, 3).shape);
  });

  it('takes its own stride', () => {
    expect(avgPool2d(map(8, 8), { kernelSize: 3, stride: 1 }).shape).toEqual(map(6, 6).shape);
  });

  it.each([
    {
      name: 'a window larger than the input',
      opts: { kernelSize: 4 },
      error: /window k=4 larger than input 3×3/,
    },
    { name: 'a zero kernel', opts: { kernelSize: 0 }, error: /bad geometry/ },
  ])('rejects $name', ({ opts, error }) => {
    expect(() => avgPool2d(map(3, 3), opts)).toThrow(error);
  });
});

describe('upsample2d', () => {
  it('multiplies each side by the scale', () => {
    expect(upsample2d(map(3, 4), { scale: 2 }).shape).toEqual(map(6, 8).shape);
  });

  it.each([0, 1.5])('rejects scale %s', (scale) => {
    expect(() => upsample2d(map(3, 4), { scale })).toThrow(/scale must be a positive integer/);
  });
});

describe('pad2d', () => {
  it('pads each side separately per axis', () => {
    expect(pad2d(map(3, 4), { padH: 1, padW: 2 }).shape).toEqual(map(5, 8).shape);
  });

  it('pads by nothing', () => {
    expect(pad2d(map(3, 4), { padH: 0, padW: 0 }).shape).toEqual(map(3, 4).shape);
  });

  it.each([
    { padH: -1, padW: 0 },
    { padH: 0, padW: 0.5 },
  ])('rejects ($padH, $padW)', (opts) => {
    expect(() => pad2d(map(3, 4), opts)).toThrow(/pads must be non-negative integers/);
  });
});

describe('resizeBilinear2d', () => {
  it('resizes to any size, up or down', () => {
    expect(resizeBilinear2d(map(3, 4), { outH: 7, outW: 2 }).shape).toEqual(map(7, 2).shape);
  });

  it('uses align-corners unless told otherwise', () => {
    // attrs: outH, outW, alignCorners
    expect(resizeBilinear2d(map(3, 4), { outH: 6, outW: 8 }).pending?.attrs).toEqual([6, 8, 1]);
    expect(
      resizeBilinear2d(map(3, 4), { outH: 6, outW: 8, mode: 'halfPixel' }).pending?.attrs,
    ).toEqual([6, 8, 0]);
  });

  it.each([
    { outH: 0, outW: 4 },
    { outH: 4, outW: 2.5 },
  ])('rejects $outH×$outW', (opts) => {
    expect(() => resizeBilinear2d(map(3, 4), opts)).toThrow(/bad output size/);
  });
});
