import { describe, expect, it } from 'vitest';
import { ACT_CODE, conv2d } from '../../../../../src/core/graph/ops.ts';
import { pending, type Value } from '../../../../../src/core/graph/value.ts';
import { hwc4Input, input } from '../../values.ts';

/** A conv weight as nn.Conv2d packs it at load: f16 [cOut, cIn/groups·kH·kW]
 *  tagged hwc4. */
function packedWeight(cOut: number, cols: number): Value {
  return pending(
    { elems: cOut * cols, dtype: 'f16', dims: [cOut, cols], layout: 'hwc4' },
    'reshape',
    [],
  );
}

const plain = { stride: 1, padding: 0, groups: 1 };

describe('conv2d', () => {
  // out = floor((in + 2·padding − k) / stride) + 1, per side
  it.each([
    {
      name: '3×3, stride 1, padding 1 keeps the size',
      h: 16,
      w: 16,
      k: 3,
      stride: 1,
      padding: 1,
      out: [16, 16],
    },
    {
      name: '3×3, stride 2, padding 1 halves it',
      h: 32,
      w: 32,
      k: 3,
      stride: 2,
      padding: 1,
      out: [16, 16],
    },
    {
      name: '3×3, stride 2, padding 1 on an odd size rounds up',
      h: 33,
      w: 33,
      k: 3,
      stride: 2,
      padding: 1,
      out: [17, 17],
    },
    {
      name: '3×3, no padding shrinks by 2',
      h: 10,
      w: 12,
      k: 3,
      stride: 1,
      padding: 0,
      out: [8, 10],
    },
    { name: '1×1 keeps the size', h: 7, w: 5, k: 1, stride: 1, padding: 0, out: [7, 5] },
    {
      name: 'a kernel as large as the input gives 1×1',
      h: 4,
      w: 4,
      k: 4,
      stride: 1,
      padding: 0,
      out: [1, 1],
    },
  ])('$name', ({ h, w, k, stride, padding, out }) => {
    const c = conv2d(hwc4Input(8, h, w), packedWeight(16, 8 * k * k), undefined, {
      kernelSize: k,
      stride,
      padding,
      groups: 1,
    });
    expect(c.shape).toEqual(hwc4Input(16, out[0]!, out[1]!).shape);
  });

  it('takes a non-square kernel', () => {
    const c = conv2d(hwc4Input(8, 10, 10), packedWeight(4, 8 * 1 * 3), undefined, {
      ...plain,
      kernelSize: [1, 3],
    });
    expect(c.shape.dims).toEqual([4, 10, 8]);
  });

  it('runs depthwise when groups equals the channel count', () => {
    const c = conv2d(hwc4Input(8, 6, 6), packedWeight(8, 9), undefined, {
      ...plain,
      kernelSize: 3,
      padding: 1,
      groups: 8,
    });
    expect(c.shape).toEqual(hwc4Input(8, 6, 6).shape);
  });

  it('takes a bias and a fused activation', () => {
    const x = hwc4Input(8, 6, 6);
    const weight = packedWeight(16, 8);
    const bias = input([16], 'f16');
    const c = conv2d(x, weight, bias, { ...plain, kernelSize: 1, activation: 'silu' });
    expect(c.pending?.inputs).toEqual([x, weight, bias]);
    expect(c.pending?.attrs?.at(-1)).toBe(ACT_CODE.silu);
  });

  it.each([
    { name: 'a row-major input', x: input([8, 6, 6], 'f16'), error: /input must be hwc4-stored/ },
    { name: 'a zero kernel', kernelSize: 0, error: /kernelSize must be positive integer/ },
    { name: 'a fractional kernel', kernelSize: 1.5, error: /kernelSize must be positive integer/ },
    {
      name: 'groups that are neither 1 nor depthwise',
      groups: 2,
      error: /groups=2 is neither 1 nor depthwise/,
    },
    {
      name: 'an unpacked weight',
      weight: input([16, 8], 'f16'),
      error: /weight must be the hwc4 mat4-tile packing/,
    },
    {
      name: 'weight cols that do not match the kernel',
      weight: packedWeight(16, 9),
      error: /weight cols 9 != C_in\/groups·kH·kW = 8·1·1/,
    },
    { name: 'an f32 bias', bias: input([16]), error: /bias must be f16 with 16 elems/ },
    {
      name: 'a bias of the wrong length',
      bias: input([8], 'f16'),
      error: /bias must be f16 with 16 elems/,
    },
  ])('rejects $name', ({ x, weight, bias, kernelSize = 1, groups = 1, error }) => {
    expect(() =>
      conv2d(x ?? hwc4Input(8, 6, 6), weight ?? packedWeight(16, 8), bias, {
        ...plain,
        kernelSize,
        groups,
      }),
    ).toThrow(error);
  });

  it('rejects depthwise groups when out channels differ from in channels', () => {
    expect(() =>
      conv2d(hwc4Input(8, 6, 6), packedWeight(16, 9), undefined, {
        ...plain,
        kernelSize: 3,
        groups: 8,
      }),
    ).toThrow(/neither 1 nor depthwise \(cIn=8, cOut=16\)/);
  });
});
