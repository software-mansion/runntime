import { describe, expect, it } from 'vitest';
import { channelAffine, toChw, toHwc, toHwc4 } from '../../../../src/core/graph/ops.ts';
import { hwc4Input, input, meta } from '../values.ts';

describe('toHwc4', () => {
  it('pads channels up to a multiple of 4 and stores f16 by default', () => {
    // 5 channels take 2 blocks of 4: 3·3 pixels × 8 lanes
    expect(toHwc4(input([5, 3, 3])).shape).toEqual({
      elems: 72,
      dtype: 'f16',
      dims: [5, 3, 3],
      layout: 'hwc4',
    });
  });

  it.each(['f32', 'f16'] as const)('stores %s when asked, whatever the input holds', (dtype) => {
    for (const from of ['f32', 'f16'] as const) {
      expect(toHwc4(input([5, 3, 3], from), dtype).shape).toEqual({
        elems: 72,
        dtype,
        dims: [5, 3, 3],
        layout: 'hwc4',
      });
    }
  });

  it.each([
    { name: 'an hwc4 input', x: hwc4Input(4, 2, 2), error: /already stored 'hwc4'/ },
    { name: 'a 2D input', x: input([4, 4]), error: /expected a 3D float CHW tensor/ },
  ])('rejects $name', ({ x, error }) => {
    expect(() => toHwc4(x)).toThrow(error);
  });
});

describe('toChw', () => {
  it('drops the padding lanes, in the dtype asked for', () => {
    expect(toChw(hwc4Input(5, 3, 3), 'f32').shape).toEqual(meta([5, 3, 3]));
  });

  it('rejects a row-major input', () => {
    expect(() => toChw(input([5, 3, 3]), 'f32')).toThrow(/must be hwc4-stored/);
  });
});

describe('toHwc', () => {
  it.each(['f32', 'f16'] as const)('gives row-major [H, W, C] in the image dtype, %s', (dtype) => {
    expect(toHwc(hwc4Input(8, 3, 5, dtype)).shape).toEqual(meta([3, 5, 8], dtype));
  });

  it('rejects a row-major input', () => {
    expect(() => toHwc(input([8, 3, 3]))).toThrow(/must be hwc4-stored/);
  });

  it('rejects channels that are not a multiple of 4 with an INVALID_ARGUMENT', () => {
    expect(() => toHwc(hwc4Input(5, 3, 3))).toThrow(
      expect.objectContaining({
        code: 'INVALID_ARGUMENT',
        message: 'toHwc: channels must be a multiple of 4, got 5; toChw() takes any count',
      }),
    );
  });
});

describe('channelAffine', () => {
  const x = hwc4Input(5, 3, 3);

  it('keeps the layout, with or without a shift', () => {
    expect(channelAffine(x, input([5])).shape).toEqual(x.shape);
    const shift = input([5]);
    expect(channelAffine(x, input([5]), shift).pending?.attrs).toEqual([1]);
  });

  it.each([
    {
      name: 'a scale per padded lane instead of per channel',
      scale: input([8]),
      error: /scale must be f32 with 5 elems, got 8/,
    },
    { name: 'an f16 scale', scale: input([5], 'f16'), error: /scale must be f32/ },
    {
      name: 'a shift of the wrong length',
      shift: input([4]),
      error: /shift must be f32 with 5 elems/,
    },
  ])('rejects $name', ({ scale = input([5]), shift, error }) => {
    expect(() => channelAffine(x, scale, shift)).toThrow(error);
  });

  it('rejects an f32 image with an INVALID_ARGUMENT', () => {
    expect(() => channelAffine(hwc4Input(5, 3, 3, 'f32'), input([5]))).toThrow(
      expect.objectContaining({
        code: 'INVALID_ARGUMENT',
        message:
          'channelAffine: no f32 kernel yet for an hwc4 input, run the model in f16 with half()',
      }),
    );
  });
});
