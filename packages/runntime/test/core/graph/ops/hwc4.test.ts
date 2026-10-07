import { describe, expect, it } from 'vitest';
import { channelAffine, toChw, toHwc4 } from '../../../../src/core/graph/ops.ts';
import { hwc4Input, input, meta } from '../values.ts';

describe('toHwc4', () => {
  it('pads channels up to a multiple of 4 and stores f16', () => {
    // 5 channels take 2 blocks of 4: 3·3 pixels × 8 lanes
    expect(toHwc4(input([5, 3, 3])).shape).toEqual({
      elems: 72,
      dtype: 'f16',
      dims: [5, 3, 3],
      layout: 'hwc4',
    });
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
});
