import { describe, expect, it } from 'vitest';
import { convTranspose2d } from '../../../../../src/core/graph/ops.ts';
import { input, meta } from '../../values.ts';

// x [C_in=4, 5, 6]; weight f16 [C_out·k·k, C_in] with C_out = 3, k = 2
const x = input([4, 5, 6]);
const weight = input([3 * 2 * 2, 4], 'f16');
const k2 = { kernelSize: 2, stride: 2 };

describe('convTranspose2d', () => {
  it('scales each side by k and reads C_out from the weight', () => {
    expect(convTranspose2d(x, weight, undefined, k2).shape).toEqual(meta([3, 10, 12]));
  });

  it('keeps f16 input as f16', () => {
    const c = convTranspose2d(input([4, 5, 6], 'f16'), weight, input([3], 'f16'), k2);
    expect(c.shape).toEqual(meta([3, 10, 12], 'f16'));
  });

  it('with k = 1 is a 1×1 conv', () => {
    const c = convTranspose2d(x, input([3, 4], 'f16'), undefined, { kernelSize: 1, stride: 1 });
    expect(c.shape).toEqual(meta([3, 5, 6]));
  });

  it.each([
    {
      name: 'a stride other than k (overlapping output)',
      opts: { kernelSize: 3, stride: 2 },
      error: /only kernelSize === stride/,
    },
    {
      name: 'a zero kernel',
      opts: { kernelSize: 0, stride: 0 },
      error: /must be positive integers/,
    },
    { name: 'an f32 weight', w: input([12, 4]), error: /weight must be 2D f16/ },
    {
      name: 'a weight for other input channels',
      w: input([12, 5], 'f16'),
      error: /weight C_in 5 != input C_in 4/,
    },
    {
      name: 'weight rows not a multiple of k²',
      w: input([10, 4], 'f16'),
      error: /weight rows 10 not a multiple of k²=4/,
    },
    {
      name: 'a bias of the wrong length',
      bias: input([4]),
      error: /bias must be f32 with 3 elems/,
    },
    { name: 'a bias of another dtype', bias: input([3], 'f16'), error: /bias must be f32/ },
  ])('rejects $name', ({ opts = k2, w = weight, bias, error }) => {
    expect(() => convTranspose2d(x, w, bias, opts)).toThrow(error);
  });
});
