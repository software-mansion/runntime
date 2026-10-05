import { describe, expect, it } from 'vitest';
import {
  bf16ToF16Bits,
  bf16ToF32,
  convToTapMajor,
  f16BitsToF32,
  f16ToF32,
  f32ArrayToF16Bits,
  f32ToF16Bits,
  packConvHwc4F16,
  packDwHwc4F16,
  permuteConvTransposeF16,
  transposeF32,
} from '../../../src/core/weights/convert.ts';

describe('f16BitsToF32', () => {
  it.each([
    { bits: 0x0000, value: 0 },
    { bits: 0x3c00, value: 1 },
    { bits: 0xc000, value: -2 },
    { bits: 0x3555, value: 0.333251953125 },
    { bits: 0x7bff, value: 65504 }, // largest finite f16
    { bits: 0x0400, value: 2 ** -14 }, // smallest normal
    { bits: 0x0001, value: 2 ** -24 }, // smallest subnormal
    { bits: 0x7c00, value: Infinity },
    { bits: 0xfc00, value: -Infinity },
  ])('$bits -> $value', ({ bits, value }) => {
    expect(f16BitsToF32(bits)).toBe(value);
  });

  it('keeps the sign of zero', () => {
    expect(Object.is(f16BitsToF32(0x8000), -0)).toBe(true);
  });

  it('decodes NaN', () => {
    expect(f16BitsToF32(0x7e00)).toBeNaN();
  });
});

describe('f32ToF16Bits', () => {
  it.each([
    { value: 0, bits: 0x0000 },
    { value: 1, bits: 0x3c00 },
    { value: -2, bits: 0xc000 },
    { value: 65504, bits: 0x7bff },
    { value: 2 ** -24, bits: 0x0001 },
    { value: 2 ** -26, bits: 0x0000 }, // below half the smallest subnormal
    { value: Infinity, bits: 0x7c00 },
    { value: -Infinity, bits: 0xfc00 },
    { value: NaN, bits: 0x7e00 },
  ])('$value -> $bits', ({ value, bits }) => {
    expect(f32ToF16Bits(value)).toBe(bits);
  });

  it('clamps finite overflow to the f16 max instead of Infinity', () => {
    expect(f32ToF16Bits(1e6)).toBe(0x7bff);
    expect(f32ToF16Bits(-1e6)).toBe(0xfbff);
  });

  it('rounds halfway cases to even', () => {
    expect(f32ToF16Bits(1 + 2 ** -11)).toBe(0x3c00); // between 0x3c00 and 0x3c01
    expect(f32ToF16Bits(1 + 3 * 2 ** -11)).toBe(0x3c02); // between 0x3c01 and 0x3c02
  });

  it('round-trips every finite f16', () => {
    const mismatches: number[] = [];
    for (let bits = 0; bits <= 0xffff; bits++) {
      if (((bits >>> 10) & 0x1f) === 31) continue; // Inf / NaN
      if (f32ToF16Bits(f16BitsToF32(bits)) !== bits) mismatches.push(bits);
    }
    expect(mismatches).toEqual([]);
  });
});

describe('array conversions', () => {
  it('f16ToF32', () => {
    expect(f16ToF32(new Uint16Array([0x3c00, 0xc000]))).toEqual(new Float32Array([1, -2]));
  });

  it('bf16ToF32', () => {
    expect(bf16ToF32(new Uint16Array([0x3f80, 0xc000, 0x0000]))).toEqual(
      new Float32Array([1, -2, 0]),
    );
  });

  it('f32ArrayToF16Bits counts clamped values', () => {
    const { data, clamped } = f32ArrayToF16Bits(new Float32Array([1, 70000, -70000, 65504]));
    expect(data).toEqual(new Uint16Array([0x3c00, 0x7bff, 0xfbff, 0x7bff]));
    expect(clamped).toBe(2);
  });

  it('bf16ToF16Bits counts clamped values', () => {
    // 0x4780 is 65536 in bf16, just past the f16 max
    const { data, clamped } = bf16ToF16Bits(new Uint16Array([0x3f80, 0x4780]));
    expect(data).toEqual(new Uint16Array([0x3c00, 0x7bff]));
    expect(clamped).toBe(1);
  });
});

describe('weight layouts', () => {
  it('transposeF32', () => {
    const src = new Float32Array([1, 2, 3, 4, 5, 6]); // [2, 3]
    expect(transposeF32(src, 2, 3)).toEqual(new Float32Array([1, 4, 2, 5, 3, 6]));
  });

  it('transposeF32 rejects a wrong length', () => {
    expect(() => transposeF32(new Float32Array(5), 2, 3)).toThrow(/length 5/);
  });

  it('convToTapMajor', () => {
    // [cOut=2, cIn=1, k=2] -> [k, cIn, cOut]
    const src = new Float32Array([1, 2, 3, 4]);
    expect(convToTapMajor(src, 2, 1, 2)).toEqual(new Float32Array([1, 3, 2, 4]));
  });

  it('convToTapMajor rejects a wrong length', () => {
    expect(() => convToTapMajor(new Float32Array(3), 2, 1, 2)).toThrow(/length 3/);
  });

  it('permuteConvTransposeF16', () => {
    // [cIn=2, cOut=2, 1x1] -> [cOut, k*k, cIn]
    const src = new Uint16Array([1, 2, 3, 4]);
    expect(permuteConvTransposeF16(src, 2, 2, 1)).toEqual(new Uint16Array([1, 3, 2, 4]));
  });

  it('packConvHwc4F16 puts each weight in its 4x4 tile and zero-pads', () => {
    // [cOut=2, cIn=2, 1x1], one tile laid out [in][out]
    const src = new Uint16Array([1, 2, 3, 4]);
    const expected = new Uint16Array(16);
    expected.set([1, 3], 0); // in 0: out 0, out 1
    expected.set([2, 4], 4); // in 1: out 0, out 1
    expect(packConvHwc4F16(src, 2, 2, 1, 1)).toEqual(expected);
  });

  it('packDwHwc4F16 groups channels by 4 and zero-pads', () => {
    const src = new Uint16Array([1, 2, 3, 4, 5]); // c=5, 1x1
    expect(packDwHwc4F16(src, 5, 1, 1)).toEqual(new Uint16Array([1, 2, 3, 4, 5, 0, 0, 0]));
  });
});
