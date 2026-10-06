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

// f16 is 1 sign bit, 5 exponent bits, 10 mantissa bits. Every value here is
// exactly representable, so each pair must convert both ways.
const F16_EXACT = [
  { name: 'zero', bits: 0x0000, value: 0 },
  { name: 'one', bits: 0x3c00, value: 1 },
  { name: 'one and a half', bits: 0x3e00, value: 1.5 },
  { name: 'minus one and a half', bits: 0xbe00, value: -1.5 },
  { name: 'a half', bits: 0x3800, value: 0.5 },
  { name: 'closest f16 to 0.1', bits: 0x2e66, value: 0.0999755859375 },
  { name: 'closest f16 to 1/3', bits: 0x3555, value: 0.333251953125 },
  { name: 'closest f16 to -0.3', bits: 0xb4cd, value: -0.300048828125 },
  { name: 'closest f16 to pi', bits: 0x4248, value: 3.140625 },
  { name: 'largest finite', bits: 0x7bff, value: 65504 },
  { name: 'smallest normal', bits: 0x0400, value: 2 ** -14 },
  { name: 'largest subnormal', bits: 0x03ff, value: 1023 * 2 ** -24 },
  { name: 'a mid subnormal', bits: 0x0200, value: 2 ** -15 },
  { name: 'smallest subnormal', bits: 0x0001, value: 2 ** -24 },
  { name: 'infinity', bits: 0x7c00, value: Infinity },
  { name: 'minus infinity', bits: 0xfc00, value: -Infinity },
];

const F16_ONE = 0x3c00;
const F16_ONE_AND_TWO_ULPS = 0x3c02; // 1 + 2^-9, two steps above one
const F16_MAX = 0x7bff;
const F16_NEG_MAX = 0xfbff;
const F16_NAN = 0x7e00;

describe('f16BitsToF32', () => {
  it.each(F16_EXACT)('$name', ({ bits, value }) => {
    expect(f16BitsToF32(bits)).toBe(value);
  });

  it('keeps the sign of zero', () => {
    expect(Object.is(f16BitsToF32(0x8000), -0)).toBe(true);
  });

  it('decodes NaN', () => {
    expect(f16BitsToF32(F16_NAN)).toBeNaN();
  });
});

describe('f32ToF16Bits', () => {
  it.each(F16_EXACT)('$name', ({ bits, value }) => {
    expect(f32ToF16Bits(value)).toBe(bits);
  });

  // Values f16 cannot hold exactly, rounded to the nearest one.
  it.each([
    { name: '0.1', value: 0.1, bits: 0x2e66 },
    { name: 'pi', value: Math.PI, bits: 0x4248 },
    { name: '-0.3', value: -0.3, bits: 0xb4cd },
    { name: '1e-5 (subnormal)', value: 1e-5, bits: 0x00a8 },
    { name: '6e-8 (rounds up to the smallest subnormal)', value: 6e-8, bits: 0x0001 },
    { name: '2^-26 (rounds down to zero)', value: 2 ** -26, bits: 0x0000 },
  ])('rounds $name', ({ value, bits }) => {
    expect(f32ToF16Bits(value)).toBe(bits);
  });

  it('encodes NaN', () => {
    expect(f32ToF16Bits(NaN)).toBe(F16_NAN);
  });

  it('clamps finite overflow to the f16 max instead of Infinity', () => {
    expect(f32ToF16Bits(1e6)).toBe(F16_MAX);
    expect(f32ToF16Bits(-1e6)).toBe(F16_NEG_MAX);
  });

  it('rounds halfway cases to even', () => {
    // halfway between one and the next f16 up: rounds down to one
    expect(f32ToF16Bits(1 + 2 ** -11)).toBe(F16_ONE);
    // halfway between one step and two steps above one: rounds up to two steps
    expect(f32ToF16Bits(1 + 3 * 2 ** -11)).toBe(F16_ONE_AND_TWO_ULPS);
  });

  it('picks the nearest f16 for random values', () => {
    // Fixed-seed random values across the whole f16 range, normal and subnormal.
    // No neighbour of the chosen f16 may be closer to the input.
    let seed = 1;
    const random = () => ((seed = (seed * 48271) % 0x7fffffff) - 1) / 0x7ffffffe;
    const wrong: number[] = [];
    for (let i = 0; i < 10_000; i++) {
      const x = Math.fround((random() < 0.5 ? -1 : 1) * 2 ** (random() * 40 - 24));
      const bits = f32ToF16Bits(x);
      const mag = bits & 0x7fff;
      const err = Math.abs(Math.abs(x) - f16BitsToF32(mag));
      const below = mag > 0 ? Math.abs(Math.abs(x) - f16BitsToF32(mag - 1)) : Infinity;
      const above = mag < F16_MAX ? Math.abs(Math.abs(x) - f16BitsToF32(mag + 1)) : Infinity;
      if ((bits & 0x8000) !== (x < 0 ? 0x8000 : 0) || below < err || above < err) wrong.push(x);
    }
    expect(wrong).toEqual([]);
  });

  it('round-trips every finite f16', () => {
    const mismatches: number[] = [];
    for (let bits = 0; bits <= 0xffff; bits++) {
      if (((bits >>> 10) & 0x1f) === 31) continue; // exponent all ones: Inf / NaN
      if (f32ToF16Bits(f16BitsToF32(bits)) !== bits) mismatches.push(bits);
    }
    expect(mismatches).toEqual([]);
  });
});

// bf16 is the top 16 bits of an f32.
const BF16 = [
  { name: 'zero', bits: 0x0000, value: 0 },
  { name: 'one and a half', bits: 0x3fc0, value: 1.5 },
  { name: 'minus a quarter', bits: 0xbe80, value: -0.25 },
  { name: 'closest bf16 to 0.1', bits: 0x3dcd, value: 0.10009765625 },
  { name: 'closest bf16 to pi', bits: 0x4049, value: 3.140625 },
  { name: 'smallest subnormal', bits: 0x0001, value: 2 ** -133 },
  { name: 'infinity', bits: 0x7f80, value: Infinity },
];

describe('array conversions', () => {
  it('f16ToF32', () => {
    const bits = new Uint16Array(F16_EXACT.map((c) => c.bits));
    expect(f16ToF32(bits)).toEqual(new Float32Array(F16_EXACT.map((c) => c.value)));
  });

  it('bf16ToF32', () => {
    const bits = new Uint16Array(BF16.map((c) => c.bits));
    expect(bf16ToF32(bits)).toEqual(new Float32Array(BF16.map((c) => c.value)));
  });

  it('f32ArrayToF16Bits counts clamped values', () => {
    const { data, clamped } = f32ArrayToF16Bits(new Float32Array([0.1, 70000, -70000, 65504]));
    expect(data).toEqual(new Uint16Array([0x2e66, F16_MAX, F16_NEG_MAX, F16_MAX]));
    expect(clamped).toBe(2); // 65504 fits exactly, so it does not count
  });

  it('bf16ToF16Bits rounds to the nearest f16 and counts clamped values', () => {
    const src = new Uint16Array([
      0x3fc0, // 1.5, exact in f16
      0x3dcd, // 0.10009765625, also exact in f16
      0x0001, // 2^-133, far below the smallest f16, becomes zero
      0x4780, // 65536, just past the f16 max
    ]);
    const { data, clamped } = bf16ToF16Bits(src);
    expect(data).toEqual(new Uint16Array([0x3e00, 0x2e68, 0x0000, F16_MAX]));
    expect(clamped).toBe(1);
  });
});

describe('array conversion edge cases', () => {
  it('empty inputs give empty outputs', () => {
    const none16 = new Uint16Array(0);
    expect(f16ToF32(none16)).toEqual(new Float32Array(0));
    expect(bf16ToF32(none16)).toEqual(new Float32Array(0));
    expect(f32ArrayToF16Bits(new Float32Array(0))).toEqual({ data: none16, clamped: 0 });
    expect(bf16ToF16Bits(none16)).toEqual({ data: none16, clamped: 0 });
  });

  it('a single element', () => {
    expect(f16ToF32(new Uint16Array([0x3e00]))).toEqual(new Float32Array([1.5]));
    expect(bf16ToF32(new Uint16Array([0xbe80]))).toEqual(new Float32Array([-0.25]));
    expect(f32ArrayToF16Bits(new Float32Array([0.5]))).toEqual({
      data: new Uint16Array([0x3800]),
      clamped: 0,
    });
  });

  it('does not count Infinity, NaN or -0 as clamped', () => {
    const { data, clamped } = f32ArrayToF16Bits(new Float32Array([Infinity, NaN, -0]));
    expect(data).toEqual(new Uint16Array([0x7c00, F16_NAN, 0x8000]));
    expect(clamped).toBe(0);
  });
});

// Weight values here are just labels (1, 2, 3...) so you can follow where each
// one lands. The layout comments use [outer, ..., inner] order.
describe('transposeF32', () => {
  it.each([
    { name: '2x3', rows: 2, cols: 3, src: [1, 2, 3, 4, 5, 6], want: [1, 4, 2, 5, 3, 6] },
    { name: '1x1', rows: 1, cols: 1, src: [7], want: [7] },
    { name: 'a single row', rows: 1, cols: 3, src: [1, 2, 3], want: [1, 2, 3] },
    { name: 'a single column', rows: 3, cols: 1, src: [1, 2, 3], want: [1, 2, 3] },
    { name: 'empty', rows: 0, cols: 3, src: [], want: [] },
  ])('$name', ({ rows, cols, src, want }) => {
    expect(transposeF32(new Float32Array(src), rows, cols)).toEqual(new Float32Array(want));
  });

  it('rejects a wrong length', () => {
    expect(() => transposeF32(new Float32Array(5), 2, 3)).toThrow(/length 5/);
    expect(() => transposeF32(new Float32Array(1), 0, 0)).toThrow(/length 1/);
  });
});

describe('convToTapMajor', () => {
  // [cOut, cIn, k] -> [k, cIn, cOut]
  it.each([
    { name: '2 out, 1 in, 2 taps', cOut: 2, cIn: 1, k: 2, src: [1, 2, 3, 4], want: [1, 3, 2, 4] },
    {
      name: '1 out, 2 in, 3 taps',
      cOut: 1,
      cIn: 2,
      k: 3,
      src: [1, 2, 3, 4, 5, 6],
      want: [1, 4, 2, 5, 3, 6],
    },
    { name: 'a single weight', cOut: 1, cIn: 1, k: 1, src: [7], want: [7] },
    { name: 'empty', cOut: 0, cIn: 0, k: 0, src: [], want: [] },
  ])('$name', ({ cOut, cIn, k, src, want }) => {
    expect(convToTapMajor(new Float32Array(src), cOut, cIn, k)).toEqual(new Float32Array(want));
  });

  it('rejects a wrong length', () => {
    expect(() => convToTapMajor(new Float32Array(3), 2, 1, 2)).toThrow(/length 3/);
  });
});

describe('permuteConvTransposeF16', () => {
  // [cIn, cOut, k*k] -> [cOut, k*k, cIn]
  it.each([
    { name: '2 in, 2 out, 1x1', cIn: 2, cOut: 2, k: 1, src: [1, 2, 3, 4], want: [1, 3, 2, 4] },
    {
      name: '2 in, 1 out, 2x2',
      cIn: 2,
      cOut: 1,
      k: 2,
      src: [1, 2, 3, 4, 5, 6, 7, 8],
      want: [1, 5, 2, 6, 3, 7, 4, 8],
    },
    { name: 'a single weight', cIn: 1, cOut: 1, k: 1, src: [7], want: [7] },
    { name: 'empty', cIn: 0, cOut: 0, k: 1, src: [], want: [] },
  ])('$name', ({ cIn, cOut, k, src, want }) => {
    expect(permuteConvTransposeF16(new Uint16Array(src), cIn, cOut, k)).toEqual(
      new Uint16Array(want),
    );
  });
});

describe('packConvHwc4F16', () => {
  // [cOut, cIn, kH*kW] -> 4x4 tiles of [in % 4][out % 4], one tile per
  // (out block, in block, tap). Channels past cOut / cIn are zero.
  function tiles(count: number, entries: Record<number, number>): Uint16Array {
    const out = new Uint16Array(count * 16);
    for (const [i, v] of Object.entries(entries)) out[Number(i)] = v;
    return out;
  }

  it.each([
    {
      name: '2 out, 2 in, 1x1: one tile',
      cOut: 2,
      cIn: 2,
      kH: 1,
      kW: 1,
      src: [1, 2, 3, 4],
      want: tiles(1, { 0: 1, 1: 3, 4: 2, 5: 4 }),
    },
    {
      name: '5 out: a second out block holding one channel',
      cOut: 5,
      cIn: 1,
      kH: 1,
      kW: 1,
      src: [1, 2, 3, 4, 5],
      want: tiles(2, { 0: 1, 1: 2, 2: 3, 3: 4, 16: 5 }),
    },
    {
      name: '1x2 kernel: one tile per tap',
      cOut: 1,
      cIn: 1,
      kH: 1,
      kW: 2,
      src: [1, 2],
      want: tiles(2, { 0: 1, 16: 2 }),
    },
    {
      name: 'a single weight',
      cOut: 1,
      cIn: 1,
      kH: 1,
      kW: 1,
      src: [7],
      want: tiles(1, { 0: 7 }),
    },
    { name: 'empty', cOut: 0, cIn: 0, kH: 1, kW: 1, src: [], want: tiles(0, {}) },
  ])('$name', ({ cOut, cIn, kH, kW, src, want }) => {
    expect(packConvHwc4F16(new Uint16Array(src), cOut, cIn, kH, kW)).toEqual(want);
  });
});

describe('packDwHwc4F16', () => {
  // [c, kH*kW] -> [c / 4, kH*kW, 4], channels past c are zero
  it.each([
    {
      name: '5 channels, 1x1',
      c: 5,
      kH: 1,
      kW: 1,
      src: [1, 2, 3, 4, 5],
      want: [1, 2, 3, 4, 5, 0, 0, 0],
    },
    {
      name: '2 channels, 1x2',
      c: 2,
      kH: 1,
      kW: 2,
      src: [1, 2, 3, 4],
      want: [1, 3, 0, 0, 2, 4, 0, 0],
    },
    { name: 'a single weight', c: 1, kH: 1, kW: 1, src: [7], want: [7, 0, 0, 0] },
    { name: 'empty', c: 0, kH: 1, kW: 1, src: [], want: [] },
  ])('$name', ({ c, kH, kW, src, want }) => {
    expect(packDwHwc4F16(new Uint16Array(src), c, kH, kW)).toEqual(new Uint16Array(want));
  });
});
