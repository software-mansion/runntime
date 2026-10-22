import { describe, expect, it } from 'vitest';
import { toArray } from '../../../../src/core/gpu/eval.ts';
import {
  add,
  asinh,
  clamp,
  gelu,
  mul,
  rsqrt,
  sigmoid,
  silu,
  sub,
  tanh,
} from '../../../../src/core/graph/ops.ts';
import { hwc4Meta, matrix, type Value } from '../../../../src/core/graph/value.ts';
import { GRID_Y_STRIDE } from '../../../../src/core/kernels/common.ts';
import { expectClose, gpu, randomF32, requireFeature, roundF16, upload } from '../../../gpu.ts';

const { root, ex } = await gpu();

/** erf from its Maclaurin series, summed in f64. Exact to ~1e-12 for
 *  |x| < 4; past that erf is ±1 to f64 precision. */
function erf(x: number): number {
  if (Math.abs(x) >= 4) return Math.sign(x);
  let term = x;
  let sum = x;
  for (let n = 1; n < 80; n++) {
    term *= (-x * x) / n;
    sum += term / (2 * n + 1);
  }
  return (2 / Math.sqrt(Math.PI)) * sum;
}

/** Each op with its formula, and the inputs it is checked on: random values
 *  spread by `scale`, plus `shift`. */
const OPS: {
  name: string;
  op: (x: Value) => Value;
  ref: (x: number) => number;
  scale?: number;
  shift?: number;
  /** f32 tolerance, when the kernel's formula is looser than a few ulp. */
  tol?: { atol: number; rtol: number };
}[] = [
  // rsqrt: inputs in [0.1, 20.1), where 1/sqrt stays finite
  { name: 'rsqrt', op: rsqrt, ref: (x) => 1 / Math.sqrt(x), scale: 10, shift: 10.1 },
  { name: 'add 2.5', op: (x) => add(x, 2.5), ref: (x) => x + 2.5, scale: 8 },
  { name: 'sub 2.5', op: (x) => sub(x, 2.5), ref: (x) => x - 2.5, scale: 8 },
  { name: 'mul -1.5', op: (x) => mul(x, -1.5), ref: (x) => x * -1.5, scale: 8 },
  { name: 'sigmoid', op: sigmoid, ref: (x) => 1 / (1 + Math.exp(-x)), scale: 8 },
  {
    name: 'clamp to [-1, 2]',
    op: (x) => clamp(x, -1, 2),
    ref: (x) => Math.min(Math.max(x, -1), 2),
    scale: 4,
  },
  {
    name: 'clamp to [0, Infinity)',
    op: (x) => clamp(x, 0, Infinity),
    ref: (x) => Math.max(x, 0),
    scale: 4,
  },
  {
    name: 'clamp to (-Infinity, 0.5]',
    op: (x) => clamp(x, -Infinity, 0.5),
    ref: (x) => Math.min(x, 0.5),
    scale: 4,
  },
  { name: 'tanh', op: tanh, ref: Math.tanh, scale: 4 },
  // gelu: erf from Abramowitz and Stegun 7.1.26, good to 1.5e-7
  {
    name: 'gelu',
    op: gelu,
    ref: (x) => 0.5 * x * (1 + erf(x / Math.SQRT2)),
    scale: 4,
    tol: { atol: 1e-6, rtol: 1e-5 },
  },
  { name: 'silu', op: silu, ref: (x) => x / (1 + Math.exp(-x)), scale: 8 },
  // asinh: log(1 + x) for small x loses the digits past f32's 2^-24
  { name: 'asinh', op: asinh, ref: Math.asinh, scale: 20, tol: { atol: 1e-6, rtol: 1e-5 } },
];

const TOLERANCE = { f32: { atol: 1e-7, rtol: 1e-6 }, f16: { atol: 1e-3, rtol: 2e-3 } };

describe('unary kernels', () => {
  // One route per op: a thread per element, in f32 whatever the storage.
  it.for(
    (['f32', 'f16'] as const).flatMap((dtype) =>
      OPS.flatMap((o) =>
        [
          { shape: '1×1', rows: 1, cols: 1 },
          { shape: 'odd sizes', rows: 37, cols: 53 },
        ].map((s) => ({ dtype, ...o, ...s })),
      ),
    ),
  )('$name $dtype, $shape: [$rows, $cols]', async (c, ctx) => {
    if (c.dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    const values = randomF32(c.rows * c.cols, 1).map((v) => (c.scale ?? 1) * v + (c.shift ?? 0));
    const x = c.dtype === 'f16' ? roundF16(values) : values;
    const y = c.op(upload(root, x, { ...matrix(c.rows, c.cols), dtype: c.dtype }));
    expect(y.shape).toEqual({ ...matrix(c.rows, c.cols), dtype: c.dtype });
    const tol = c.dtype === 'f16' ? TOLERANCE.f16 : (c.tol ?? TOLERANCE.f32);
    expectClose(await toArray(y, ex), Float64Array.from(x, c.ref), tol);
  });

  // Values worked out by hand: sigmoid(0) = 0.5, silu(-1) = -1/(1 + e),
  // gelu(1) = Φ(1) = 0.841344746, asinh(1) = ln(1 + √2), rsqrt(4) = 0.5.
  it.each([
    { name: 'sigmoid(0)', op: sigmoid, x: 0, want: 0.5 },
    { name: 'silu(-1)', op: silu, x: -1, want: -1 / (1 + Math.E) },
    { name: 'gelu(1)', op: gelu, x: 1, want: 0.841344746 },
    { name: 'asinh(1)', op: asinh, x: 1, want: Math.log(1 + Math.SQRT2) },
    { name: 'rsqrt(4)', op: rsqrt, x: 4, want: 0.5 },
  ])('$name', async ({ op, x, want }) => {
    const got = await toArray(op(upload(root, new Float32Array([x]), [1, 1])), ex);
    expectClose(got, [want], { atol: 1e-7, rtol: 1e-6 });
  });

  // Far from 0 each op has to saturate, not overflow into NaN: exp(−x)
  // passes f32's range past |x| = 88.
  it.each([
    { name: 'sigmoid', op: sigmoid, want: [0, 1] },
    { name: 'silu', op: silu, want: [-0, 1000] },
    { name: 'gelu', op: gelu, want: [-0, 1000] },
  ])('$name saturates at ±1000', async ({ op, want }) => {
    const got = await toArray(op(upload(root, new Float32Array([-1000, 1000]), [1, 2])), ex);
    expectClose(got, want, { atol: 1e-6, rtol: 1e-6 });
  });

  // Known bug: the tanh kernel calls WGSL tanh, which returns NaN for large
  // |x| on some backends (Metal here) where it should saturate to ±1. The
  // geluTanh epilogue already works around it with 1 − 2/(e^{2x} + 1).
  // Turn into a plain `it` with the fix.
  it.fails('tanh saturates at ±1000', async () => {
    const got = await toArray(tanh(upload(root, new Float32Array([-1000, 1000]), [1, 2])), ex);
    expectClose(got, [-1, 1], { atol: 1e-6, rtol: 1e-6 });
  });

  // Known bug: asinh squares x, which overflows f32 past |x| ≈ 1.8e19, so it
  // returns ±Infinity where torch.asinh(1e20) is 46.7448. Turn into a plain
  // `it` with the fix.
  it.fails('asinh past the square root of f32 max', async () => {
    const got = await toArray(asinh(upload(root, new Float32Array([1e20, -1e20]), [1, 2])), ex);
    expectClose(got, [Math.asinh(1e20), Math.asinh(-1e20)], { atol: 0, rtol: 1e-6 });
  });

  it('runs past 2^24 elements, in the second grid dimension', async () => {
    const n = 2 ** 24 + 3;
    expect(n).toBeGreaterThan(GRID_Y_STRIDE);
    const x = Float32Array.from({ length: n }, (_, i) => (i % 13) - 6);
    const got = await toArray(mul(upload(root, x, [1, n]), 2), ex);
    expectClose(
      got,
      x.map((v) => 2 * v),
      { atol: 0, rtol: 0 },
    );
  });

  it('rejects an empty input with an INVALID_ARGUMENT', () => {
    expect(() => upload(root, new Float32Array(0), [0, 4])).toThrow(
      expect.objectContaining({ code: 'INVALID_ARGUMENT' }),
    );
  });

  it('clamp refuses lo above hi', () => {
    const x = upload(root, new Float32Array(4), [2, 2]);
    expect(() => clamp(x, 2, 1)).toThrow(/lo \(2\) must be <= hi \(1\)/);
  });

  // hwc4 storage pads channels with zeros, so an op with f(0) != 0 would
  // write into the padding.
  it.for([
    { name: 'rsqrt', op: rsqrt },
    { name: 'sigmoid', op: sigmoid },
    { name: 'add of a non-zero number', op: (x: Value) => add(x, 1) },
    { name: 'clamp to a range without 0', op: (x: Value) => clamp(x, 1, 2) },
  ])('$name refuses hwc4', ({ op }, ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const x = upload(root, new Float32Array(16), hwc4Meta(3, 2, 2));
    expect(() => op(x)).toThrow(/refuses hwc4 input/);
  });
});
