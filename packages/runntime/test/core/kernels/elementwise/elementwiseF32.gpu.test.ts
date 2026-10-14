import { describe, expect, it } from 'vitest';
import { toArray } from '../../../../src/core/gpu/eval.ts';
import { add, mul, sub, swiglu } from '../../../../src/core/graph/ops.ts';
import { matrix } from '../../../../src/core/graph/value.ts';
import { GRID_Y_STRIDE } from '../../../../src/core/kernels/common.ts';
import { expectClose, gpu, randomF32, requireFeature, upload } from '../../../gpu.ts';

const { root, ex } = await gpu();

const A = [1, 2, 3, 4, 5, 6]; // [2, 3]

/** Clamped SwiGLU from its definition: g·σ(1.702·g)·(l + 1), g = min(glu, 7),
 *  l = clamp(lin, ±7). */
function swigluRef(glu: Float32Array, lin: Float32Array): Float64Array {
  return Float64Array.from(glu, (x, i) => {
    const g = Math.min(x, 7);
    const l = Math.min(Math.max(lin[i]!, -7), 7);
    return (g / (1 + Math.exp(-1.702 * g))) * (l + 1);
  });
}

describe('elementwiseF32 kernel', () => {
  // Small integers and halves are exact in f32 and f16, so these need no tolerance.
  it.each([
    {
      name: 'add, same shape',
      op: add,
      b: [10, 20, 30, 40, 50, 60],
      bDims: [2, 3],
      want: [11, 22, 33, 44, 55, 66],
    },
    {
      name: 'add, row [1, N]',
      op: add,
      b: [10, 20, 30],
      bDims: [1, 3],
      want: [11, 22, 33, 14, 25, 36],
    },
    {
      name: 'add, column [M, 1]',
      op: add,
      b: [10, 20],
      bDims: [2, 1],
      want: [11, 12, 13, 24, 25, 26],
    },
    { name: 'sub, row [1, N]', op: sub, b: [1, 1, 1], bDims: [1, 3], want: [0, 1, 2, 3, 4, 5] },
    {
      name: 'mul, same shape',
      op: mul,
      b: [2, 2, 2, 0.5, 0.5, 0.5],
      bDims: [2, 3],
      want: [2, 4, 6, 2, 2.5, 3],
    },
    { name: 'mul, single value', op: mul, b: [-1], bDims: [1, 1], want: [-1, -2, -3, -4, -5, -6] },
  ])('$name', async ({ op, b, bDims, want }) => {
    const a = upload(root, new Float32Array(A), [2, 3]);
    const bv = upload(root, new Float32Array(b), [bDims[0]!, bDims[1]!]);
    expectClose(await toArray(op(a, bv), ex), want, { atol: 0, rtol: 0 });
  });

  it('adds 1×1', async () => {
    const a = upload(root, new Float32Array([2.5]), [1, 1]);
    const b = upload(root, new Float32Array([-0.5]), [1, 1]);
    expectClose(await toArray(add(a, b), ex), [2], { atol: 0, rtol: 0 });
  });

  it('swiglu, with values past both clamps', async () => {
    const glu = new Float32Array([-9, -1, 0, 0.5, 3, 7, 12]);
    const lin = new Float32Array([-12, -7, -0.5, 0, 2, 7, 9]);
    const got = await toArray(swiglu(upload(root, glu, [1, 7]), upload(root, lin, [1, 7])), ex);
    expectClose(got, swigluRef(glu, lin), { atol: 1e-6, rtol: 1e-5 });
  });

  it.for([
    { name: 'add, row [1, N]', op: add, want: [1.5, 2.25, 2, 4.5, 5.25, 5] },
    { name: 'mul, row [1, N]', op: mul, want: [0.5, 0.5, -3, 2, 1.25, -6] },
  ])('f16 $name', async ({ op, want }, ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const a = upload(root, new Float32Array(A), { ...matrix(2, 3), dtype: 'f16' });
    const b = upload(root, new Float32Array([0.5, 0.25, -1]), { ...matrix(1, 3), dtype: 'f16' });
    expectClose(await toArray(op(a, b), ex), want, { atol: 0, rtol: 0 });
  });

  // The kernel reads lin[i] with no broadcast index, so a smaller lin would be
  // read past its end. The op has to refuse it before the kernel runs.
  it('swiglu refuses a broadcast lin', () => {
    const glu = upload(root, new Float32Array(6), [2, 3]);
    const lin = upload(root, new Float32Array(3), [1, 3]);
    expect(() => swiglu(glu, lin)).toThrow(/glu\/lin must match/);
  });

  it('f16 swiglu', async (ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const glu = new Float32Array([-9, -1, 0, 0.5, 3, 7, 12]);
    const lin = new Float32Array([-12, -7, -0.5, 0, 2, 7, 9]);
    const f16 = { ...matrix(1, 7), dtype: 'f16' } as const;
    const got = await toArray(swiglu(upload(root, glu, f16), upload(root, lin, f16)), ex);
    // The inputs are exact in f16; the output is rounded to f16 (2^-11 relative).
    expectClose(got, swigluRef(glu, lin), { atol: 1e-3, rtol: 1e-3 });
  });

  it('adds past 2^24 elements, which takes the second grid dimension', async () => {
    const n = 2 ** 24 + 3;
    expect(n).toBeGreaterThan(GRID_Y_STRIDE);
    const a = Float32Array.from({ length: n }, (_, i) => i % 7);
    const b = new Float32Array(n).fill(1);
    const got = await toArray(add(upload(root, a, [1, n]), upload(root, b, [1, n])), ex);
    expectClose(
      got,
      Float32Array.from(a, (x) => x + 1),
      { atol: 0, rtol: 0 },
    );
  });

  it('adds random values in a shape no vec4 or workgroup divides', async () => {
    const a = randomF32(37 * 53, 1);
    const b = randomF32(53, 2);
    const got = await toArray(add(upload(root, a, [37, 53]), upload(root, b, [1, 53])), ex);
    // A sum of two f32 is exact in f64, so storing it as f32 rounds it once,
    // the same as IEEE f32 addition on the GPU.
    expectClose(
      got,
      Float32Array.from(a, (x, i) => x + b[i % 53]!),
      { atol: 0, rtol: 0 },
    );
  });

  // Known bug: an empty tensor reaches createBuffer and fails with the
  // browser's TypeError instead of a RunntimeError. Turn into `it` with the fix.
  it.fails('rejects an empty input with an INVALID_ARGUMENT', () => {
    expect(() => upload(root, new Float32Array(0), [0, 3])).toThrow(
      expect.objectContaining({ code: 'INVALID_ARGUMENT' }),
    );
  });

  it('refuses inputs of different dtypes', () => {
    const a = upload(root, new Float32Array(6), [2, 3]);
    const b = upload(root, new Float32Array(6), { ...matrix(2, 3), dtype: 'f16' });
    expect(() => add(a, b)).toThrow(/incompatible shapes — a=\(6,f32\) b=\(6,f16\)/);
  });

  it('refuses an input past maxStorageBufferBindingSize instead of running on it', () => {
    const n = root.device.limits.maxStorageBufferBindingSize / 4 + 1;
    expect(() => upload(root, new Float32Array(n), [1, n])).toThrow(
      expect.objectContaining({ code: 'UNSUPPORTED_DEVICE' }),
    );
  });
});
