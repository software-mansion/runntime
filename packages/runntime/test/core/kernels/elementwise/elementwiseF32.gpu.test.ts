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
  // Every op in every way b can be read, worked out by hand from
  // A = [[1, 2, 3], [4, 5, 6]]. broadcastPlan in graph/ops/elementwise.ts picks
  // the mode from b's shape.
  const ROW_WANT = {
    add: [1.5, 1, 5, 4.5, 4, 8],
    sub: [0.5, 3, 1, 3.5, 6, 4],
    mul: [0.5, -2, 6, 2, -5, 12],
  };
  const COL_WANT = {
    add: [3, 4, 5, 3.5, 4.5, 5.5],
    sub: [-1, 0, 1, 4.5, 5.5, 6.5],
    mul: [2, 4, 6, -2, -2.5, -3],
  };
  const SAME_WANT = {
    add: [1.5, 3, 5, 3, 9, 6.5],
    sub: [0.5, 1, 1, 5, 1, 5.5],
    mul: [0.5, 2, 6, -4, 20, 3],
  };
  const BROADCASTS = [
    // mode 0: b has a's shape
    { name: 'same shape', b: [0.5, 1, 2, -1, 4, 0.5], dims: [2, 3], want: SAME_WANT },
    // mode 1: b is one row, as [1, N] or [N]
    { name: 'row [1, N]', b: [0.5, -1, 2], dims: [1, 3], want: ROW_WANT },
    { name: 'row [N]', b: [0.5, -1, 2], dims: [3], want: ROW_WANT },
    // mode 2: b is one column [M, 1]
    { name: 'column [M, 1]', b: [2, -0.5], dims: [2, 1], want: COL_WANT },
    // mode 3: b is a single value
    {
      name: 'single value',
      b: [-2],
      dims: [1, 1],
      want: { add: [-1, 0, 1, 2, 3, 4], sub: [3, 4, 5, 6, 7, 8], mul: [-2, -4, -6, -8, -10, -12] },
    },
    // b without dims: the mode comes from its element count alone, M * N
    // elements mode 0, N mode 1, M mode 2
    { name: 'same count, no dims', b: [0.5, 1, 2, -1, 4, 0.5], want: SAME_WANT },
    { name: 'row, no dims', b: [0.5, -1, 2], want: ROW_WANT },
    { name: 'column, no dims', b: [2, -0.5], want: COL_WANT },
  ];
  const OPS = { add, sub, mul };

  // Small integers and halves are exact in f32 and f16, so these need no tolerance.
  it.for(
    (['f32', 'f16'] as const).flatMap((dtype) =>
      BROADCASTS.flatMap((c) =>
        (['add', 'sub', 'mul'] as const).map((op) => ({ dtype, op, ...c, want: c.want[op] })),
      ),
    ),
  )('$dtype $op, b $name', async ({ dtype, op, b, dims, want }, ctx) => {
    if (dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    const a = upload(root, new Float32Array(A), { ...matrix(2, 3), dtype });
    const bv = upload(root, new Float32Array(b), { elems: b.length, dtype, dims });
    expectClose(await toArray(OPS[op](a, bv), ex), want, { atol: 0, rtol: 0 });
  });

  // The op puts the full operand first, so a broadcast b passed as the first
  // argument gives the same result. sub is not commutative and refuses it.
  it.each([
    { name: 'add, row first', op: add, b: [0.5, -1, 2], bDims: [1, 3], want: ROW_WANT.add },
    { name: 'mul, column first', op: mul, b: [2, -0.5], bDims: [2, 1], want: COL_WANT.mul },
  ])('$name', async ({ op, b, bDims, want }) => {
    const a = upload(root, new Float32Array(A), [2, 3]);
    const bv = upload(root, new Float32Array(b), [bDims[0]!, bDims[1]!]);
    expectClose(await toArray(op(bv, a), ex), want, { atol: 0, rtol: 0 });
  });

  it('sub refuses a broadcast operand first', () => {
    const a = upload(root, new Float32Array(A), [2, 3]);
    const row = upload(root, new Float32Array(3), [1, 3]);
    expect(() => sub(row, a)).toThrow(/a must be the full operand/);
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

  it('rejects an empty input with an INVALID_ARGUMENT', () => {
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
