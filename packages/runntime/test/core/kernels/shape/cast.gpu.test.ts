import { describe, expect, it } from 'vitest';
import { toArray } from '../../../../src/core/gpu/eval.ts';
import { astype } from '../../../../src/core/graph/ops.ts';
import { hwc4Meta, matrix } from '../../../../src/core/graph/value.ts';
import { GRID_Y_STRIDE } from '../../../../src/core/kernels/common.ts';
import { expectClose, gpu, randomF32, requireFeature, roundF16, upload } from '../../../gpu.ts';

const { root, ex } = await gpu();

describe('cast kernel', () => {
  // One route per direction: f32 → f16 rounds to nearest even, f16 → f32 is
  // exact.
  it.for([
    { name: '1×1', rows: 1, cols: 1 },
    { name: 'odd sizes', rows: 3, cols: 7 },
    { name: 'one past a workgroup', rows: 1, cols: 65 },
  ])('f32 → f16 → f32, $name: [$rows, $cols]', async ({ rows, cols }, ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const x = randomF32(rows * cols, 1).map((v) => 100 * v);
    const half = astype(upload(root, x, [rows, cols]), 'f16');
    expect(half.shape).toEqual({ ...matrix(rows, cols), dtype: 'f16' });
    expectClose(await toArray(half, ex), roundF16(x), { atol: 0, rtol: 0 });
    const back = astype(half, 'f32');
    expect(back.shape).toEqual(matrix(rows, cols));
    expectClose(await toArray(back, ex), roundF16(x), { atol: 0, rtol: 0 });
  });

  // Nearest even: 1 + 2^-11 is halfway between 1 and the next f16 (1 + 2^-10)
  // and rounds to 1; 1 + 3·2^-11 rounds up to 1 + 2^-9. 65504 is the largest
  // f16, 2^-14 the smallest normal one.
  it('rounds f32 to the nearest f16, ties to even', async (ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const x = new Float32Array([1 + 2 ** -11, 1 + 3 * 2 ** -11, -2.5, 65504, 2 ** -14, 0]);
    const got = await toArray(astype(upload(root, x, [1, 6]), 'f16'), ex);
    expectClose(got, [1, 1 + 2 ** -9, -2.5, 65504, 2 ** -14, 0], { atol: 0, rtol: 0 });
  });

  it('runs past GRID_Y_STRIDE elements, in the second grid dimension', async (ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const n = 2 ** 24 + 3;
    expect(n).toBeGreaterThan(GRID_Y_STRIDE);
    const x = Float32Array.from({ length: n }, (_, i) => (i % 2049) - 1024);
    const half = astype(upload(root, x, [1, n]), 'f16');
    // Integers up to 2048 are exact in f16.
    expectClose(await toArray(astype(half, 'f32'), ex), x, { atol: 0, rtol: 0 });
  });

  it.for(['f32', 'f16'] as const)('%s to itself returns the input, no dispatch', (dtype, ctx) => {
    if (dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    const x = upload(root, new Float32Array(4), { ...matrix(2, 2), dtype });
    expect(astype(x, dtype)).toBe(x);
  });

  it('keeps dims off a flat input', async (ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const x = upload(root, new Float32Array([1, 2, 3]), { elems: 3, dtype: 'f32' });
    const half = astype(x, 'f16');
    expect(half.shape).toEqual({ elems: 3, dtype: 'f16', dims: undefined });
    expectClose(await toArray(half, ex), [1, 2, 3], { atol: 0, rtol: 0 });
  });

  it('rejects an empty input with an INVALID_ARGUMENT', () => {
    expect(() => upload(root, new Float32Array(0), [0, 4])).toThrow(
      expect.objectContaining({ code: 'INVALID_ARGUMENT' }),
    );
  });

  it('refuses an hwc4 input', (ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const x = upload(root, new Float32Array(16), hwc4Meta(3, 2, 2));
    expect(() => astype(x, 'f32')).toThrow(/toChw\(\) converts dtype and layout/);
  });
});
