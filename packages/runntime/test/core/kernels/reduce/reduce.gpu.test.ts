import { describe, expect, it } from 'vitest';
import { toArray } from '../../../../src/core/gpu/eval.ts';
import { mean } from '../../../../src/core/graph/ops.ts';
import { matrix } from '../../../../src/core/graph/value.ts';
import { GRID_Y_STRIDE } from '../../../../src/core/kernels/common.ts';
import { expectClose, gpu, randomF32, requireFeature, roundF16, upload } from '../../../gpu.ts';

const { root, ex } = await gpu();

/** Mean of each row, in f64. */
function meanRef(x: Float32Array, rows: number, cols: number): Float64Array {
  const out = new Float64Array(rows);
  for (let r = 0; r < rows; r++) {
    let sum = 0;
    for (let j = 0; j < cols; j++) sum += x[r * cols + j]!;
    out[r] = sum / cols;
  }
  return out;
}

describe('mean kernel', () => {
  // One route: a thread per row sums it in f32.
  it.for(
    (['f32', 'f16'] as const).flatMap((dtype) =>
      [
        { name: '1×1', rows: 1, cols: 1 },
        { name: 'odd sizes', rows: 3, cols: 7 },
        { name: 'one row past a workgroup', rows: 65, cols: 16 },
        { name: 'a wide row', rows: 2, cols: 10001 },
      ].map((shape) => ({ dtype, ...shape })),
    ),
  )('$dtype $name: [$rows, $cols]', async ({ dtype, rows, cols }, ctx) => {
    if (dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    const x = dtype === 'f16' ? roundF16(randomF32(rows * cols, 1)) : randomF32(rows * cols, 1);
    const got = await toArray(mean(upload(root, x, { ...matrix(rows, cols), dtype })), ex);
    // f32: a sequential sum of `cols` values. f16 output adds 2^-11 relative.
    const tol = dtype === 'f16' ? { atol: 1e-3, rtol: 1e-3 } : { atol: 1e-6, rtol: 1e-5 };
    expectClose(got, meanRef(x, rows, cols), tol);
  });

  // [1, 2, 4] → 7 / 3, [-1, 1, 0.5] → 1 / 6, as torch.mean(x, dim=-1).
  it('matches torch.mean worked out by hand', async () => {
    const x = upload(root, new Float32Array([1, 2, 4, -1, 1, 0.5]), [2, 3]);
    const got = await toArray(mean(x), ex);
    expectClose(got, [7 / 3, 1 / 6], { atol: 0, rtol: 1e-6 });
  });

  it('returns [M, 1]', () => {
    const x = upload(root, new Float32Array(6), [2, 3]);
    expect(mean(x).shape).toEqual({ elems: 2, dtype: 'f32', dims: [2, 1] });
  });

  // Past GRID_Y_STRIDE rows the threads take the second grid dimension.
  it('runs past GRID_Y_STRIDE rows', async () => {
    const rows = GRID_Y_STRIDE + 5;
    const x = Float32Array.from({ length: rows * 2 }, (_, i) => i % 9);
    const got = await toArray(mean(upload(root, x, [rows, 2])), ex);
    expectClose(got, meanRef(x, rows, 2), { atol: 0, rtol: 0 });
  });

  it('runs past 2^24 elements', async () => {
    const rows = 4100;
    const cols = 4096;
    expect(rows * cols).toBeGreaterThan(2 ** 24);
    const x = randomF32(rows * cols, 2);
    const got = await toArray(mean(upload(root, x, [rows, cols])), ex);
    expectClose(got, meanRef(x, rows, cols), { atol: 1e-6, rtol: 1e-5 });
  });

  it('rejects an empty input with an INVALID_ARGUMENT', () => {
    expect(() => upload(root, new Float32Array(0), [0, 4])).toThrow(
      expect.objectContaining({ code: 'INVALID_ARGUMENT' }),
    );
  });

  it('refuses an input that is not 2D', () => {
    const x = upload(root, new Float32Array(8), { elems: 8, dtype: 'f32', dims: [8] });
    expect(() => mean(x)).toThrow(/expected a 2D f32 or f16 tensor/);
  });
});
