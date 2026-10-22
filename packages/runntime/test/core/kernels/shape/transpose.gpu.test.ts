import { describe, expect, it } from 'vitest';
import { toArray } from '../../../../src/core/gpu/eval.ts';
import { transpose } from '../../../../src/core/graph/ops.ts';
import { matrix } from '../../../../src/core/graph/value.ts';
import { GRID_Y_STRIDE } from '../../../../src/core/kernels/common.ts';
import { expectClose, gpu, requireFeature, upload } from '../../../gpu.ts';

const { root, ex } = await gpu();

/** out[c, r] = x[r, c] for row-major x [rows, cols]. */
function transposeRef(x: Float32Array, rows: number, cols: number): Float32Array {
  const out = new Float32Array(rows * cols);
  for (let r = 0; r < rows; r++)
    for (let c = 0; c < cols; c++) out[c * rows + r] = x[r * cols + c]!;
  return out;
}

/** 0, 1, 2, …, wrapped under 2048 so f16 holds every value exactly. */
const count = (n: number) => Float32Array.from({ length: n }, (_, i) => i % 2039);

describe('transpose kernel', () => {
  // One route: a thread per element.
  it.for(
    (['f32', 'f16'] as const).flatMap((dtype) =>
      [
        { name: '1×1', rows: 1, cols: 1 },
        { name: 'one row', rows: 1, cols: 7 },
        { name: 'one column', rows: 7, cols: 1 },
        { name: 'odd sizes', rows: 3, cols: 5 },
        { name: 'past a workgroup', rows: 33, cols: 65 },
      ].map((c) => ({ dtype, ...c })),
    ),
  )('$dtype $name: [$rows, $cols]', async ({ dtype, rows, cols }, ctx) => {
    if (dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    const x = count(rows * cols);
    const y = transpose(upload(root, x, { ...matrix(rows, cols), dtype }));
    expect(y.shape).toEqual({ ...matrix(cols, rows), dtype });
    expectClose(await toArray(y, ex), transposeRef(x, rows, cols), { atol: 0, rtol: 0 });
  });

  it('transposed twice gives the input back', async () => {
    const x = count(6 * 11);
    const got = await toArray(transpose(transpose(upload(root, x, [6, 11]))), ex);
    expectClose(got, x, { atol: 0, rtol: 0 });
  });

  it('runs past GRID_Y_STRIDE elements, in the second grid dimension', async () => {
    const rows = 2048;
    const cols = 4097;
    expect(rows * cols).toBeGreaterThan(GRID_Y_STRIDE);
    const x = count(rows * cols);
    const got = await toArray(transpose(upload(root, x, [rows, cols])), ex);
    expectClose(got, transposeRef(x, rows, cols), { atol: 0, rtol: 0 });
  });

  // Known bug: the kernel splits the flat index into row and column with
  // d.u32(i / cols), which TypeGPU compiles to u32(f32(i) / f32(cols)). f32
  // counts exactly only up to 2^24, so past it elements land in the wrong
  // place. Turn into a plain `it` with the fix.
  it.fails('runs past 2^24 elements', async () => {
    const rows = 4100;
    const cols = 4097;
    expect(rows * cols).toBeGreaterThan(2 ** 24);
    const x = count(rows * cols);
    const got = await toArray(transpose(upload(root, x, [rows, cols])), ex);
    expectClose(got, transposeRef(x, rows, cols), { atol: 0, rtol: 0 });
  });

  it('rejects an empty input with an INVALID_ARGUMENT', () => {
    expect(() => upload(root, new Float32Array(0), [0, 4])).toThrow(
      expect.objectContaining({ code: 'INVALID_ARGUMENT' }),
    );
  });

  it('refuses an input that is not 2D', () => {
    const x = upload(root, new Float32Array(8), { elems: 8, dtype: 'f32', dims: [2, 2, 2] });
    expect(() => transpose(x)).toThrow(/expected a 2D f32 or f16 tensor/);
  });
});
