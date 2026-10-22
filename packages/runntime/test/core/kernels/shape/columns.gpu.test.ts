import { describe, expect, it } from 'vitest';
import { toArray } from '../../../../src/core/gpu/eval.ts';
import { cat, chunk, slice, split } from '../../../../src/core/graph/ops.ts';
import { matrix } from '../../../../src/core/graph/value.ts';
import { GRID_Y_STRIDE } from '../../../../src/core/kernels/common.ts';
import { expectClose, gpu, requireFeature, upload } from '../../../gpu.ts';

const { root, ex } = await gpu();

/** x[r, c] = 1000·r + c, exact in f32 and, while under 2048, in f16. */
const grid = (rows: number, cols: number, scale = 1000) =>
  Float32Array.from({ length: rows * cols }, (_, i) => scale * Math.floor(i / cols) + (i % cols));

/** Columns [start, end) of a row-major [rows, cols] matrix. */
function sliceColsRef(x: Float32Array, rows: number, cols: number, start: number, end: number) {
  const out: number[] = [];
  for (let r = 0; r < rows; r++) out.push(...x.subarray(r * cols + start, r * cols + end));
  return out;
}

/** Row r of the result is row r of a, then row r of b. */
function catColsRef(a: Float32Array, aCols: number, b: Float32Array, bCols: number, rows: number) {
  const out: number[] = [];
  for (let r = 0; r < rows; r++) {
    out.push(...a.subarray(r * aCols, (r + 1) * aCols), ...b.subarray(r * bCols, (r + 1) * bCols));
  }
  return out;
}

describe('columns kernels', () => {
  // sliceCols: slice along dim 1 of a 2D tensor. Copies are exact.
  it.for(
    (['f32', 'f16'] as const).flatMap((dtype) =>
      [
        { name: 'one column of 1×1', rows: 1, cols: 1, start: 0, end: 1 },
        { name: 'the first columns', rows: 3, cols: 7, start: 0, end: 3 },
        { name: 'the middle', rows: 5, cols: 9, start: 2, end: 7 },
        { name: 'up to the last column', rows: 4, cols: 9, start: 5, end: 9 },
        { name: 'a single column', rows: 6, cols: 5, start: 3, end: 4 },
        { name: 'the whole row', rows: 2, cols: 70, start: 0, end: 70 },
      ].map((c) => ({ dtype, ...c })),
    ),
  )('sliceCols $dtype, $name: [$rows, $cols][:, $start:$end]', async (c, ctx) => {
    if (c.dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    const scale = c.dtype === 'f16' ? 100 : 1000;
    const x = grid(c.rows, c.cols, scale);
    const got = await toArray(
      slice(upload(root, x, { ...matrix(c.rows, c.cols), dtype: c.dtype }), 1, c.start, c.end),
      ex,
    );
    expectClose(got, sliceColsRef(x, c.rows, c.cols, c.start, c.end), { atol: 0, rtol: 0 });
  });

  // concatCols: cat along dim 1 of 2D tensors.
  it.for(
    (['f32', 'f16'] as const).flatMap((dtype) =>
      [
        { name: '1×1 and 1×1', rows: 1, aCols: 1, bCols: 1 },
        { name: 'odd widths', rows: 3, aCols: 5, bCols: 2 },
        { name: 'one wide side', rows: 4, aCols: 1, bCols: 67 },
      ].map((c) => ({ dtype, ...c })),
    ),
  )('concatCols $dtype, $name: [$rows, $aCols] ++ [$rows, $bCols]', async (c, ctx) => {
    if (c.dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    const a = grid(c.rows, c.aCols, 100);
    const b = grid(c.rows, c.bCols, 100).map((v) => -v - 1);
    const meta = (cols: number) => ({ ...matrix(c.rows, cols), dtype: c.dtype });
    const got = await toArray(
      cat([upload(root, a, meta(c.aCols)), upload(root, b, meta(c.bCols))], 1),
      ex,
    );
    expectClose(got, catColsRef(a, c.aCols, b, c.bCols, c.rows), { atol: 0, rtol: 0 });
  });

  // cat folds left to right, so three inputs are two concatCols.
  it('cat of three along dim -1', async () => {
    const parts = [grid(2, 3), grid(2, 1).map((v) => v + 0.5), grid(2, 2).map((v) => -v)];
    const got = await toArray(
      cat(
        parts.map((p) => upload(root, p, [2, p.length / 2])),
        -1,
      ),
      ex,
    );
    const ab = Float32Array.from(catColsRef(parts[0]!, 3, parts[1]!, 1, 2));
    expectClose(got, catColsRef(ab, 4, parts[2]!, 2, 2), { atol: 0, rtol: 0 });
  });

  // chunk and split are column slices along dim 1, then cat puts them back.
  it('chunk and split along dim 1 round-trip through cat', async () => {
    const x = grid(3, 12);
    const xv = upload(root, x, [3, 12]);
    const chunks = chunk(xv, 3, 1);
    const parts = split(xv, [5, 7], 1);
    expectClose(await toArray(chunks[1]!, ex), sliceColsRef(x, 3, 12, 4, 8), { atol: 0, rtol: 0 });
    expectClose(await toArray(cat(parts, 1), ex), x, { atol: 0, rtol: 0 });
  });

  /** Slices off the first column of [rows, cols], then puts it back with
   *  cat, checking both. */
  async function sliceAndCat(rows: number, cols: number): Promise<void> {
    const x = Float32Array.from({ length: rows * cols }, (_, i) => i % 8191);
    const xv = upload(root, x, [rows, cols]);
    const got = await toArray(slice(xv, 1, 1, cols), ex);
    expectClose(got, sliceColsRef(x, rows, cols, 1, cols), { atol: 0, rtol: 0 });
    const back = await toArray(cat([slice(xv, 1, 0, 1), slice(xv, 1, 1, cols)], 1), ex);
    expectClose(back, x, { atol: 0, rtol: 0 });
  }

  it('runs past GRID_Y_STRIDE elements, in the second grid dimension', async () => {
    expect(2048 * 4096).toBeGreaterThan(GRID_Y_STRIDE);
    await sliceAndCat(2048, 4097);
  });

  // Known bug: the kernels split the flat index into row and column with
  // d.u32(i / cols), which TypeGPU compiles to u32(f32(i) / f32(cols)). f32
  // counts exactly only up to 2^24, so past it rows and columns come out
  // wrong. Turn into a plain `it` with the fix.
  it.fails('runs past 2^24 elements', async () => {
    expect(4100 * 4096).toBeGreaterThan(2 ** 24);
    await sliceAndCat(4100, 4097);
  });

  it('rejects an empty input with an INVALID_ARGUMENT', () => {
    expect(() => upload(root, new Float32Array(0), [4, 0])).toThrow(
      expect.objectContaining({ code: 'INVALID_ARGUMENT' }),
    );
  });

  it.each([
    { name: 'an empty range', start: 2, end: 2 },
    { name: 'a range past the last column', start: 1, end: 6 },
    { name: 'a negative start', start: -1, end: 2 },
  ])('slice refuses $name', ({ start, end }) => {
    const x = upload(root, new Float32Array(10), [2, 5]);
    expect(() => slice(x, 1, start, end)).toThrow(`slice: bad range [${start}, ${end}) for 5 cols`);
  });

  it('cat refuses inputs with different row counts', () => {
    const a = upload(root, new Float32Array(6), [2, 3]);
    const b = upload(root, new Float32Array(9), [3, 3]);
    expect(() => cat([a, b], 1)).toThrow(/row mismatch — a has 2, b has 3/);
  });

  it('cat refuses inputs of different dtypes', (ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const a = upload(root, new Float32Array(6), [2, 3]);
    const b = upload(root, new Float32Array(6), { ...matrix(2, 3), dtype: 'f16' });
    expect(() => cat([a, b], 1)).toThrow(/dtype mismatch — a is f32, b is f16/);
  });
});
