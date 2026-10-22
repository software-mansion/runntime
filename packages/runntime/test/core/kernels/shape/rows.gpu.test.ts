import { describe, expect, it } from 'vitest';
import { evalValues, toArray } from '../../../../src/core/gpu/eval.ts';
import {
  cat,
  gatherRows,
  gatherRowsFrom,
  reshape,
  slice,
  writeRows,
} from '../../../../src/core/graph/ops.ts';
import { matrix, tensor3d } from '../../../../src/core/graph/value.ts';
import { GRID_Y_STRIDE } from '../../../../src/core/kernels/common.ts';
import { expectClose, gpu, requireFeature, upload } from '../../../gpu.ts';

const { root, ex } = await gpu();

/** 0, 1, 2, … shifted by `from`, exact in f32 and, while under 2048, in f16. */
const count = (n: number, from = 0) => Float32Array.from({ length: n }, (_, i) => from + i);

const EXACT = { atol: 0, rtol: 0 };

interface GatherCase {
  name: string;
  /** Rows of the table gathered from. */
  table: number;
  cols: number;
  /** Rows gathered. */
  count: number;
  op: 'gatherRows' | 'gatherRowsFrom';
  dtype?: 'f32' | 'f16';
}

/** out[i, c] = x[idx[i], c], with x[r, c] = r · cols + c so every value says
 *  where it came from. Every table stays under 2048 values, exact in f16. */
async function checkGather({
  table,
  cols,
  count: gathered,
  op,
  dtype = 'f32',
}: GatherCase): Promise<void> {
  const x = Float32Array.from({ length: table * cols }, (_, i) => i);
  const idx = Float32Array.from({ length: gathered }, (_, i) => (i * 7) % table);
  const xs = upload(root, x, { ...matrix(table, cols), dtype });
  const y =
    op === 'gatherRows'
      ? gatherRows(xs, Array.from(idx))
      : gatherRowsFrom(xs, upload(root, idx, matrix(gathered, 1)));
  const want = new Float32Array(gathered * cols);
  for (let i = 0; i < gathered; i++) {
    for (let c = 0; c < cols; c++) want[i * cols + c] = x[idx[i]! * cols + c]!;
  }
  expectClose(await toArray(y, ex), want, EXACT);
}

describe('rows kernels', () => {
  // sliceRows: slice along dim 0 of a 2D tensor is a flat copy from start·cols.
  it.for(
    (['f32', 'f16'] as const).flatMap((dtype) =>
      [
        { name: '1×1', rows: 1, cols: 1, start: 0, end: 1 },
        { name: 'the first rows', rows: 5, cols: 3, start: 0, end: 2 },
        { name: 'the last row', rows: 5, cols: 7, start: 4, end: 5 },
        { name: 'past a workgroup', rows: 40, cols: 3, start: 3, end: 37 },
      ].map((c) => ({ dtype, ...c })),
    ),
  )('sliceRows $dtype, $name: [$rows, $cols][$start:$end]', async (c, ctx) => {
    if (c.dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    const x = count(c.rows * c.cols);
    const xv = upload(root, x, { ...matrix(c.rows, c.cols), dtype: c.dtype });
    const got = await toArray(slice(xv, 0, c.start, c.end), ex);
    expectClose(got, x.subarray(c.start * c.cols, c.end * c.cols), EXACT);
  });

  // concatRows: cat along dim 0 of 2D tensors is a flat copy of a then b.
  it.for(
    (['f32', 'f16'] as const).flatMap((dtype) =>
      [
        { name: '1×1 and 1×1', aRows: 1, bRows: 1, cols: 1 },
        { name: 'odd sizes', aRows: 3, bRows: 2, cols: 5 },
        { name: 'past a workgroup', aRows: 1, bRows: 30, cols: 3 },
      ].map((c) => ({ dtype, ...c })),
    ),
  )('concatRows $dtype, $name: [$aRows, $cols] ++ [$bRows, $cols]', async (c, ctx) => {
    if (c.dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    const a = count(c.aRows * c.cols);
    const b = count(c.bRows * c.cols, 500);
    const meta = (rows: number) => ({ ...matrix(rows, c.cols), dtype: c.dtype });
    const got = await toArray(
      cat([upload(root, a, meta(c.aRows)), upload(root, b, meta(c.bRows))], 0),
      ex,
    );
    expectClose(got, [...a, ...b], EXACT);
  });

  // reshape: a flat copy into the new dims.
  it.for(['f32', 'f16'] as const)('reshape %s [2, 6] → [3, 4]', async (dtype, ctx) => {
    if (dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    const x = count(12);
    const y = reshape(upload(root, x, { ...matrix(2, 6), dtype }), [3, 4]);
    expect(y.shape).toEqual({ elems: 12, dtype, dims: [3, 4] });
    expectClose(await toArray(y, ex), x, EXACT);
  });

  // sliceChannels: a channel slice of [C, H, W] is a flat copy from start·H·W.
  it('sliceChannels [4, 2, 3][1:3]', async () => {
    const x = count(24);
    const got = await toArray(slice(upload(root, x, tensor3d(4, 2, 3)), 0, 1, 3), ex);
    expectClose(got, x.subarray(6, 18), EXACT);
  });

  // concatChannels: cat of [C, H, W] along dim 0. Inputs that are uploads
  // are copied; an input produced by reshape, sliceChannels or another
  // concatChannels, with no other consumer, writes straight into the concat's
  // buffer at its offset (concat elision in graph/evalCore.ts).
  it('concatChannels of two uploads, copied', async () => {
    const a = count(12);
    const b = count(6, 100);
    const got = await toArray(
      cat([upload(root, a, tensor3d(2, 2, 3)), upload(root, b, tensor3d(1, 2, 3))], 0),
      ex,
    );
    expectClose(got, [...a, ...b], EXACT);
  });

  it.for(['f32', 'f16'] as const)(
    'concatChannels %s with every input written in place',
    async (dtype, ctx) => {
      if (dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
      const a = count(12);
      const b = count(24, 100);
      const meta = (c: number) => ({ ...tensor3d(c, 2, 3), dtype });
      const av = reshape(upload(root, a, { ...matrix(2, 6), dtype }), [2, 2, 3]);
      const bv = slice(upload(root, b, meta(4)), 0, 1, 3);
      // Three inputs fold into two concats; the inner one lands in the outer's buffer.
      const cv = reshape(upload(root, count(6, 200), { ...matrix(1, 6), dtype }), [1, 2, 3]);
      const got = await toArray(cat([av, bv, cv], 0), ex);
      expectClose(got, [...a, ...b.subarray(6, 18), ...count(6, 200)], EXACT);
    },
  );

  it('concatChannels with one input written in place and one copied', async () => {
    const a = count(12);
    const b = count(6, 100);
    const av = reshape(upload(root, a, [2, 6]), [2, 2, 3]);
    const got = await toArray(cat([av, upload(root, b, tensor3d(1, 2, 3))], 0), ex);
    expectClose(got, [...a, ...b], EXACT);
  });

  // The inner concat lands in the outer's buffer at the outer's offset, so
  // its copied input goes past both written ones.
  it('concatChannels with a copied input inside a nested concat', async () => {
    const a = count(6);
    const b = count(12, 100);
    const c = count(6, 200);
    const av = reshape(upload(root, a, [1, 6]), [1, 2, 3]);
    const bv = reshape(upload(root, b, [2, 6]), [2, 2, 3]);
    const got = await toArray(cat([av, cat([bv, upload(root, c, tensor3d(1, 2, 3))], 0)], 0), ex);
    expectClose(got, [...a, ...b, ...c], EXACT);
  });

  // An input that is also a target keeps its own buffer, so it is copied.
  // It goes second: at offset 0 an elided target would still read back right.
  it('concatChannels with an input that is also a target', async () => {
    const a = count(12);
    const b = count(6, 100);
    const av = reshape(upload(root, a, [2, 6]), [2, 2, 3]);
    const bv = reshape(upload(root, b, [1, 6]), [1, 2, 3]);
    const both = cat([bv, av], 0);
    evalValues([av, both], ex);
    expectClose(await toArray(av, ex), a, EXACT);
    expectClose(await toArray(both, ex), [...b, ...a], EXACT);
  });

  // writeRows: copies src into rows of a caller-owned dst and returns a view
  // of that buffer up to the last written row.
  it.for(['f32', 'f16'] as const)('writeRows %s into the middle of dst', async (dtype, ctx) => {
    if (dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    const meta = (rows: number) => ({ ...matrix(rows, 3), dtype });
    const dst = upload(root, count(18), meta(6));
    const view = writeRows(dst, upload(root, count(6, 100), meta(2)), 2);
    expect(view.shape).toEqual(meta(4));
    expectClose(await toArray(view, ex), [...count(6), ...count(6, 100)], EXACT);
    // Rows past the view keep what they held.
    expectClose(await toArray(dst, ex), [...count(6), ...count(6, 100), ...count(6, 12)], EXACT);
  });

  // A KV cache append: each step writes the next rows of the same dst.
  it('writeRows appends twice to the same dst', async () => {
    const dst = upload(root, new Float32Array(12), [4, 3]);
    await toArray(writeRows(dst, upload(root, count(3, 1), [1, 3]), 0), ex);
    const second = writeRows(dst, upload(root, count(6, 10), [2, 3]), 1);
    expectClose(await toArray(second, ex), [...count(3, 1), ...count(6, 10)], EXACT);
  });

  // gatherRows takes its ids as a list, gatherRowsFrom from a GPU buffer.
  it.for<GatherCase>([
    { name: 'gatherRows, a few rows', table: 16, cols: 8, count: 5, op: 'gatherRows' },
    { name: 'gatherRows, 3 columns', table: 16, cols: 3, count: 9, op: 'gatherRows' },
    { name: 'gatherRows f16', table: 16, cols: 6, count: 11, op: 'gatherRows', dtype: 'f16' },
    { name: 'gatherRowsFrom, a few rows', table: 16, cols: 8, count: 5, op: 'gatherRowsFrom' },
    { name: 'gatherRowsFrom, one column', table: 16, cols: 1, count: 33, op: 'gatherRowsFrom' },
    {
      name: 'gatherRowsFrom f16',
      table: 16,
      cols: 6,
      count: 11,
      op: 'gatherRowsFrom',
      dtype: 'f16',
    },
    // Past 2^24 output values an f32 quotient rounds the index and picks the
    // wrong row; the kernels divide in integers.
    {
      name: 'gatherRowsFrom, output past 2^24 values',
      table: 64,
      cols: 8,
      count: 2 ** 21 + 16,
      op: 'gatherRowsFrom',
    },
    {
      name: 'gatherRows, output past 2^24 values',
      table: 64,
      cols: 8,
      count: 2 ** 21 + 16,
      op: 'gatherRows',
    },
  ])('$name', async (c, ctx) => {
    if (c.dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    await checkGather(c);
  });

  // The flat copies never split the index, so they hold past 2^24 elements,
  // in the second grid dimension.
  it('sliceRows, concatRows, reshape and writeRows past 2^24 elements', async () => {
    const rows = 4100;
    const cols = 4096;
    expect(rows * cols).toBeGreaterThan(2 ** 24);
    expect(rows * cols).toBeGreaterThan(GRID_Y_STRIDE);
    const x = Float32Array.from({ length: rows * cols }, (_, i) => i % 8191);
    const xv = upload(root, x, [rows, cols]);
    const tail = slice(xv, 0, 1, rows);
    expectClose(await toArray(tail, ex), x.subarray(cols), EXACT);
    expectClose(await toArray(cat([slice(xv, 0, 0, 1), tail], 0), ex), x, EXACT);
    expectClose(await toArray(reshape(xv, [cols, rows]), ex), x, EXACT);
    const dst = upload(root, new Float32Array(rows * cols), [rows, cols]);
    expectClose(await toArray(writeRows(dst, xv, 0), ex), x, EXACT);
  });

  it('rejects an empty input with an INVALID_ARGUMENT', () => {
    expect(() => upload(root, new Float32Array(0), [0, 4])).toThrow(
      expect.objectContaining({ code: 'INVALID_ARGUMENT' }),
    );
  });

  it.each([
    { name: 'an empty range', start: 2, end: 2 },
    { name: 'a range past the last row', start: 1, end: 6 },
  ])('slice refuses $name', ({ start, end }) => {
    const x = upload(root, new Float32Array(10), [5, 2]);
    expect(() => slice(x, 0, start, end)).toThrow(`slice: bad range [${start}, ${end}) for 5 rows`);
  });

  it('cat refuses rows of different widths', () => {
    const a = upload(root, new Float32Array(6), [2, 3]);
    const b = upload(root, new Float32Array(8), [2, 4]);
    expect(() => cat([a, b], 0)).toThrow(/col mismatch — a has 3, b has 4/);
  });

  it('cat refuses channels of different spatial sizes', () => {
    const a = upload(root, new Float32Array(12), tensor3d(2, 2, 3));
    const b = upload(root, new Float32Array(12), tensor3d(2, 3, 2));
    expect(() => cat([a, b], 0)).toThrow(/spatial mismatch/);
  });

  it('reshape refuses another element count', () => {
    const x = upload(root, new Float32Array(12), [3, 4]);
    expect(() => reshape(x, [5, 2])).toThrow(/number of elements mismatch/);
  });

  it('writeRows refuses rows past the capacity of dst', () => {
    const dst = upload(root, new Float32Array(12), [4, 3]);
    const src = upload(root, new Float32Array(6), [2, 3]);
    expect(() => writeRows(dst, src, 3)).toThrow(/rows \[3, 5\) exceed dst capacity 4/);
  });

  it('writeRows refuses a pending dst', () => {
    const x = upload(root, new Float32Array(12), [4, 3]);
    const src = upload(root, new Float32Array(3), [1, 3]);
    expect(() => writeRows(slice(x, 0, 0, 2), src, 0)).toThrow(/dst must be a materialized/);
  });

  it.each([
    { name: 'an empty index list', idx: [], err: /empty index list/ },
    { name: 'an index past the last row', idx: [0, 5], err: /index 5 out of range for 5 rows/ },
    { name: 'a fractional index', idx: [1.5], err: /index 1.5 out of range for 5 rows/ },
  ])('gatherRows refuses $name', ({ idx, err }) => {
    const x = upload(root, new Float32Array(10), [5, 2]);
    expect(() => gatherRows(x, idx)).toThrow(err);
  });

  it('gatherRowsFrom refuses f16 indices', (ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const x = upload(root, new Float32Array(10), [5, 2]);
    const idx = upload(root, new Float32Array(2), { elems: 2, dtype: 'f16' });
    expect(() => gatherRowsFrom(x, idx)).toThrow(/idx must be f32, got f16/);
  });
});
