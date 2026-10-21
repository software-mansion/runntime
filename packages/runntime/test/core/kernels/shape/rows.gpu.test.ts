import { describe, it } from 'vitest';
import { toArray } from '../../../../src/core/gpu/eval.ts';
import { gatherRows, gatherRowsFrom } from '../../../../src/core/graph/ops.ts';
import { matrix } from '../../../../src/core/graph/value.ts';
import { expectClose, gpu, requireFeature, upload } from '../../../gpu.ts';

const { root, ex } = await gpu();

interface Case {
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
async function check({ table, cols, count, op, dtype = 'f32' }: Case): Promise<void> {
  const x = Float32Array.from({ length: table * cols }, (_, i) => i);
  const idx = Float32Array.from({ length: count }, (_, i) => (i * 7) % table);
  const xs = upload(root, x, { ...matrix(table, cols), dtype });
  const y =
    op === 'gatherRows'
      ? gatherRows(xs, Array.from(idx))
      : gatherRowsFrom(xs, upload(root, idx, matrix(count, 1)));
  const want = new Float32Array(count * cols);
  for (let i = 0; i < count; i++) {
    for (let c = 0; c < cols; c++) want[i * cols + c] = x[idx[i]! * cols + c]!;
  }
  expectClose(await toArray(y, ex), want, { atol: 0, rtol: 0 });
}

describe('gather kernels', () => {
  // gatherRows takes its ids as a list, gatherRowsFrom from a GPU buffer.
  // it.for, not it.each: only it.for passes the test context the skip needs.
  it.for<Case>([
    { name: 'gatherRows, a few rows', table: 16, cols: 8, count: 5, op: 'gatherRows' },
    { name: 'gatherRows, 3 columns', table: 16, cols: 3, count: 9, op: 'gatherRows' },
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
    await check(c);
  });
});
