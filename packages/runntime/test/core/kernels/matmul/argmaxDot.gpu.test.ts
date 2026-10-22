import { describe, expect, it } from 'vitest';
import { evalValues, toArray } from '../../../../src/core/gpu/eval.ts';
import { argmaxDot } from '../../../../src/core/graph/ops.ts';
import { matrix } from '../../../../src/core/graph/value.ts';
import { ARGMAX_SPLIT } from '../../../../src/core/kernels/matmul/argmaxDot.ts';
import { gpu, randomF32, requireFeature, roundF16, upload } from '../../../gpu.ts';

const { root, ex } = await gpu();

/** Rows a workgroup scans per step. */
const ROWS_PER_STEP = 32;

/** Random w [rows, cols] and x [1, cols] with row `winner` set to sign(x), so
 *  its dot, Σ|x|, beats every other row by a margin no rounding can close. */
function planted(rows: number, cols: number, winner: number, dtype: 'f32' | 'f16') {
  const round = dtype === 'f16' ? roundF16 : (v: Float32Array) => v;
  const x = round(randomF32(cols, 1).map((v) => v + (v >= 0 ? 0.5 : -0.5)));
  // Scaled down so a random row's dot stays well under Σ|x| ≥ cols / 2.
  const w = round(randomF32(rows * cols, 2).map((v) => 0.1 * v));
  for (let j = 0; j < cols; j++) w[winner * cols + j] = Math.sign(x[j]!);
  return { w, x };
}

async function checkArgmax(rows: number, cols: number, winner: number, dtype: 'f32' | 'f16') {
  const { w, x } = planted(rows, cols, winner, dtype);
  const meta = (r: number) => ({ ...matrix(r, cols), dtype });
  const got = await toArray(argmaxDot(upload(root, w, meta(rows)), upload(root, x, meta(1))), ex);
  expect(Array.from(got)).toEqual([winner]);
}

describe('argmaxDot kernel', () => {
  // One route: ARGMAX_SPLIT workgroups stride the rows ROWS_PER_STEP at a
  // time, 8 threads share a row's columns, and a second dispatch reduces the
  // partials.
  const STEP = ROWS_PER_STEP * ARGMAX_SPLIT;
  it.for(
    (['f32', 'f16'] as const).flatMap((dtype) =>
      [
        { name: 'a single row', rows: 1, cols: 7, winner: 0 },
        { name: 'one column', rows: 50, cols: 1, winner: 17 },
        { name: 'fewer columns than threads per row', rows: 40, cols: 5, winner: 39 },
        { name: 'one row past a step', rows: ROWS_PER_STEP + 1, cols: 9, winner: ROWS_PER_STEP },
        { name: 'the last row of the first pass', rows: STEP, cols: 16, winner: STEP - 1 },
        { name: 'one row past the first pass', rows: STEP + 1, cols: 16, winner: STEP },
        { name: 'a vocabulary', rows: 50257, cols: 64, winner: 31337 },
        { name: 'the first row of a vocabulary', rows: 50257, cols: 64, winner: 0 },
      ].map((c) => ({ dtype, ...c })),
    ),
  )('$dtype $name: [$rows, $cols], row $winner wins', async (c, ctx) => {
    if (c.dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    await checkArgmax(c.rows, c.cols, c.winner, c.dtype);
  });

  // Ties go to the lowest row, as torch.argmax does: within a step, across
  // steps of one workgroup, and across workgroups. The last two put the lower
  // row after the higher one in the reduce order, so only the index compare
  // can pick it: row STEP sits in thread row 0 and workgroup 0, row 1 in
  // thread row 1, row 40 in workgroup 1.
  it.each([
    { name: 'in one step', rows: 20, tied: [3, 7] },
    { name: 'across steps of one workgroup', rows: 3 * STEP, tied: [2 * STEP + 5, 5] },
    { name: 'across workgroups', rows: STEP, tied: [STEP - 1, 40] },
    { name: 'with the lower row reduced later in a workgroup', rows: STEP + 1, tied: [STEP, 1] },
    {
      name: 'with the lower row reduced later across workgroups',
      rows: STEP + 1,
      tied: [STEP, 40],
    },
    { name: 'every row', rows: 100, tied: Array.from({ length: 100 }, (_, i) => 99 - i) },
  ])('breaks a tie $name toward the lowest row', async ({ rows, tied }) => {
    const cols = 8;
    const w = new Float32Array(rows * cols);
    const x = new Float32Array(cols).fill(1);
    for (const r of tied) w.fill(1, r * cols, (r + 1) * cols);
    const got = await toArray(
      argmaxDot(upload(root, w, [rows, cols]), upload(root, x, [1, cols])),
      ex,
    );
    expect(Array.from(got)).toEqual([Math.min(...tied)]);
  });

  // The partials scratch is shared, so two calls in one submit must each
  // reduce their own.
  it('two calls in one submit', async () => {
    const a = planted(300, 8, 123, 'f32');
    const b = planted(300, 8, 7, 'f32');
    const call = ({ w, x }: { w: Float32Array; x: Float32Array }) =>
      argmaxDot(upload(root, w, [300, 8]), upload(root, x, [1, 8]));
    const [ia, ib] = [call(a), call(b)];
    evalValues([ia, ib], ex);
    expect(Array.from(await toArray(ia, ex))).toEqual([123]);
    expect(Array.from(await toArray(ib, ex))).toEqual([7]);
  });

  it('rejects an empty input with an INVALID_ARGUMENT', () => {
    expect(() => upload(root, new Float32Array(0), [0, 4])).toThrow(
      expect.objectContaining({ code: 'INVALID_ARGUMENT' }),
    );
  });

  it.each([
    { name: 'x of more than one row', xRows: 2, xCols: 4, err: /x must be a single row, got 2/ },
    { name: 'x of another width', xRows: 1, xCols: 5, err: /x cols 5 != w cols 4/ },
  ])('refuses $name', ({ xRows, xCols, err }) => {
    const w = upload(root, new Float32Array(12), [3, 4]);
    const x = upload(root, new Float32Array(xRows * xCols), [xRows, xCols]);
    expect(() => argmaxDot(w, x)).toThrow(err);
  });

  it('refuses x in another dtype than w', (ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const w = upload(root, new Float32Array(12), [3, 4]);
    const x = upload(root, new Float32Array(4), { ...matrix(1, 4), dtype: 'f16' });
    expect(() => argmaxDot(w, x)).toThrow(/w dtype f32 must match x's f16/);
  });
});
