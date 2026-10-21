import { describe, expect, it } from 'vitest';
import { toArray } from '../../../../src/core/gpu/eval.ts';
import { topk } from '../../../../src/core/graph/ops.ts';
import { matrix } from '../../../../src/core/graph/value.ts';
import { GRID_Y_STRIDE } from '../../../../src/core/kernels/common.ts';
import { expectClose, gpu, randomF32, requireFeature, roundF16, upload } from '../../../gpu.ts';

const { root, ex } = await gpu();

const K = 4;

/** torch.topk(x, 4) per row, lowest index first on ties, then the softmax of
 *  the four picked logits. Packed per row as [ids, weights], in f64. */
function topkRef(x: Float32Array, rows: number, cols: number): Float64Array {
  const out = new Float64Array(rows * 2 * K);
  for (let r = 0; r < rows; r++) {
    const row = x.subarray(r * cols, (r + 1) * cols);
    const ids = Array.from(row, (_, j) => j)
      .sort((a, b) => row[b]! - row[a]! || a - b)
      .slice(0, K);
    const max = row[ids[0]!]!;
    const exps = ids.map((j) => Math.exp(row[j]! - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    for (let i = 0; i < K; i++) {
      out[r * 2 * K + i] = ids[i]!;
      out[r * 2 * K + K + i] = exps[i]! / sum;
    }
  }
  return out;
}

describe('topkSelect kernel', () => {
  // One route: a thread per row. f16 logits are read as f16, the output is
  // always f32.
  it.for(
    (['f32', 'f16'] as const).flatMap((dtype) =>
      [
        { name: 'exactly k columns', rows: 3, cols: 4 },
        { name: 'odd sizes', rows: 7, cols: 9 },
        { name: 'one row past a workgroup', rows: 65, cols: 32 },
        { name: 'many experts', rows: 5, cols: 1000 },
      ].map((shape) => ({ dtype, ...shape })),
    ),
  )('$dtype $name: [$rows, $cols]', async ({ dtype, rows, cols }, ctx) => {
    if (dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    // Scaled so the four weights differ; f16 rounds before the reference
    // sorts, so ties come out the same.
    const scaled = randomF32(rows * cols, 1).map((v) => 4 * v);
    const x = dtype === 'f16' ? roundF16(scaled) : scaled;
    const got = await toArray(topk(upload(root, x, { ...matrix(rows, cols), dtype }), K), ex);
    expectClose(got, topkRef(x, rows, cols), { atol: 1e-7, rtol: 1e-5 });
  });

  // Ties go to the lowest index, as in torch.topk: [5, 7, 7, 5, 7, 1] keeps
  // ids [1, 2, 4, 0], and a row of equal logits keeps [0, 1, 2, 3] with 1/4 each.
  it('breaks ties toward the lowest index', async () => {
    const x = new Float32Array([5, 7, 7, 5, 7, 1, 2, 2, 2, 2, 2, 2]);
    const got = await toArray(topk(upload(root, x, [2, 6]), K), ex);
    expectClose(got, topkRef(x, 2, 6), { atol: 1e-7, rtol: 1e-6 });
    expect(Array.from(got.subarray(0, K))).toEqual([1, 2, 4, 0]);
    expect(Array.from(got.subarray(2 * K, 3 * K))).toEqual([0, 1, 2, 3]);
  });

  // Known bug: a logit at or below the kernel's -3.4e38 start value never
  // enters the top four, so with fewer than four finite logits the empty
  // slots keep id 0. torch.topk([-inf, -inf, -inf, -inf, 1], 4) gives ids
  // [4, 0, 1, 2]; the kernel gives [4, 0, 0, 0]. The weights match (0).
  it.fails('keeps distinct ids when fewer than four logits are finite', async () => {
    const x = new Float32Array([-Infinity, -Infinity, -Infinity, -Infinity, 1]);
    const got = await toArray(topk(upload(root, x, [1, 5]), K), ex);
    expect(Array.from(got)).toEqual([4, 0, 1, 2, 1, 0, 0, 0]);
  });

  // Past GRID_Y_STRIDE rows the threads take the second grid dimension. The
  // [M, 8] output is then just under 128 MB.
  it('runs past GRID_Y_STRIDE rows', async (ctx) => {
    const rows = GRID_Y_STRIDE + 5;
    if (rows * 2 * K * 4 > root.device.limits.maxStorageBufferBindingSize) {
      ctx.skip('the output does not fit maxStorageBufferBindingSize');
    }
    const x = Float32Array.from({ length: rows * K }, (_, i) => ((i * 7) % 13) - (i % K));
    const got = await toArray(topk(upload(root, x, [rows, K]), K), ex);
    expectClose(got, topkRef(x, rows, K), { atol: 1e-7, rtol: 1e-5 });
  });

  it('rejects an empty input with an INVALID_ARGUMENT', () => {
    expect(() => upload(root, new Float32Array(0), [0, 4])).toThrow(
      expect.objectContaining({ code: 'INVALID_ARGUMENT' }),
    );
  });

  it.each([
    { name: 'k other than 4', cols: 8, k: 2, err: /only k=4 is supported, got 2/ },
    { name: 'fewer columns than k', cols: 3, k: 4, err: /need at least 4 columns, got 3/ },
  ])('refuses $name', ({ cols, k, err }) => {
    const x = upload(root, new Float32Array(2 * cols), [2, cols]);
    expect(() => topk(x, k)).toThrow(err);
  });
});
