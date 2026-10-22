import { describe, expect, it } from 'vitest';
import { toArray } from '../../../../src/core/gpu/eval.ts';
import { matmulGather } from '../../../../src/core/graph/ops.ts';
import { matrix } from '../../../../src/core/graph/value.ts';
import { GRID_Y_STRIDE } from '../../../../src/core/kernels/common.ts';
import { expectClose, gpu, randomF32, requireFeature, roundF16, upload } from '../../../gpu.ts';

const { root, ex } = await gpu();

/** out[i] = a[i] · W[e_i] + bias[e_i], where W[e] is rows [e·K, (e+1)·K) of
 *  w [E·K, N], in f64. */
function matmulGatherRef(
  a: Float32Array,
  w: Float32Array,
  bias: Float32Array,
  experts: readonly number[],
  k: number,
  n: number,
): Float64Array {
  const m = experts.length;
  const out = new Float64Array(m * n);
  for (let i = 0; i < m; i++) {
    const e = experts[i]!;
    for (let c = 0; c < n; c++) {
      let sum = bias[e * n + c]!;
      for (let j = 0; j < k; j++) sum += a[i * k + j]! * w[(e * k + j) * n + c]!;
      out[i * n + c] = sum;
    }
  }
  return out;
}

interface Case {
  k: number;
  n: number;
  numExperts: number;
  experts: readonly number[];
}

async function checkGather({ k, n, numExperts, experts }: Case, dtype: 'f32' | 'f16') {
  const m = experts.length;
  const round = dtype === 'f16' ? roundF16 : (v: Float32Array) => v;
  const a = round(randomF32(m * k, 1));
  const w = round(randomF32(numExperts * k * n, 2));
  const bias = round(randomF32(numExperts * n, 3));
  const meta = (rows: number, cols: number) => ({ ...matrix(rows, cols), dtype });
  const got = await toArray(
    matmulGather(
      upload(root, a, meta(m, k)),
      upload(root, w, meta(numExperts * k, n)),
      upload(root, new Float32Array(experts), { elems: m, dtype: 'f32' }),
      { bias: upload(root, bias, meta(numExperts, n)) },
    ),
    ex,
  );
  // f32 rounding grows with K; f16 output adds 2^-11 relative.
  const tol = dtype === 'f16' ? { atol: 2e-3, rtol: 2e-3 } : { atol: 1e-6 * k, rtol: 1e-5 };
  expectClose(got, matmulGatherRef(a, w, bias, experts, k, n), tol);
}

describe('matmulGather kernel', () => {
  // One route: a thread per (row, 4-column unit), N % 4 == 0.
  it.for(
    (['f32', 'f16'] as const).flatMap((dtype) =>
      [
        { name: 'one row, one expert, one quad', c: { k: 1, n: 4, numExperts: 1, experts: [0] } },
        { name: 'odd K', c: { k: 7, n: 8, numExperts: 3, experts: [2, 0, 1] } },
        {
          name: 'the same expert on every row',
          c: { k: 16, n: 12, numExperts: 4, experts: [3, 3, 3, 3, 3] },
        },
        { name: 'the last expert', c: { k: 9, n: 4, numExperts: 5, experts: [4] } },
        {
          name: 'rows past a workgroup of units',
          c: {
            k: 33,
            n: 68,
            numExperts: 4,
            experts: Array.from({ length: 70 }, (_, i) => (i * 7) % 4),
          },
        },
        // Past GRID_Y_STRIDE units the dispatch goes 2-D.
        {
          name: 'units past GRID_Y_STRIDE',
          c: {
            k: 4,
            n: 8,
            numExperts: 3,
            experts: Array.from({ length: GRID_Y_STRIDE / 2 + 5 }, (_, i) => i % 3),
          },
        },
      ].map((c) => ({ dtype, ...c })),
    ),
  )('$dtype $name', async ({ dtype, c }, ctx) => {
    if (dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    await checkGather(c, dtype);
  });

  // The row must come from u32 division: an f32 idx / n4 rounds up to the
  // next row past ~2^23 units, so the end of a row is never written.
  it('f16 rows past 2^23 units', async (ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const [m, n] = [2100, 16380];
    // out[i, c] = a[i]: K 1, every weight 1, bias 0. f16 output keeps the
    // buffer under the default maxStorageBufferBindingSize.
    const a = Float32Array.from({ length: m }, (_, i) => (i % 7) + 1);
    const meta = (rows: number, cols: number) => ({ ...matrix(rows, cols), dtype: 'f16' as const });
    const got = await toArray(
      matmulGather(
        upload(root, a, meta(m, 1)),
        upload(root, new Float32Array(n).fill(1), meta(1, n)),
        upload(root, new Float32Array(m), { elems: m, dtype: 'f32' }),
        { bias: upload(root, new Float32Array(n), meta(1, n)) },
      ),
      ex,
    );
    const firstWrong = got.findIndex((v, i) => v !== a[Math.floor(i / n)]);
    expect(firstWrong).toBe(-1);
  });

  // Known bug: the op checks bias's dtype but not its size, so a bias of N
  // elements for 4 experts passes and the kernel reads past its end for
  // experts 1 to 3. It should refuse anything but [E, N]. Turn into `it`
  // with the fix.
  it.fails('refuses a bias that is not [E, N]', () => {
    const a = upload(root, new Float32Array(8), [2, 4]);
    const w = upload(root, new Float32Array(4 * 4 * 8), [16, 8]);
    const idx = upload(root, new Float32Array([1, 3]), { elems: 2, dtype: 'f32' });
    const bias = upload(root, new Float32Array(8), [1, 8]);
    expect(() => matmulGather(a, w, idx, { bias })).toThrow();
  });

  it('rejects an empty input with an INVALID_ARGUMENT', () => {
    expect(() => upload(root, new Float32Array(0), [0, 4])).toThrow(
      expect.objectContaining({ code: 'INVALID_ARGUMENT' }),
    );
  });

  it.each([
    { name: 'N not a multiple of 4', wRows: 8, n: 6, idx: 2, err: /n 6 must be divisible by 4/ },
    {
      name: 'w rows not a multiple of K',
      wRows: 10,
      n: 8,
      idx: 2,
      err: /w rows 10 not a multiple of K 4/,
    },
    {
      name: 'an index per row missing',
      wRows: 8,
      n: 8,
      idx: 1,
      err: /expertIdx must be f32 with 2 elems, got 1/,
    },
  ])('refuses $name', ({ wRows, n, idx, err }) => {
    const a = upload(root, new Float32Array(8), [2, 4]);
    const w = upload(root, new Float32Array(wRows * n), [wRows, n]);
    const e = upload(root, new Float32Array(idx), { elems: idx, dtype: 'f32' });
    const bias = upload(root, new Float32Array(2 * n), [2, n]);
    expect(() => matmulGather(a, w, e, { bias })).toThrow(err);
  });

  it('refuses w in another dtype than a', (ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const a = upload(root, new Float32Array(8), [2, 4]);
    const w = upload(root, new Float32Array(32), { ...matrix(4, 8), dtype: 'f16' });
    const e = upload(root, new Float32Array(2), { elems: 2, dtype: 'f32' });
    const bias = upload(root, new Float32Array(8), [1, 8]);
    expect(() => matmulGather(a, w, e, { bias })).toThrow(/w is f16 but a is f32/);
  });
});
