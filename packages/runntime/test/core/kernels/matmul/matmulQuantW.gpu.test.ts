import { describe, expect, it } from 'vitest';
import { toArray } from '../../../../src/core/gpu/eval.ts';
import { matmul, type MatmulOpts } from '../../../../src/core/graph/ops.ts';
import { matrix } from '../../../../src/core/graph/value.ts';
import { GRID_Y_STRIDE } from '../../../../src/core/kernels/common.ts';
import { expectClose, gpu, randomF32, requireFeature, upload } from '../../../gpu.ts';
import { randomQuantWeight } from './quantW.ts';

const { root, ex } = await gpu();

interface Case {
  m: number;
  k: number;
  n: number;
  bits: 4 | 8;
  groupSize: number;
  experts?: number;
  /** The expert the call reads, through baseRow and scaleBase. */
  expert?: number;
}

async function checkQuantMatmul({ m, k, n, bits, groupSize, experts = 1, expert = 0 }: Case) {
  const weight = randomQuantWeight(root, { experts, k, n, bits, groupSize }, 2);
  const a = randomF32(m * k, 1);
  const got = await toArray(
    matmul(upload(root, a, [m, k]), weight.w, {
      scales: upload(root, weight.scales, [weight.scales.length / n, n]),
      bits,
      groupSize,
      baseRow: expert * k,
      scaleBase: ((expert * k) / groupSize) * n,
    }),
    ex,
  );
  // a · W[expert] from the dequantized weights, in f64.
  const want = new Float64Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let c = 0; c < n; c++) {
      let sum = 0;
      for (let j = 0; j < k; j++) sum += a[i * k + j]! * weight.dequant[(expert * k + j) * n + c]!;
      want[i * n + c] = sum;
    }
  }
  expectClose(got, want, { atol: 1e-6 * k, rtol: 1e-5 });
}

describe('matmulQuantW kernel', () => {
  // Routes are picked in gpu/dispatch/matmul.ts. This kernel takes an f32
  // input with no epilogue on one expert, below 8 rows, or when K is not a
  // multiple of 8. Other calls on a whole weight take matmulQuantTiled. K and
  // groupSize are multiples of 4, N of 4 for int8 and of 8 for int4.
  it.for(
    ([8, 4] as const).flatMap((bits) =>
      [
        // plain: m < 8
        { name: 'one tile', m: 1, k: 4, n: 8, groupSize: 4 },
        { name: 'one group per column', m: 3, k: 12, n: 8, groupSize: 12 },
        { name: 'a group every 4 rows', m: 3, k: 12, n: 16, groupSize: 4 },
        { name: 'groups of 8', m: 5, k: 64, n: 24, groupSize: 8 },
        { name: 'a wide row', m: 2, k: 640, n: 640, groupSize: 32 },
        { name: 'one row short of the tiled route', m: 7, k: 32, n: 8, groupSize: 16 },
        // tiled: a whole weight, m >= 8 and K % 8 == 0
        { name: 'the first m on the tiled route', m: 8, k: 32, n: 8, groupSize: 16 },
        // plain: one expert of a fused weight, whatever m
        {
          name: 'rows past a workgroup of units on one expert',
          m: 70,
          k: 32,
          n: 8,
          groupSize: 16,
          experts: 2,
          expert: 1,
        },
        // plain: K % 8 != 0. Past GRID_Y_STRIDE units the dispatch goes 2-D.
        { name: 'units past GRID_Y_STRIDE', m: GRID_Y_STRIDE / 2 + 5, k: 4, n: 8, groupSize: 4 },
      ].map((c) => ({ bits, ...c })),
    ),
  )('int$bits $name: [$m, $k] × [$k, $n], groupSize $groupSize', (c) => checkQuantMatmul(c));

  // plain: baseRow and scaleBase read one expert of a fused [E·K, N] weight.
  it.for(([8, 4] as const).flatMap((bits) => [0, 1, 2].map((expert) => ({ bits, expert }))))(
    'int$bits expert $expert of 3',
    ({ bits, expert }) =>
      checkQuantMatmul({ m: 3, k: 16, n: 8, bits, groupSize: 8, experts: 3, expert }),
  );

  it('rejects an empty input with an INVALID_ARGUMENT', () => {
    expect(() => upload(root, new Float32Array(0), [0, 4])).toThrow(
      expect.objectContaining({ code: 'INVALID_ARGUMENT' }),
    );
  });

  const fused = { experts: 2, k: 8, n: 8, bits: 8 as const, groupSize: 4 };
  it.each([
    {
      name: 'a bias on one expert',
      opts: { bias: true },
      err: /an epilogue on a quantW weight needs the whole weight/,
    },
    {
      name: 'no scales',
      opts: { scales: false },
      err: /quantW weights need scales, bits and groupSize/,
    },
    {
      name: 'a groupSize that does not divide K',
      opts: { groupSize: 3 },
      err: /groupSize 3 must divide K 8/,
    },
    {
      name: 'a baseRow inside an expert',
      opts: { baseRow: 4 },
      err: /baseRow 4 must be K-aligned \(8\)/,
    },
    { name: 'a baseRow past the last expert', opts: { baseRow: 16 }, err: /bad baseRow 16/ },
    { name: 'bits other than 4 or 8', opts: { bits: 2 }, err: /bits must be 8 or 4, got 2/ },
  ])('refuses $name', ({ opts, err }) => {
    const weight = randomQuantWeight(root, fused, 2);
    const a = upload(root, new Float32Array(2 * 8), [2, 8]);
    const scales = upload(root, weight.scales, [4, 8]);
    const call: MatmulOpts = {
      scales: opts.scales === false ? undefined : scales,
      bits: (opts.bits ?? 8) as 8 | 4,
      groupSize: opts.groupSize ?? 4,
      baseRow: opts.baseRow,
      bias: opts.bias ? upload(root, new Float32Array(8), [1, 8]) : undefined,
    };
    expect(() => matmul(a, weight.w, call)).toThrow(err);
  });

  // Known bug: the op checks scales' dtype and that scaleBase is >= 0, but not
  // that scaleBase + (K / groupSize) · N fits in scales, so the kernel reads
  // past its end. Turn into `it` with the fix.
  it.fails.each([
    { name: 'scales shorter than one expert', rows: 1, scaleBase: 0 },
    { name: 'a scaleBase past the end of scales', rows: 4, scaleBase: 1000 },
  ])('refuses $name', ({ rows, scaleBase }) => {
    const weight = randomQuantWeight(root, fused, 2);
    const a = upload(root, new Float32Array(2 * 8), [2, 8]);
    const scales = upload(root, weight.scales.subarray(0, rows * 8), [rows, 8]);
    expect(() => matmul(a, weight.w, { scales, scaleBase, bits: 8, groupSize: 4 })).toThrow();
  });

  it('refuses int4 columns that are not a multiple of 8', () => {
    const weight = randomQuantWeight(root, { experts: 1, k: 8, n: 4, bits: 8, groupSize: 4 }, 2);
    const a = upload(root, new Float32Array(8), [1, 8]);
    const scales = upload(root, weight.scales, [2, 4]);
    expect(() => matmul(a, weight.w, { scales, bits: 4, groupSize: 4 })).toThrow(
      /cols 4 must be divisible by 8 for int4 packing/,
    );
  });

  it('refuses an f16 input on one expert', (ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const weight = randomQuantWeight(root, fused, 2);
    const a = upload(root, new Float32Array(16), { ...matrix(2, 8), dtype: 'f16' });
    const scales = upload(root, weight.scales, [4, 8]);
    expect(() => matmul(a, weight.w, { scales, bits: 8, groupSize: 4 })).toThrow(
      /an f16 input or an epilogue on a quantW weight needs the whole weight/,
    );
  });
});
