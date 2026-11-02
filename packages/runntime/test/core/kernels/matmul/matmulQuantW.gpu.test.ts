import { describe, expect, it } from 'vitest';
import { toArray } from '../../../../src/core/gpu/eval.ts';
import { matmul, type MatmulOpts } from '../../../../src/core/graph/ops.ts';
import { matrix } from '../../../../src/core/graph/value.ts';
import { MAX_WORKGROUPS_PER_DIM } from '../../../../src/core/kernels/common.ts';
import { expectClose, gpu, randomF32, requireFeature, roundF16, upload } from '../../../gpu.ts';
import { ACTIVATION_REF, EPILOGUES, type Epilogue } from './checkMatmul.ts';
import { randomQuantWeight } from './quantW.ts';

const { root, ex } = await gpu();

interface Case extends Epilogue {
  m: number;
  k: number;
  n: number;
  bits: 4 | 8;
  groupSize: number;
  dtype?: 'f32' | 'f16';
  experts?: number;
  /** The expert the call reads, through baseRow and scaleBase. */
  expert?: number;
}

/** act(a · dequant(W[expert]) + bias) + addend against a plain loop in f64. */
async function checkQuantMatmul(c: Case) {
  const { m, k, n, bits, groupSize, dtype = 'f32', experts = 1, expert = 0 } = c;
  const round = dtype === 'f16' ? roundF16 : (v: Float32Array) => v;
  const meta = (rows: number, cols: number) => ({ ...matrix(rows, cols), dtype });
  const weight = randomQuantWeight(root, { experts, k, n, bits, groupSize }, 2);
  const a = round(randomF32(m * k, 1));
  const bias = c.bias ? round(randomF32(n, 3)) : undefined;
  const addend = c.addend ? round(randomF32(m * n, 4)) : undefined;
  const got = await toArray(
    matmul(upload(root, a, meta(m, k)), weight.w, {
      scales: upload(root, weight.scales, [weight.scales.length / n, n]),
      bits,
      groupSize,
      baseRow: expert * k,
      scaleBase: ((expert * k) / groupSize) * n,
      bias: bias && upload(root, bias, meta(1, n)),
      addend: addend && upload(root, addend, meta(m, n)),
      activation: c.activation,
    }),
    ex,
  );
  const want = new Float64Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let col = 0; col < n; col++) {
      let sum = bias ? bias[col]! : 0;
      for (let j = 0; j < k; j++)
        sum += a[i * k + j]! * weight.dequant[(expert * k + j) * n + col]!;
      if (c.activation) sum = ACTIVATION_REF[c.activation](sum);
      if (addend) sum += addend[i * n + col]!;
      want[i * n + col] = sum;
    }
  }
  // Sums are f32 for both dtypes; f16 rounds the output to half precision.
  const tol = dtype === 'f16' ? { atol: 1e-5 * k, rtol: 1e-3 } : { atol: 1e-6 * k, rtol: 1e-5 };
  expectClose(got, want, tol);
}

describe('matmulQuantW kernel', () => {
  // Routes are picked in gpu/dispatch/matmul.ts. This kernel takes a whole
  // weight below 48·1024 output values (m · n), one expert of a fused weight
  // whatever m, and any call with K not a multiple of 8. Other calls take
  // matmulQuantTiled. K and groupSize are multiples of 4, N of 4 for int8 and
  // of 8 for int4. A workgroup serves 4 rows × 8 four-column units, with K
  // split across 16 lanes.
  // f16 needs a whole weight with K % 8 == 0, which the op checks.
  const both = ['f32', 'f16'] as const;
  const f32 = ['f32'] as const;
  it.for(
    [
      // matmulQuantW: m · n < 48·1024
      { name: 'one tile', m: 1, k: 4, n: 8, groupSize: 4, dtypes: f32 },
      { name: 'one group per column', m: 3, k: 12, n: 8, groupSize: 12, dtypes: f32 },
      { name: 'a group every 4 rows', m: 3, k: 12, n: 16, groupSize: 4, dtypes: f32 },
      { name: 'one 8-row tile', m: 1, k: 8, n: 8, groupSize: 4, dtypes: both },
      { name: 'groups of 8, a partial row block', m: 5, k: 64, n: 24, groupSize: 8, dtypes: both },
      { name: 'a partial block of units', m: 8, k: 96, n: 40, groupSize: 32, dtypes: both },
      { name: 'a wide row', m: 2, k: 640, n: 640, groupSize: 32, dtypes: both },
      { name: 'an LLM decode row', m: 1, k: 2048, n: 2048, groupSize: 64, dtypes: both },
      // matmulQuantW: one below 48·1024 outputs; the tiled suite has the first above
      {
        name: 'one output short of the tiled route',
        m: 1535,
        k: 32,
        n: 32,
        groupSize: 16,
        dtypes: both,
      },
      // matmulQuantW: one expert of a fused weight, whatever m
      {
        name: 'rows past a workgroup on one expert',
        m: 70,
        k: 32,
        n: 8,
        groupSize: 16,
        experts: 2,
        expert: 1,
        dtypes: f32,
      },
    ].flatMap(({ dtypes, ...c }) =>
      dtypes.flatMap((dtype) => ([8, 4] as const).map((bits) => ({ dtype, bits, ...c }))),
    ),
  )('$dtype int$bits $name: [$m, $k] × [$k, $n], groupSize $groupSize', async (c, ctx) => {
    if (c.dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    await checkQuantMatmul(c);
  });

  // matmulQuantW: K % 8 != 0. Past MAX_WORKGROUPS_PER_DIM workgroups the
  // dispatch goes 2-D.
  it.for([8, 4] as const)('int%i workgroups past MAX_WORKGROUPS_PER_DIM', (bits) =>
    checkQuantMatmul({ m: 4 * MAX_WORKGROUPS_PER_DIM + 5, k: 4, n: 8, bits, groupSize: 4 }),
  );

  // matmulQuantW: every epilogue on a few rows, m · n < 48·1024
  it.for(
    (['f32', 'f16'] as const).flatMap((dtype) =>
      ([8, 4] as const).flatMap((bits) => EPILOGUES.map((e) => ({ dtype, bits, ...e }))),
    ),
  )('$dtype int$bits with $name on 5 rows', async ({ dtype, bits, epilogue }, ctx) => {
    if (dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    await checkQuantMatmul({ m: 5, k: 64, n: 40, bits, groupSize: 16, dtype, ...epilogue });
  });

  // matmulQuantW: baseRow and scaleBase read one expert of a fused [E·K, N] weight.
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
    {
      name: 'a scaleBase that is not a multiple of 4',
      opts: { baseRow: 8, scaleBase: 2 },
      err: /scaleBase 2 must be a multiple of 4/,
    },
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
      scaleBase: opts.scaleBase,
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
