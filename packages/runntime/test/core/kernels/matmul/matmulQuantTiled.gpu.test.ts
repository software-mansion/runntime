import { describe, it } from 'vitest';
import { toArray } from '../../../../src/core/gpu/eval.ts';
import { matmul, type SlotAct } from '../../../../src/core/graph/ops.ts';
import { matrix } from '../../../../src/core/graph/value.ts';
import { expectClose, gpu, randomF32, requireFeature, roundF16, upload } from '../../../gpu.ts';
import { ACTIVATION_REF } from './checkMatmul.ts';
import { randomQuantWeight } from './quantW.ts';

const { root, ex } = await gpu();

interface Case {
  m: number;
  k: number;
  n: number;
  bits: 4 | 8;
  groupSize: number;
  bias?: boolean;
  addend?: boolean;
  activation?: SlotAct;
}

/** act(a · dequant(W) + bias) + addend on the whole quantW weight, through the
 *  tiled route: an f16 input, an epilogue, or at least 8 rows. */
async function checkQuantTiled(c: Case, dtype: 'f32' | 'f16') {
  const { m, k, n, bits, groupSize } = c;
  const round = dtype === 'f16' ? roundF16 : (v: Float32Array) => v;
  const meta = (rows: number, cols: number) => ({ ...matrix(rows, cols), dtype });
  const weight = randomQuantWeight(root, { experts: 1, k, n, bits, groupSize }, 2);
  const a = round(randomF32(m * k, 1));
  const bias = c.bias ? round(randomF32(n, 3)) : undefined;
  const addend = c.addend ? round(randomF32(m * n, 4)) : undefined;
  const got = await toArray(
    matmul(upload(root, a, meta(m, k)), weight.w, {
      scales: upload(root, weight.scales, [k / groupSize, n]),
      bits,
      groupSize,
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
      for (let j = 0; j < k; j++) sum += a[i * k + j]! * weight.dequant[j * n + col]!;
      if (c.activation) sum = ACTIVATION_REF[c.activation](sum);
      if (addend) sum += addend[i * n + col]!;
      want[i * n + col] = sum;
    }
  }
  // The f16 route stores dequantized weights in f16 before the f32 sums.
  const tol =
    dtype === 'f16' ? { atol: 2e-3 * Math.sqrt(k), rtol: 5e-3 } : { atol: 1e-6 * k, rtol: 1e-5 };
  expectClose(got, want, tol);
}

describe('matmulQuantTiled kernel', () => {
  // tiled: a whole weight from 8 rows, K % 8 == 0
  const SHAPES = [
    { name: 'one tile', m: 8, k: 8, n: 8, groupSize: 4 },
    { name: 'partial row and column tiles', m: 37, k: 48, n: 40, groupSize: 8 },
    { name: 'one group per column', m: 9, k: 64, n: 16, groupSize: 64 },
    { name: 'a fused q|k projection', m: 97, k: 1024, n: 2304, groupSize: 32 },
    { name: 'a per-channel down projection', m: 40, k: 2048, n: 1024, groupSize: 2048 },
  ];
  it.for(
    (['f32', 'f16'] as const).flatMap((dtype) =>
      ([4, 8] as const).flatMap((bits) => SHAPES.map((s) => ({ dtype, bits, ...s }))),
    ),
  )('$dtype int$bits $name: [$m, $k] × [$k, $n], groupSize $groupSize', async (c, ctx) => {
    if (c.dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    await checkQuantTiled(c, c.dtype);
  });

  // tiled: an f16 input or an epilogue on a whole weight, whatever m
  it.for(
    (['f32', 'f16'] as const).flatMap((dtype) =>
      ([4, 8] as const).flatMap((bits) =>
        [
          { name: 'bias', bias: true },
          { name: 'addend', addend: true },
          { name: 'silu', activation: 'silu' as const },
          { name: 'bias, gelu and addend', bias: true, activation: 'gelu' as const, addend: true },
        ].map((e) => ({ dtype, bits, ...e })),
      ),
    ),
  )('$dtype int$bits with $name on 3 rows', async ({ dtype, ...e }, ctx) => {
    if (dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    await checkQuantTiled({ m: 3, k: 64, n: 24, groupSize: 16, ...e }, dtype);
  });
});
