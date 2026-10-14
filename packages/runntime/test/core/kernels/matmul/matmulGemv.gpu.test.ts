import { describe, it } from 'vitest';
import { MATMUL_SMALL_M_MAX_ROWS } from '../../../../src/core/kernels/matmul/matmulGemv.ts';
import { gpu, requireFeature } from '../../../gpu.ts';
import { checkMatmul, EPILOGUES } from './checkMatmul.ts';

const { root } = await gpu();

describe('matmulGemv kernels', () => {
  // gemv: m = 1 with no activation (smallm needs m >= 4)
  it.each([
    { name: '1×1', k: 1, n: 1 },
    { name: 'a single row', k: 288, n: 864 },
    { name: 'a single column out', k: 64, n: 1 },
    { name: 'odd sizes', k: 33, n: 67 },
    { name: 'long K', k: 8192, n: 256 },
  ])('gemv, $name: [1, $k] × [$k, $n]', ({ k, n }) => checkMatmul(1, k, n));

  it.for([
    { name: 'a single row', k: 288, n: 864 },
    { name: 'odd sizes', k: 33, n: 67 },
  ])('f16 gemv, $name: [1, $k] × [$k, $n]', async ({ k, n }, ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    await checkMatmul(1, k, n, 'f16');
  });

  // gemv: m = 1. An activation sends m = 1 to plain, so only bias and addend
  // run here; matmul.gpu.test.ts covers the activations.
  it.for(
    (['f32', 'f16'] as const).flatMap((dtype) =>
      EPILOGUES.filter((e) => !e.epilogue.activation).map((e) => ({ dtype, ...e })),
    ),
  )('gemv $dtype with $name: [1, 33] × [33, 67]', async (c, ctx) => {
    if (c.dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    await checkMatmul(1, 33, 67, c.dtype, c.epilogue);
  });

  // smallm: m % 4 == 0, 4 <= m <= 32, n % 4 == 0, subgroups. k = 70 leaves a
  // k % 32 tail.
  it.for((['f32', 'f16'] as const).flatMap((dtype) => EPILOGUES.map((e) => ({ dtype, ...e }))))(
    'smallm $dtype with $name: [8, 70] × [70, 36]',
    async (c, ctx) => {
      requireFeature(ctx, root, 'subgroups');
      if (c.dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
      await checkMatmul(8, 70, 36, c.dtype, c.epilogue);
    },
  );

  // smallm: m % 4 == 0, 4 <= m <= 32, n % 4 == 0, subgroups. k = 70 leaves a
  // k % 32 tail; 96 and 256 do not.
  // it.for, not it.each: only it.for passes the test context the skip needs.
  it.for([
    { name: 'four rows', m: 4, k: 256, n: 128, dtype: 'f32' },
    { name: 'eight rows', m: 8, k: 96, n: 36, dtype: 'f32' },
    { name: 'the most rows it takes', m: MATMUL_SMALL_M_MAX_ROWS, k: 70, n: 4, dtype: 'f32' },
    { name: 'four rows', m: 4, k: 256, n: 128, dtype: 'f16' },
    { name: 'a k % 32 tail', m: 8, k: 70, n: 36, dtype: 'f16' },
  ] as const)('smallm, $dtype $name: [$m, $k] × [$k, $n]', async ({ m, k, n, dtype }, ctx) => {
    requireFeature(ctx, root, 'subgroups');
    if (dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    await checkMatmul(m, k, n, dtype);
  });
});
