import { describe, it } from 'vitest';
import { MATMUL_SMALL_M_MAX_ROWS } from '../../../../src/core/kernels/matmul/matmulGemv.ts';
import { gpu, requireFeature } from '../../../gpu.ts';
import { checkMatmul } from './checkMatmul.ts';

const { root } = await gpu();

describe('matmulGemv kernels', () => {
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

  // it.for, not it.each: only it.for passes the test context the skip needs.
  it.for([
    { name: 'four rows', m: 4, k: 256, n: 128, dtype: 'f32' },
    { name: 'eight rows', m: 8, k: 96, n: 36, dtype: 'f32' },
    { name: 'the most rows it takes', m: MATMUL_SMALL_M_MAX_ROWS, k: 70, n: 4, dtype: 'f32' },
    { name: 'four rows', m: 4, k: 256, n: 128, dtype: 'f16' },
  ] as const)('smallm, $dtype $name: [$m, $k] × [$k, $n]', async ({ m, k, n, dtype }, ctx) => {
    requireFeature(ctx, root, 'subgroups');
    if (dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    await checkMatmul(m, k, n, dtype);
  });
});
