import { describe, it } from 'vitest';
import { gpu, requireFeature } from '../../../gpu.ts';
import { checkMatmul } from './checkMatmul.ts';

const { root } = await gpu();

describe('matmulGemv kernels', () => {
  it.each([
    { name: '1×1', k: 1, n: 1 },
    { name: 'a single row', k: 288, n: 864 },
    { name: 'a single column out', k: 64, n: 1 },
  ])('gemv, $name: [1, $k] × [$k, $n]', ({ k, n }) => checkMatmul(1, k, n));

  // it.for, not it.each: only it.for passes the test context the skip needs.
  it.for([
    { name: 'four rows', m: 4, k: 256, n: 128 },
    { name: 'eight rows', m: 8, k: 96, n: 36 },
  ])('smallm, $name: [$m, $k] × [$k, $n]', async ({ m, k, n }, ctx) => {
    requireFeature(ctx, root, 'subgroups');
    await checkMatmul(m, k, n);
  });
});
