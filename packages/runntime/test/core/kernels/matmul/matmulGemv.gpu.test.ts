import { describe, it } from 'vitest';
import { MATMUL_SMALL_M_MAX_ROWS } from '../../../../src/core/kernels/matmul/matmulGemv.ts';
import { gpu, requireFeature } from '../../../gpu.ts';
import { checkMatmul, EPILOGUES } from './checkMatmul.ts';

const { root } = await gpu();

describe('matmulGemv kernels', () => {
  // m = 1 with no activation (smallm needs m >= 4). gemvVec4 takes n % 4 == 0,
  // gemv the rest.
  it.each([
    // gemv: n % 4 != 0
    { name: '1×1 (gemv)', k: 1, n: 1 },
    { name: 'a single column out (gemv)', k: 64, n: 1 },
    { name: 'odd sizes (gemv)', k: 33, n: 67 },
    // gemvVec4: n % 4 == 0
    { name: 'one quad out (gemvVec4)', k: 33, n: 4 },
    { name: 'a single row (gemvVec4)', k: 288, n: 864 },
    { name: 'long K (gemvVec4)', k: 8192, n: 256 },
  ])('$name: [1, $k] × [$k, $n]', ({ k, n }) => checkMatmul(1, k, n));

  it.for([
    // gemv: n % 4 != 0
    { name: 'odd sizes (gemv)', k: 33, n: 67 },
    // gemvVec4: n % 4 == 0
    { name: 'a single row (gemvVec4)', k: 288, n: 864 },
    { name: 'odd K (gemvVec4)', k: 33, n: 68 },
  ])('f16 $name: [1, $k] × [$k, $n]', async ({ k, n }, ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    await checkMatmul(1, k, n, 'f16');
  });

  // m = 1. An activation sends m = 1 to plain, so only bias and addend run
  // here; matmul.gpu.test.ts covers the activations. gemv: n = 67, gemvVec4:
  // n = 68.
  it.for(
    (['f32', 'f16'] as const).flatMap((dtype) =>
      [
        { route: 'gemv', n: 67 },
        { route: 'gemvVec4', n: 68 },
      ].flatMap((shape) =>
        EPILOGUES.filter((e) => !e.epilogue.activation).map((e) => ({ dtype, ...shape, ...e })),
      ),
    ),
  )('$route $dtype with $name: [1, 33] × [33, $n]', async (c, ctx) => {
    if (c.dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    await checkMatmul(1, 33, c.n, c.dtype, c.epilogue);
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
    // smallm: n < 1024, so tiledVec4 does not take it
    { name: 'one quad under the tiledVec4 handoff', m: 8, k: 64, n: 1020, dtype: 'f32' },
    // smallm: n >= 1024, but tiledVec4 needs m >= 8
    { name: 'four rows at the tiledVec4 handoff', m: 4, k: 64, n: 1024, dtype: 'f32' },
  ] as const)('smallm, $dtype $name: [$m, $k] × [$k, $n]', async ({ m, k, n, dtype }, ctx) => {
    requireFeature(ctx, root, 'subgroups');
    if (dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    await checkMatmul(m, k, n, dtype);
  });

  // Just past each smallm limit, so the other route runs. No subgroups
  // needed: these never take smallm.
  it.for([
    // tiledVec4: n >= 1024 with m >= 8, k >= 64, k and n % 4 == 0
    { name: 'at the tiledVec4 handoff (tiledVec4)', m: 8, k: 64, n: 1024, dtype: 'f32' },
    { name: 'at the tiledVec4 handoff (tiledVec4)', m: 8, k: 64, n: 1024, dtype: 'f16' },
    // tiled: m = 36 is past smallm's 32 rows
    { name: 'one quad past the most rows (tiled)', m: 36, k: 70, n: 36, dtype: 'f32' },
    // plain: m = 2 is under smallm's 4 rows and over gemv's 1
    { name: 'two rows (plain)', m: 2, k: 70, n: 36, dtype: 'f32' },
    // plain: m = 6 is not a multiple of 4
    { name: 'six rows (plain)', m: 6, k: 70, n: 36, dtype: 'f32' },
    // tiled: n = 34 is not a multiple of 4
    { name: 'n % 4 != 0 (tiled)', m: 8, k: 70, n: 34, dtype: 'f32' },
    // plain: same, and tiledVec4 needs n % 4 == 0 too
    { name: 'n % 4 != 0 (plain)', m: 8, k: 70, n: 34, dtype: 'f16' },
  ] as const)('past smallm, $dtype $name: [$m, $k] × [$k, $n]', async ({ m, k, n, dtype }, ctx) => {
    if (dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    await checkMatmul(m, k, n, dtype);
  });
});
