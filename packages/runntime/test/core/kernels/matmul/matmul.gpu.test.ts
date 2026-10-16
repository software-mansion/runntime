import { describe, expect, it, onTestFinished } from 'vitest';
import { toArray } from '../../../../src/core/gpu/eval.ts';
import { matmul } from '../../../../src/core/graph/ops.ts';
import { matrix } from '../../../../src/core/graph/value.ts';
import { ACC_F16, ACC_F32, setMatmulAccum } from '../../../../src/core/kernels/matmul/matmul.ts';
import { gpu, requireFeature, upload } from '../../../gpu.ts';
import { checkMatmul, EPILOGUES } from './checkMatmul.ts';

const { root, ex } = await gpu();

describe('matmul kernel: plain and tiled routes', () => {
  // Routes are picked in gpu/dispatch/matmul.ts. smallm takes m % 4 == 0,
  // m <= 32, n % 4 == 0 on a subgroups device first, so the cases below avoid it.
  it.each([
    // plain: m < 8 and not a multiple of 4
    { name: 'odd sizes (plain)', m: 3, k: 7, n: 5 },
    { name: 'one row under the tiled cutoff (plain)', m: 7, k: 16, n: 12 },
    // tiled: f32, m >= 8, not a smallm shape and not a tiledVec4 shape
    // tiled: m = 9 is the first m >= 8 that smallm does not take on a subgroups device
    { name: 'one row past the tiled cutoff (tiled)', m: 9, k: 16, n: 12 },
    { name: 'K too short for tiledVec4 (tiled)', m: 9, k: 60, n: 8 },
    { name: 'one past the tile edge in every dim (tiled)', m: 65, k: 33, n: 67 },
    { name: 'tall and thin (tiled)', m: 1000, k: 3, n: 2 },
    // tiledVec4: m >= 8, k >= 64, k and n % 4 == 0
    { name: 'exactly one tile (tiledVec4)', m: 64, k: 64, n: 64 },
    { name: 'one row past the tiledVec4 cutoff (tiledVec4)', m: 9, k: 64, n: 64 },
    { name: 'past the tile edge (tiledVec4)', m: 70, k: 132, n: 68 },
  ])('$name: [$m, $k] × [$k, $n]', ({ m, k, n }) => checkMatmul(m, k, n));

  it.for([
    // plain: m < 8
    { name: 'odd sizes (plain)', m: 3, k: 16, n: 5 },
    { name: 'one row under the tiledVec4 cutoff (plain)', m: 7, k: 64, n: 64 },
    // plain: m >= 8, but tiledVec4 needs k >= 64 and f16 has no tiled route
    { name: 'K too short for tiledVec4 (plain)', m: 9, k: 60, n: 8 },
    // plain: m > 32 and k >= 64, but tiledVec4 also needs k and n % 4 == 0
    { name: 'K not a multiple of 4 (plain)', m: 40, k: 66, n: 64 },
    { name: 'N not a multiple of 4 (plain)', m: 40, k: 64, n: 66 },
    // tiledVec4: m >= 8, k >= 64, k and n % 4 == 0
    { name: 'tiledVec4 at the shortest K', m: 64, k: 64, n: 64 },
    // tiledVec4: m = 9 is the first m >= 8 that smallm does not take on a subgroups device
    { name: 'one row past the tiledVec4 cutoff', m: 9, k: 64, n: 64 },
    { name: 'tiledVec4 past the tile edge', m: 70, k: 132, n: 68 },
  ])('f16 $name: [$m, $k] × [$k, $n]', async ({ m, k, n }, ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    await checkMatmul(m, k, n, 'f16');
  });

  const epilogueCases = [
    ...[
      // plain: m < 8. n = 5 runs the vec4 columns and the scalar tail, n = 8
      // the vec4 columns alone.
      { route: 'plain', m: 3, k: 7, n: 5, dtype: 'f32' },
      { route: 'plain', m: 3, k: 7, n: 8, dtype: 'f32' },
      { route: 'plain', m: 3, k: 16, n: 5, dtype: 'f16' },
      { route: 'plain', m: 3, k: 16, n: 8, dtype: 'f16' },
      // tiled: f32, m >= 8, not a smallm shape and not a tiledVec4 shape
      { route: 'tiled', m: 65, k: 33, n: 67, dtype: 'f32' },
      // tiledVec4: m >= 8, k >= 64, k and n % 4 == 0
      { route: 'tiledVec4', m: 70, k: 132, n: 68, dtype: 'f32' },
      { route: 'tiledVec4', m: 70, k: 132, n: 68, dtype: 'f16' },
    ].flatMap((shape) => EPILOGUES.map((e) => ({ ...shape, ...e }))),
    // plain: m = 1 with an activation skips gemv, which has no activation slot
    ...['f32', 'f16'].flatMap((dtype) =>
      EPILOGUES.filter((e) => e.epilogue.activation).map((e) => ({
        route: 'plain',
        m: 1,
        k: 33,
        n: 67,
        dtype,
        ...e,
      })),
    ),
  ];
  it.for(epilogueCases)('$route $dtype with $name: [$m, $k] × [$k, $n]', async (c, ctx) => {
    if (c.dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    await checkMatmul(c.m, c.k, c.n, c.dtype as 'f32' | 'f16', c.epilogue);
  });

  // plain: a baseRow view always takes plain, whatever the shape
  it.for([
    { m: 3, k: 7, n: 5, baseRow: 4, dtype: 'f32' },
    { m: 40, k: 64, n: 64, baseRow: 64, dtype: 'f32' },
    { m: 3, k: 16, n: 5, baseRow: 4, dtype: 'f16' },
    { m: 40, k: 64, n: 64, baseRow: 64, dtype: 'f16' },
  ] as const)(
    'plain $dtype, B as a view from row $baseRow: [$m, $k] × [$k, $n]',
    async ({ m, k, n, baseRow, dtype }, ctx) => {
      if (dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
      await checkMatmul(m, k, n, dtype, { baseRow, bias: true, addend: true });
    },
  );

  // plain: f16 with the f16 accumulator. The vec4 columns honour it; the
  // scalar tail always sums in f32. Each of the 16 adds rounds a partial sum
  // of size up to ~4 to f16 (2^-11 relative), so the error stays under 3e-2.
  it.for([
    { name: 'vec4 columns only', n: 8 },
    { name: 'with the scalar tail', n: 5 },
  ])('plain f16 with the f16 accumulator, $name: [3, 16] × [16, $n]', async ({ n }, ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    setMatmulAccum(ACC_F16);
    onTestFinished(() => setMatmulAccum(ACC_F32));
    await checkMatmul(3, 16, n, 'f16', {}, { atol: 3e-2, rtol: 1e-2 });
  });

  // Known bug: the tiled kernel's B index goes through f32, which counts
  // exactly only up to 2^24. Turn this into a plain `it` with the fix.
  // tiled: f32, m = 9 is not a smallm shape, k % 4 != 0 rules out tiledVec4
  it.fails('B larger than 2^24 elements (tiled): [9, 2561] × [2561, 9728]', () =>
    checkMatmul(9, 2561, 9728),
  );

  it('rejects M = 0 with an INVALID_ARGUMENT', () => {
    expect(() => upload(root, new Float32Array(0), [0, 4])).toThrow(
      expect.objectContaining({ code: 'INVALID_ARGUMENT' }),
    );
  });

  it('refuses f32 × f16', () => {
    const a = upload(root, new Float32Array(6), [2, 3]);
    const b = upload(root, new Float32Array(15), { ...matrix(3, 5), dtype: 'f16' });
    expect(() => matmul(a, b)).toThrow(/must be f32 or both f16, got f32 and f16/);
  });

  it('refuses an output past maxStorageBufferBindingSize instead of running', async () => {
    // [n, 1] × [1, n] is small in, n² out.
    const n = Math.ceil(Math.sqrt(root.device.limits.maxStorageBufferBindingSize / 4)) + 1;
    const a = upload(root, new Float32Array(n), [n, 1]);
    const b = upload(root, new Float32Array(n), [1, n]);
    await expect(toArray(matmul(a, b), ex)).rejects.toMatchObject({ code: 'UNSUPPORTED_DEVICE' });
  });
});
