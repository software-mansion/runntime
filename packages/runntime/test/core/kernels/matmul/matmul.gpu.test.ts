import { describe, expect, it } from 'vitest';
import { toArray } from '../../../../src/core/gpu/eval.ts';
import { matmul } from '../../../../src/core/graph/ops.ts';
import { matrix } from '../../../../src/core/graph/value.ts';
import { gpu, requireFeature, upload } from '../../../gpu.ts';
import { checkMatmul } from './checkMatmul.ts';

const { root, ex } = await gpu();

describe('matmul kernel: plain and tiled routes', () => {
  it.each([
    { name: 'odd sizes (plain)', m: 3, k: 7, n: 5 },
    { name: 'one row under the tiled cutoff (plain)', m: 7, k: 16, n: 12 },
    { name: 'exactly one tile (tiled)', m: 64, k: 64, n: 64 },
    { name: 'one past the tile edge in every dim (tiled)', m: 65, k: 33, n: 67 },
    { name: 'tall and thin (tiled)', m: 1000, k: 3, n: 2 },
  ])('$name: [$m, $k] × [$k, $n]', ({ m, k, n }) => checkMatmul(m, k, n));

  it.for([
    { name: 'odd sizes (plain)', m: 3, k: 16, n: 5 },
    { name: 'K too short for tiledF16 (plain)', m: 9, k: 60, n: 8 },
    { name: 'tiledF16', m: 16, k: 64, n: 64 },
    { name: 'tiledF16 past the tile edge', m: 70, k: 132, n: 68 },
  ])('f16 $name: [$m, $k] × [$k, $n]', async ({ m, k, n }, ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    await checkMatmul(m, k, n, 'f16');
  });

  // Known bug: the tiled kernel's B index goes through f32, which counts
  // exactly only up to 2^24. Turn this into a plain `it` with the fix.
  it.fails('B larger than 2^24 elements (tiled): [9, 2560] × [2560, 9728]', () =>
    checkMatmul(9, 2560, 9728),
  );

  // Known bug: an empty tensor reaches createBuffer and fails with the
  // browser's TypeError instead of a RunntimeError. Turn into `it` with the fix.
  it.fails('rejects M = 0 with an INVALID_ARGUMENT', () => {
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
