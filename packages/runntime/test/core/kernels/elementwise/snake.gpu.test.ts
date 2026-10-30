import { describe, expect, it } from 'vitest';
import { toArray } from '../../../../src/core/gpu/eval.ts';
import { snake } from '../../../../src/core/graph/ops.ts';
import { matrix } from '../../../../src/core/graph/value.ts';
import { GRID_Y_STRIDE } from '../../../../src/core/kernels/common.ts';
import { expectClose, gpu, randomF32, requireFeature, roundF16, upload } from '../../../gpu.ts';

const { root, ex } = await gpu();

/** DAC's Snake on time-major x [rows, C]: x + sin²(αx) / (α + 1e-9), one α
 *  per channel, in f64. */
function snakeRef(x: Float32Array, alpha: Float32Array): Float64Array {
  const c = alpha.length;
  return Float64Array.from(x, (v, i) => {
    const a = alpha[i % c]!;
    return v + Math.sin(a * v) ** 2 / (a + 1e-9);
  });
}

/** f32: a sin and two multiplies. f16 output adds 2^-11 relative on values
 *  up to 1 / α. */
const TOLERANCE = { f32: { atol: 1e-5, rtol: 1e-5 }, f16: { atol: 1e-2, rtol: 3e-3 } };

async function checkSnake(rows: number, c: number, dtype: 'f32' | 'f16') {
  const scaled = randomF32(rows * c, 1).map((v) => 4 * v);
  const x = dtype === 'f16' ? roundF16(scaled) : scaled;
  const alpha = randomF32(c, 2).map((v) => Math.abs(v) * 3 + 0.05);
  const ab = new Float32Array(2 * c);
  for (let i = 0; i < c; i++) {
    ab[i] = alpha[i]!;
    ab[c + i] = 1 / (alpha[i]! + 1e-9);
  }
  const y = snake(upload(root, x, { ...matrix(rows, c), dtype }), upload(root, ab, [2, c]));
  expect(y.shape).toEqual({ ...matrix(rows, c), dtype });
  expectClose(await toArray(y, ex), snakeRef(x, alpha), TOLERANCE[dtype]);
}

describe('snake kernel', () => {
  // One route: a thread per 4 channels of one frame.
  it.for(
    (['f32', 'f16'] as const).flatMap((dtype) =>
      [
        { name: 'one unit', rows: 1, c: 4 },
        { name: 'odd frames', rows: 37, c: 24 },
        { name: 'one past a workgroup', rows: 1, c: 260 },
        { name: 'many frames', rows: 300, c: 64 },
      ].map((c) => ({ dtype, ...c })),
    ),
  )('$dtype $name: [$rows, $c]', async ({ dtype, rows, c }, ctx) => {
    if (dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    await checkSnake(rows, c, dtype);
  });

  it('runs past GRID_Y_STRIDE units, in the second grid dimension', async () => {
    expect((262_145 * 64) / 4).toBeGreaterThan(GRID_Y_STRIDE);
    await checkSnake(262_145, 64, 'f32');
  });
});
