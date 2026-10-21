import { describe, expect, it } from 'vitest';
import { toArray } from '../../../../src/core/gpu/eval.ts';
import { layerNorm } from '../../../../src/core/graph/ops.ts';
import { matrix } from '../../../../src/core/graph/value.ts';
import { expectClose, gpu, randomF32, requireFeature, roundF16, upload } from '../../../gpu.ts';

const { root, ex } = await gpu();

/** (x − mean) / sqrt(var + eps) · weight + bias over each row, with torch's
 *  biased 1/N variance, in f64. */
function layerNormRef(
  x: Float32Array,
  weight: Float32Array,
  rows: number,
  cols: number,
  eps: number,
  bias?: Float32Array,
): Float64Array {
  const out = new Float64Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    let sum = 0;
    for (let j = 0; j < cols; j++) sum += x[r * cols + j]!;
    const mean = sum / cols;
    let sq = 0;
    for (let j = 0; j < cols; j++) sq += (x[r * cols + j]! - mean) ** 2;
    const inv = 1 / Math.sqrt(sq / cols + eps);
    for (let j = 0; j < cols; j++) {
      out[r * cols + j] = (x[r * cols + j]! - mean) * inv * weight[j]! + (bias ? bias[j]! : 0);
    }
  }
  return out;
}

/** f32: two tree sums and an inverseSqrt, a few ulp each. f16 output adds
 *  2^-11 relative. */
const TOLERANCE = { f32: { atol: 1e-5, rtol: 1e-5 }, f16: { atol: 2e-3, rtol: 2e-3 } };

interface Case {
  rows: number;
  cols: number;
  dtype: 'f32' | 'f16';
  bias?: boolean;
  eps?: number;
  /** Added to every x, to check the variance does not cancel. */
  offset?: number;
}

async function checkLayerNorm(
  { rows, cols, dtype, bias, eps = 1e-5, offset = 0 }: Case,
  tolerance = TOLERANCE[dtype],
) {
  const round = dtype === 'f16' ? roundF16 : (v: Float32Array) => v;
  const x = round(randomF32(rows * cols, 1).map((v) => v + offset));
  const weight = round(randomF32(cols, 2));
  const b = bias ? round(randomF32(cols, 3)) : undefined;
  const meta = (r: number, n: number) => ({ ...matrix(r, n), dtype });
  const got = await toArray(
    layerNorm(
      upload(root, x, meta(rows, cols)),
      upload(root, weight, meta(1, cols)),
      eps,
      b && upload(root, b, meta(1, cols)),
    ),
    ex,
  );
  expectClose(got, layerNormRef(x, weight, rows, cols, eps, b), tolerance);
}

describe('layerNorm kernel', () => {
  // One route: a 64-thread workgroup per row, whatever the shape.
  const SHAPES = [
    { name: '1×1', rows: 1, cols: 1 },
    { name: 'odd sizes', rows: 3, cols: 7 },
    { name: 'exactly the 64-thread span', rows: 4, cols: 64 },
    { name: 'one past the 64-thread span', rows: 4, cols: 65 },
    { name: 'MiniLM width', rows: 128, cols: 384 },
    { name: 'a wide row', rows: 2, cols: 16385 },
    // past 65535 rows the grid is 2-D and its last row runs past the end
    { name: 'past 65535 rows', rows: 70000, cols: 3 },
  ];
  it.for(
    (['f32', 'f16'] as const).flatMap((dtype) =>
      [false, true].flatMap((bias) => SHAPES.map((shape) => ({ dtype, bias, ...shape }))),
    ),
  )('$dtype $name, bias $bias: [$rows, $cols]', async (c, ctx) => {
    if (c.dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    await checkLayerNorm(c);
  });

  // The variance is taken around the mean, so a large offset with a small
  // spread does not cancel the way E[x²] − mean² would, which is off by
  // O(1) here. The f32 mean of values near 1000 is good to ~1e-4, so the
  // output is too.
  it.for([
    { dtype: 'f32', offset: 1000, tolerance: { atol: 1e-3, rtol: 0 } },
    { dtype: 'f16', offset: 100, tolerance: TOLERANCE.f16 },
  ] as const)('$dtype keeps the variance under an offset of $offset', async (c, ctx) => {
    if (c.dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    await checkLayerNorm({ rows: 3, cols: 100, ...c, bias: true }, c.tolerance);
  });

  it('honours eps', () => checkLayerNorm({ rows: 3, cols: 50, dtype: 'f32', eps: 0.5 }));

  // x = [1, 3]: mean 2, biased variance 1, so (x − 2) / sqrt(1 + 1e-5) =
  // [-0.999995, 0.999995] as torch.nn.functional.layer_norm gives, then
  // · [2, 3] + [0.5, -0.5].
  it('matches torch layer_norm worked out by hand', async () => {
    const x = upload(root, new Float32Array([1, 3]), [1, 2]);
    const w = upload(root, new Float32Array([2, 3]), [1, 2]);
    const b = upload(root, new Float32Array([0.5, -0.5]), [1, 2]);
    expectClose(await toArray(layerNorm(x, w, 1e-5, b), ex), [-1.49999, 2.499985], {
      atol: 0,
      rtol: 1e-6,
    });
  });

  it('runs past 2^24 elements', async () => {
    const rows = 4100;
    const cols = 4096;
    expect(rows * cols).toBeGreaterThan(2 ** 24);
    await checkLayerNorm({ rows, cols, dtype: 'f32', bias: true });
  });

  it('rejects an empty input with an INVALID_ARGUMENT', () => {
    expect(() => upload(root, new Float32Array(0), [0, 4])).toThrow(
      expect.objectContaining({ code: 'INVALID_ARGUMENT' }),
    );
  });

  it.for([
    {
      name: 'a weight of the wrong length',
      wl: 3,
      bl: 4,
      eps: 1e-5,
      err: /weight must be f32 \[N\] = 4 elems, got \(3,f32\)/,
    },
    {
      name: 'a bias of the wrong length',
      wl: 4,
      bl: 5,
      eps: 1e-5,
      err: /bias must be f32 \[N\] = 4 elems, got \(5,f32\)/,
    },
    { name: 'eps 0', wl: 4, bl: 4, eps: 0, err: /eps must be positive, got 0/ },
  ])('refuses $name', ({ wl, bl, eps, err }) => {
    const x = upload(root, new Float32Array(8), [2, 4]);
    const w = upload(root, new Float32Array(wl), [1, wl]);
    const b = upload(root, new Float32Array(bl), [1, bl]);
    expect(() => layerNorm(x, w, eps, b)).toThrow(err);
  });

  it('refuses an f16 weight on f32 x', (ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const x = upload(root, new Float32Array(8), [2, 4]);
    const w = upload(root, new Float32Array(4), { ...matrix(1, 4), dtype: 'f16' });
    expect(() => layerNorm(x, w)).toThrow(/weight must be f32 \[N\] = 4 elems, got \(4,f16\)/);
  });
});
