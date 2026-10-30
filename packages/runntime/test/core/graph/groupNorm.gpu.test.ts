import { describe, expect, it } from 'vitest';
import { groupNorm } from '../../../src/core/graph/compose.ts';
import { toArray } from '../../../src/core/gpu/eval.ts';
import { matrix } from '../../../src/core/graph/value.ts';
import { expectClose, gpu, randomF32, requireFeature, roundF16, upload } from '../../gpu.ts';

const { root, ex } = await gpu();

/** torch's GroupNorm over [m, c], channels along the columns, in f64. */
function groupNormRef(
  x: Float32Array,
  m: number,
  c: number,
  groups: number,
  w: Float32Array,
  b: Float32Array,
  eps: number,
): Float64Array {
  const out = new Float64Array(m * c);
  const size = c / groups;
  for (let g = 0; g < groups; g++) {
    let sum = 0;
    for (let t = 0; t < m; t++) for (let j = 0; j < size; j++) sum += x[t * c + g * size + j]!;
    const mean = sum / (m * size);
    let sq = 0;
    for (let t = 0; t < m; t++) {
      for (let j = 0; j < size; j++) sq += (x[t * c + g * size + j]! - mean) ** 2;
    }
    const inv = 1 / Math.sqrt(sq / (m * size) + eps);
    for (let t = 0; t < m; t++) {
      for (let j = 0; j < size; j++) {
        const k = g * size + j;
        out[t * c + k] = (x[t * c + k]! - mean) * inv * w[k]! + b[k]!;
      }
    }
  }
  return out;
}

describe('groupNorm', () => {
  it.for([
    { name: 'one group', m: 37, c: 8, groups: 1 },
    { name: 'two groups', m: 37, c: 8, groups: 2 },
    { name: 'a group per channel', m: 37, c: 8, groups: 8 },
    // Row means sum one value after another in f32, so a long group keeps
    // fewer digits.
    {
      name: 'a group per channel, HuBERT conv0 [3199,512]',
      m: 3199,
      c: 512,
      groups: 512,
      tol: { atol: 1e-4, rtol: 1e-3 },
    },
  ])('$name: [$m, $c] in $groups', async ({ m, c, groups, tol }) => {
    // An offset per channel in [0, 4), so a wrong group mean shows.
    const x = randomF32(m * c, 1).map((v, i) => 3 * v + (4 * (i % c)) / c);
    const w = randomF32(c, 2).map((v) => v + 1.5);
    const b = randomF32(c, 3);
    const y = groupNorm(
      upload(root, x, [m, c]),
      groups,
      upload(root, w, [1, c]),
      upload(root, b, [1, c]),
    );
    expect(y.shape).toEqual(matrix(m, c));
    expectClose(
      await toArray(y, ex),
      groupNormRef(x, m, c, groups, w, b, 1e-5),
      tol ?? { atol: 1e-5, rtol: 1e-4 },
    );
  });

  it('runs a group per channel in f16', async (ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const [m, c] = [37, 8];
    const x = roundF16(randomF32(m * c, 1).map((v) => 2 * v));
    const w = roundF16(randomF32(c, 2).map((v) => v + 1.5));
    const b = roundF16(randomF32(c, 3));
    const f16 = (data: Float32Array, rows: number) =>
      upload(root, data, { ...matrix(rows, data.length / rows), dtype: 'f16' });
    const y = groupNorm(f16(x, m), c, f16(w, 1), f16(b, 1));
    expectClose(await toArray(y, ex), groupNormRef(x, m, c, c, w, b, 1e-5), {
      atol: 1e-2,
      rtol: 1e-2,
    });
  });

  it('rejects a group count that does not divide the channels', () => {
    const one = (n: number) => upload(root, new Float32Array(n), [1, n]);
    expect(() => groupNorm(upload(root, new Float32Array(16), [2, 8]), 3, one(8), one(8))).toThrow(
      /numGroups 3 must divide the 8 channels/,
    );
  });
});
