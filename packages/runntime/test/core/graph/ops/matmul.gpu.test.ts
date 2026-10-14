import { describe, it } from 'vitest';
import { tensor } from '../../../../src/core/gpu/buffers.ts';
import { toArray } from '../../../../src/core/gpu/eval.ts';
import { matmul } from '../../../../src/core/graph/ops.ts';
import { expectClose, gpu, randomF32 } from '../../../gpu.ts';

const { root, ex } = await gpu();

/** C = A·B for row-major A [m, k] and B [k, n], accumulated in f64. */
function matmulRef(
  a: Float32Array,
  b: Float32Array,
  m: number,
  k: number,
  n: number,
): Float64Array {
  const c = new Float64Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let p = 0; p < k; p++) sum += a[i * k + p]! * b[p * n + j]!;
      c[i * n + j] = sum;
    }
  }
  return c;
}

/** Runs [m, k] × [k, n] on random inputs and checks it against matmulRef. */
async function checkMatmul(m: number, k: number, n: number): Promise<void> {
  const a = randomF32(m * k, 1);
  const b = randomF32(k * n, 2);
  const got = await toArray(matmul(tensor(root, a, [m, k]), tensor(root, b, [k, n])), ex);
  // f32 rounding grows with the length of each dot product.
  expectClose(got, matmulRef(a, b, m, k, n), { atol: 1e-6 * k, rtol: 1e-4 });
}

describe('matmul on the GPU', () => {
  // The route each shape takes is picked in gpu/dispatch/matmul.ts.
  it.each([
    { name: '1×1', m: 1, k: 1, n: 1 },
    { name: 'odd sizes (plain)', m: 3, k: 7, n: 5 },
    { name: 'a single row (gemv)', m: 1, k: 288, n: 864 },
    { name: 'four rows (smallm with subgroups)', m: 4, k: 256, n: 128 },
    { name: 'M past the tile edge (tiled)', m: 65, k: 33, n: 67 },
  ])('$name: [$m, $k] × [$k, $n]', ({ m, k, n }) => checkMatmul(m, k, n));

  // Known bug: the tiled kernel's B index goes through f32, which counts
  // exactly only up to 2^24. Turn this into a plain `it` with the fix.
  it.fails('B larger than 2^24 elements (tiled): [9, 2560] × [2560, 9728]', () =>
    checkMatmul(9, 2560, 9728),
  );
});
