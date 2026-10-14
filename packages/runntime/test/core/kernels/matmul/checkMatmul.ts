/** Shared by the matmul kernel suites: runs one matmul on the GPU and checks
 *  it against a plain CPU loop. */
import { tensor } from '../../../../src/core/gpu/buffers.ts';
import { toArray } from '../../../../src/core/gpu/eval.ts';
import { matmul } from '../../../../src/core/graph/ops.ts';
import { expectClose, gpu, randomF32 } from '../../../gpu.ts';

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

/** Runs [m, k] × [k, n] on random inputs and checks it against matmulRef.
 *  The route the shape takes is picked in gpu/dispatch/matmul.ts. */
export async function checkMatmul(m: number, k: number, n: number): Promise<void> {
  const { root, ex } = await gpu();
  const a = randomF32(m * k, 1);
  const b = randomF32(k * n, 2);
  const got = await toArray(matmul(tensor(root, a, [m, k]), tensor(root, b, [k, n])), ex);
  // f32 rounding grows with the length of each dot product.
  expectClose(got, matmulRef(a, b, m, k, n), { atol: 1e-6 * k, rtol: 1e-4 });
}
