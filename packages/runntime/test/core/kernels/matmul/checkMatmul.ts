/** Shared by the matmul kernel suites: runs one matmul on the GPU and checks
 *  it against a plain CPU loop. */
import { toArray } from '../../../../src/core/gpu/eval.ts';
import { matmul } from '../../../../src/core/graph/ops.ts';
import { matrix } from '../../../../src/core/graph/value.ts';
import { expectClose, gpu, randomF32, roundF16, upload } from '../../../gpu.ts';

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
export async function checkMatmul(
  m: number,
  k: number,
  n: number,
  dtype: 'f32' | 'f16' = 'f32',
): Promise<void> {
  const { root, ex } = await gpu();
  // The reference sees the f16-rounded inputs the kernel reads.
  const round = dtype === 'f16' ? roundF16 : (x: Float32Array) => x;
  const a = round(randomF32(m * k, 1));
  const b = round(randomF32(k * n, 2));
  const av = upload(root, a, { ...matrix(m, k), dtype });
  const bv = upload(root, b, { ...matrix(k, n), dtype });
  const got = await toArray(matmul(av, bv), ex);
  // f32 rounding grows with the length of each dot product; f16 output adds
  // 2^-11 relative on top.
  const tol = dtype === 'f16' ? { atol: 1e-3, rtol: 2e-3 } : { atol: 1e-6 * k, rtol: 1e-4 };
  expectClose(got, matmulRef(a, b, m, k, n), tol);
}
