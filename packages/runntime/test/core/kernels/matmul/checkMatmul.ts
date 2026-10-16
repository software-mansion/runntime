/** Shared by the matmul kernel suites: runs one matmul on the GPU and checks
 *  it against a plain CPU loop. */
import { toArray } from '../../../../src/core/gpu/eval.ts';
import { matmul } from '../../../../src/core/graph/ops.ts';
import type { SlotAct } from '../../../../src/core/graph/ops/shared.ts';
import { matrix } from '../../../../src/core/graph/value.ts';
import { expectClose, gpu, randomF32, roundF16, upload } from '../../../gpu.ts';

/** The fused epilogue, out = activation(a·b + bias) + addend, and the
 *  baseRow view that reads B as K rows of a taller tensor. */
export interface Epilogue {
  bias?: boolean;
  addend?: boolean;
  activation?: SlotAct;
  baseRow?: number;
}

/** One case per epilogue branch, plus all of them together. Every matmul
 *  route runs this list. */
export const EPILOGUES: readonly { name: string; epilogue: Epilogue }[] = [
  { name: 'bias', epilogue: { bias: true } },
  { name: 'addend', epilogue: { addend: true } },
  { name: 'tanh', epilogue: { activation: 'tanh' } },
  { name: 'gelu', epilogue: { activation: 'gelu' } },
  { name: 'silu', epilogue: { activation: 'silu' } },
  { name: 'relu', epilogue: { activation: 'relu' } },
  { name: 'geluTanh', epilogue: { activation: 'geluTanh' } },
  { name: 'bias, gelu and addend', epilogue: { bias: true, addend: true, activation: 'gelu' } },
];

/** erf from its Maclaurin series, summed in f64. Exact to ~1e-12 for
 *  |x| < 4; past that erf is ±1 to f64 precision. */
function erf(x: number): number {
  if (Math.abs(x) >= 4) return Math.sign(x);
  let term = x;
  let sum = x;
  for (let n = 1; n < 80; n++) {
    term *= (-x * x) / n;
    sum += term / (2 * n + 1);
  }
  return (2 / Math.sqrt(Math.PI)) * sum;
}

const ACTIVATION_REF: Record<SlotAct, (x: number) => number> = {
  tanh: Math.tanh,
  gelu: (x) => 0.5 * x * (1 + erf(x / Math.SQRT2)),
  silu: (x) => x / (1 + Math.exp(-x)),
  relu: (x) => Math.max(x, 0),
  geluTanh: (x) => 0.5 * x * (1 + Math.tanh(Math.sqrt(2 / Math.PI) * (x + 0.044715 * x ** 3))),
};

/** C = activation(A·B + bias) + addend for row-major A [m, k] and B [k, n],
 *  accumulated in f64. */
function matmulRef(
  a: Float32Array,
  b: Float32Array,
  m: number,
  k: number,
  n: number,
  bias?: Float32Array,
  addend?: Float32Array,
  activation?: SlotAct,
): Float64Array {
  const c = new Float64Array(m * n);
  for (let i = 0; i < m; i++) {
    for (let j = 0; j < n; j++) {
      let sum = 0;
      for (let p = 0; p < k; p++) sum += a[i * k + p]! * b[p * n + j]!;
      if (bias) sum += bias[j]!;
      if (activation) sum = ACTIVATION_REF[activation](sum);
      if (addend) sum += addend[i * n + j]!;
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
  epilogue: Epilogue = {},
  tolerance?: { atol: number; rtol: number },
): Promise<void> {
  const { root, ex } = await gpu();
  // The reference sees the f16-rounded inputs the kernel reads.
  const round = dtype === 'f16' ? roundF16 : (x: Float32Array) => x;
  const a = round(randomF32(m * k, 1));
  // With baseRow, B is rows [baseRow, baseRow + k) of a tensor two rows taller.
  const baseRow = epilogue.baseRow ?? 0;
  const bRows = epilogue.baseRow === undefined ? k : baseRow + k + 2;
  const bFull = round(randomF32(bRows * n, 2));
  const b = bFull.subarray(baseRow * n, (baseRow + k) * n);
  const bias = epilogue.bias ? round(randomF32(n, 3)) : undefined;
  const addend = epilogue.addend ? round(randomF32(m * n, 4)) : undefined;
  const meta = (rows: number, cols: number) => ({ ...matrix(rows, cols), dtype });
  const got = await toArray(
    matmul(upload(root, a, meta(m, k)), upload(root, bFull, meta(bRows, n)), {
      bias: bias && upload(root, bias, meta(1, n)),
      addend: addend && upload(root, addend, meta(m, n)),
      activation: epilogue.activation,
      baseRow: epilogue.baseRow,
    }),
    ex,
  );
  // f32 rounding grows with the length of each dot product; f16 output adds
  // 2^-11 relative on top.
  const tol =
    tolerance ?? (dtype === 'f16' ? { atol: 1e-3, rtol: 2e-3 } : { atol: 1e-6 * k, rtol: 1e-4 });
  expectClose(got, matmulRef(a, b, m, k, n, bias, addend, epilogue.activation), tol);
}
