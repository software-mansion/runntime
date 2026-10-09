import type { SlotAct } from './ops/shared.ts';
/** Higher-level ops composed from core primitives rather than their own
 *  kernels — the nn-functional layer. Model-specific ones compose in zoo. */

import { add, matmul, mean, mul, rsqrt, sub, transpose } from './ops.ts';
import { rmsNormFused } from './ops/reduce.ts';
import type { Value } from './value.ts';

/** RMSNorm(x[M,N], weight[N], eps) = x * rsqrt(mean(x², -1) + eps) * weight. */
export function rmsNorm(x: Value, weight: Value, eps = 1e-6): Value {
  return rmsNormFused(x, weight, eps);
}

/** GroupNorm restricted to numGroups = 1: statistics are global over the whole
 *  [M,N] map, not per-row, and the affine is per-column. Variance is biased,
 *  matching torch. numGroups > 1 is rejected loudly until a model needs it. */
export function groupNorm(
  x: Value,
  numGroups: number,
  weight: Value,
  bias: Value,
  eps = 1e-5,
): Value {
  if (numGroups !== 1) {
    throw new Error(`groupNorm: only numGroups = 1 is supported, got ${numGroups}`);
  }
  const g = mean(transpose(mean(x))); // [M,1] → [1,M] → [1,1] global mean
  const xc = sub(x, g); // scalar-broadcast centering
  const vG = mean(transpose(mean(mul(xc, xc)))); // [1,1] global biased variance
  const inv = rsqrt(add(vG, eps)); // [1,1]
  return add(mul(mul(xc, inv), weight), bias); // scalar, then row broadcasts
}

/** Linear(x[M,K], weight[K,N], bias[N], addend[M,N]) = x @ weight (+ bias)
 *  (+ addend) — one dispatch via the matmul epilogue. Omitting `bias` is
 *  torch's Linear(bias=False); `addend` folds a residual add into the same
 *  kernel (pre-norm transformer sublayers end in exactly this shape). */
export function linear(
  x: Value,
  weight: Value,
  bias?: Value,
  addend?: Value,
  activation?: SlotAct,
): Value {
  return matmul(x, weight, { bias, addend, activation });
}
