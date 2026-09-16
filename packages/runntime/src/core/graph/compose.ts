import type { SlotAct } from './ops/shared.ts';
/** Higher-level ops composed from core primitives rather than their own
 *  kernels — the nn-functional layer. Model-specific ones compose in zoo. */

import { add, astype, matmul, mean, meanSquare, mul, rsqrt, sub, transpose } from './ops.ts';
import type { Value } from './value.ts';

/** RMSNorm(x[M,N], weight[N], eps) = x * rsqrt(mean(x², -1) + eps) * weight. */
export function rmsNorm(x: Value, weight: Value, eps = 1e-6): Value {
  const ms = meanSquare(x); // [M,1] f32 — squares AND holds the sum in f32
  // rsqrt stays in f32 too; only the finished scale factor narrows, and that
  // one is small enough to survive (rsqrt(2.9e7) = 1.9e-4).
  const denom = astype(rsqrt(add(ms, eps)), x.shape.dtype === 'f16' ? 'f16' : 'f32');
  return mul(mul(x, denom), weight); // col-broadcast then row-broadcast
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
