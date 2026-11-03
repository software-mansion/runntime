/** Higher-level ops composed from core primitives rather than their own
 *  kernels — the nn-functional layer. Model-specific ones compose in zoo. */

import { add, matmul, mean, mul, reshape, rsqrt, sub, transpose } from './ops.ts';
import { FLOAT_DTYPES, want2d, type SlotAct } from './ops/shared.ts';
import type { Value } from './value.ts';

/** GroupNorm over [M, C] with channels along the columns: numGroups groups
 *  of C / numGroups adjacent channels, each normalized over its channels and
 *  all M rows, then a per-channel affine. Variance is biased, matching torch.
 *
 *  One group takes the statistics over the whole map. More groups transpose
 *  to [C, M], where a group's channels are adjacent rows, so each group is
 *  one row of a reshape and its statistics are row means. */
export function groupNorm(
  x: Value,
  numGroups: number,
  weight: Value,
  bias: Value,
  eps = 1e-5,
): Value {
  const [m, c] = want2d('groupNorm', x, FLOAT_DTYPES);
  if (!Number.isInteger(numGroups) || numGroups < 1 || c % numGroups !== 0) {
    throw new Error(`groupNorm: numGroups ${numGroups} must divide the ${c} channels`);
  }
  if (numGroups === 1) {
    const g = mean(transpose(mean(x))); // [M,1] → [1,M] → [1,1] global mean
    const xc = sub(x, g); // scalar-broadcast centering
    const vG = mean(transpose(mean(mul(xc, xc)))); // [1,1] global biased variance
    const inv = rsqrt(add(vG, eps)); // [1,1]
    return add(mul(mul(xc, inv), weight), bias); // scalar, then row broadcasts
  }
  const rows =
    numGroups === c ? transpose(x) : reshape(transpose(x), [numGroups, (c / numGroups) * m]);
  const xc = sub(rows, mean(rows)); // [G,1] column broadcast
  const inv = rsqrt(add(mean(mul(xc, xc)), eps)); // [G,1] biased variance
  const y = mul(xc, inv);
  return add(mul(transpose(numGroups === c ? y : reshape(y, [c, m])), weight), bias);
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
