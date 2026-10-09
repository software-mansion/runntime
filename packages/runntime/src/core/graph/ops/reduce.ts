/** Reductions and normalizations over the last dim. */

import { pending, Value } from '../value.ts';
import { FLOAT_DTYPES, want2d } from './shared.ts';

/** Mean over the last axis: [M,N] to [M,1]. */
export function mean(x: Value): Value {
  const [m, , dtype] = want2d('mean', x, FLOAT_DTYPES);
  return pending({ elems: m, dtype, dims: [m, 1] }, 'mean', [x]);
}

/** Row-wise softmax over the last dim. */
export function softmax(x: Value): Value {
  const [m, n, dtype] = want2d('softmax', x, FLOAT_DTYPES);
  return pending({ elems: m * n, dtype, dims: [m, n] }, 'softmax', [x]);
}

/** Fused LayerNorm over the last dim, with torch's biased 1/N variance. eps
 *  rides the uniform, so it never keys a pipeline. */
export function layerNorm(x: Value, weight: Value, eps = 1e-5, bias?: Value): Value {
  const [m, n, dtype] = want2d('layerNorm', x, FLOAT_DTYPES);
  if (weight.shape.dtype !== dtype || weight.shape.elems !== n) {
    throw new Error(
      `layerNorm: weight must be ${dtype} [N] = ${n} elems, got (${weight.shape.elems},${weight.shape.dtype})`,
    );
  }
  if (bias !== undefined && (bias.shape.dtype !== dtype || bias.shape.elems !== n)) {
    throw new Error(
      `layerNorm: bias must be ${dtype} [N] = ${n} elems, got (${bias.shape.elems},${bias.shape.dtype})`,
    );
  }
  if (!(eps > 0)) throw new Error(`layerNorm: eps must be positive, got ${eps}`);
  const inputs = bias !== undefined ? [x, weight, bias] : [x, weight];
  return pending({ elems: m * n, dtype, dims: [m, n] }, 'layerNorm', inputs, eps);
}

/** RMSNorm over the last dim in one dispatch: x · rsqrt(mean(x²) + eps) ·
 *  weight. Sums in f32 even for f16 x, so x² never overflows. */
export function rmsNorm(x: Value, weight: Value, eps = 1e-6): Value {
  const [m, n, dtype] = want2d('rmsNorm', x, FLOAT_DTYPES);
  if (weight.shape.dtype !== dtype || weight.shape.elems !== n) {
    throw new Error(
      `rmsNorm: weight must be ${dtype} [N] = ${n} elems, got (${weight.shape.elems},${weight.shape.dtype})`,
    );
  }
  if (!(eps > 0)) throw new Error(`rmsNorm: eps must be positive, got ${eps}`);
  return pending({ elems: m * n, dtype, dims: [m, n] }, 'rmsNorm', [x, weight], eps);
}

/** Router top-k plus softmax over the selected logits, lowest index on ties.
 *  Packed [M, 2k]: ids first, then weights, descending by logit. k=4 only. */
export function topk(x: Value, k: number): Value {
  const [m, n] = want2d('topk', x, FLOAT_DTYPES);
  if (k !== 4) throw new Error(`topk: only k=4 is supported, got ${k}`);
  if (n < k) throw new Error(`topk: need at least ${k} columns, got ${n}`);
  return pending({ elems: m * 2 * k, dtype: 'f32', dims: [m, 2 * k] }, 'topk', [x]);
}
