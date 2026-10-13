/** Reductions and normalizations over the last dim. */

import { pending, Value } from '../value.ts';
import { FLOAT_DTYPES, WINDOW_INF, want2d } from './shared.ts';

/** Mean over the last axis: [M,N] to [M,1]. */
export function mean(x: Value): Value {
  const [m, , dtype] = want2d('mean', x, FLOAT_DTYPES);
  return pending({ elems: m, dtype, dims: [m, 1] }, 'mean', [x]);
}

/** Row-wise softmax over the last dim.
 *
 *  With `segments` ([M, 2] f32, a [start, end) range per row) x is M×M
 *  attention scores: row r keeps the keys in its range and window, r
 *  included, and the rest come out 0. Windows default to Infinity. */
export function softmax(
  x: Value,
  opts: { segments?: Value; windowLeft?: number; windowRight?: number } = {},
): Value {
  const [m, n, dtype] = want2d('softmax', x, FLOAT_DTYPES);
  const shape = { elems: m * n, dtype, dims: [m, n] };
  const { segments } = opts;
  if (!segments) return pending(shape, 'softmax', [x], undefined, [0, 0, 0]);
  if (m !== n) throw new Error(`softmax: a masked softmax needs square scores, got [${m}, ${n}]`);
  if (segments.shape.dtype !== 'f32' || segments.shape.elems !== 2 * m) {
    throw new Error(`softmax: segments must be f32 [${m}, 2]`);
  }
  const lower = (name: string, w = Infinity): number => {
    if (w === Infinity) return WINDOW_INF;
    if (!Number.isInteger(w) || w < 0) {
      throw new Error(`softmax: ${name} must be a non-negative integer or Infinity, got ${w}`);
    }
    return Math.min(w, WINDOW_INF);
  };
  const wl = lower('windowLeft', opts.windowLeft);
  const wr = lower('windowRight', opts.windowRight);
  return pending(shape, 'softmax', [x, segments], undefined, [1, wl, wr]);
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
 *  weight. Sums in f32 even for f16 x, so x² never overflows.
 *
 *  `group` normalizes each run of `group` columns on its own (per-head
 *  norms); the weight is then `group` long, shared by every run, or N long
 *  with a slice per run. `residual` and `scale` turn the output into
 *  (residual + norm(x)) · scale. */
export function rmsNorm(
  x: Value,
  weight: Value,
  eps = 1e-6,
  opts: { group?: number; residual?: Value; scale?: number } = {},
): Value {
  const [m, n, dtype] = want2d('rmsNorm', x, FLOAT_DTYPES);
  const { group = n, residual, scale = 1 } = opts;
  if (!Number.isInteger(group) || group < 1 || n % group !== 0) {
    throw new Error(`rmsNorm: group ${group} must divide the row width ${n}`);
  }
  const wl = weight.shape.elems;
  if (weight.shape.dtype !== dtype || (wl !== group && wl !== n)) {
    const lens = group === n ? `${n}` : `${group} or ${n}`;
    throw new Error(
      `rmsNorm: weight must be ${dtype} with ${lens} elems, got (${wl},${weight.shape.dtype})`,
    );
  }
  if (!(eps > 0)) throw new Error(`rmsNorm: eps must be positive, got ${eps}`);
  if (residual && (residual.shape.dtype !== dtype || residual.shape.elems !== m * n)) {
    throw new Error(`rmsNorm: residual must be ${dtype} [${m}, ${n}]`);
  }
  const inputs = residual ? [x, weight, residual] : [x, weight];
  return pending({ elems: m * n, dtype, dims: [m, n] }, 'rmsNorm', inputs, eps, [group, scale]);
}

/** Router top-k plus softmax over the selected logits, lowest index on ties.
 *  Packed [M, 2k]: ids first, then weights, descending by logit. k=4 only. */
export function topk(x: Value, k: number): Value {
  const [m, n] = want2d('topk', x, FLOAT_DTYPES);
  if (k !== 4) throw new Error(`topk: only k=4 is supported, got ${k}`);
  if (n < k) throw new Error(`topk: need at least ${k} columns, got ${n}`);
  return pending({ elems: m * 2 * k, dtype: 'f32', dims: [m, 2 * k] }, 'topk', [x]);
}
