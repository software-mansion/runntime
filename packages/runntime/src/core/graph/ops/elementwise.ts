/** Elementwise ops: add, mul and sub with row and column broadcasting and
 *  their plain-number overloads, unary activations, clamp, and the fused
 *  SwiGLU variants. */

import { pending, Value, type ValueMeta } from '../value.ts';
import { FLOAT_DTYPES, want2d } from './shared.ts';

function dimsMatch(a?: readonly number[], b?: readonly number[]): boolean {
  return !!a && !!b && a.length === b.length && a.every((v, i) => v === b[i]);
}

/** Describes how `a op b` broadcasts, or null if the pair is incompatible.
 *  `mode` is 0 elementwise, 1 row, 2 column, 3 scalar.
 *
 *  Decided from dims where available: element counts alone cannot tell [M,1]
 *  from [N] on a square matrix. */
function broadcastPlan(
  x: Value,
  y: Value,
): { full: Value; other: Value; shape: ValueMeta; mode: 0 | 1 | 2 | 3 } | null {
  if (x.shape.dtype !== y.shape.dtype) return null;
  // hwc4 and chw can share an element count with different byte order, and
  // pairing them would silently mix pixels.
  if (x.shape.layout !== y.shape.layout) return null;
  if (x.shape.elems === y.shape.elems) {
    // [5,H,W] and [8,H,W] share padded elems, so dims must match too.
    if (x.shape.layout && !dimsMatch(x.shape.dims, y.shape.dims)) return null;
    return { full: x, other: y, shape: { ...x.shape }, mode: 0 };
  }
  // Broadcasting is row-major arithmetic, so it never applies to hwc4.
  if (x.shape.layout !== undefined) return null;
  const full = x.shape.elems > y.shape.elems ? x : y;
  const other = full === x ? y : x;
  if (other.shape.elems === 1) return { full, other, shape: { ...full.shape }, mode: 3 };
  const fd = full.shape.dims;
  if (!fd || fd.length !== 2) return null;
  const [m, n] = [fd[0]!, fd[1]!];
  const od = other.shape.dims;
  if (od) {
    if (od.length === 2 && od[0] === m && od[1] === 1) {
      return { full, other, shape: { ...full.shape }, mode: 2 }; // col [M,1]
    }
    if ((od.length === 1 && od[0] === n) || (od.length === 2 && od[0] === 1 && od[1] === n)) {
      return { full, other, shape: { ...full.shape }, mode: 1 }; // row [N]/[1,N]
    }
  }
  // Fallback for values without dims: infer from element counts.
  if (other.shape.elems === n) return { full, other, shape: { ...full.shape }, mode: 1 }; // row [N]
  if (other.shape.elems === m) return { full, other, shape: { ...full.shape }, mode: 2 }; // col [M]/[M,1]
  return null;
}

/** Shared binary-elementwise builder. The full [M,N] operand goes first, so
 *  the kernel's `a` is always the full tensor. */
function elementwise(name: 'add' | 'sub' | 'mul' | 'swiglu', a: Value, b: Value): Value {
  // The equal-dtype check below would happily pair two quantized operands.
  if (a.shape.dtype === 'quantW' || b.shape.dtype === 'quantW') {
    throw new Error(
      `${name}: quantW Values are weight-only — only matmul/matmulGather consume them`,
    );
  }
  const plan = broadcastPlan(a, b);
  if (!plan) {
    throw new Error(
      `${name}: incompatible shapes — a=(${a.shape.elems},${a.shape.dtype}) b=(${b.shape.elems},${b.shape.dtype})`,
    );
  }
  return pending(plan.shape, name, [plan.full, plan.other], undefined, [plan.mode]);
}

/** Elementwise add. A number `b` folds into the dispatch with no upload. */
export function add(a: Value, b: Value | number): Value {
  if (typeof b === 'number') return addScalar(a, b);
  return elementwise('add', a, b);
}

/** Elementwise multiply. A number `b` folds into the dispatch. */
export function mul(a: Value, b: Value | number): Value {
  if (typeof b === 'number') return mulScalar(a, b);
  return elementwise('mul', a, b);
}

/** Elementwise subtract, with b broadcast as a row or column.
 *
 *  Non-commutative, so the full operand must be `a`: the shared builder orders
 *  inputs full-first and would otherwise flip the sign. */
export function sub(a: Value, b: Value | number): Value {
  if (typeof b === 'number') return addScalar(a, -b);
  if (a.shape.dtype !== b.shape.dtype) {
    throw new Error(`sub: dtype mismatch — ${a.shape.dtype} − ${b.shape.dtype}`);
  }
  if (b.shape.elems > a.shape.elems) {
    throw new Error(
      `sub: a must be the full operand — b (${b.shape.elems} elems) exceeds a (${a.shape.elems})`,
    );
  }
  return elementwise('sub', a, b);
}

/** Ops where f(0) is not 0 refuse hwc4: the padded lanes would stop being
 *  padding. */
function refuseHwc4(name: string, x: Value): void {
  if (x.shape.layout === 'hwc4') {
    throw new Error(`${name}: refuses hwc4 input — f(0) !== 0 corrupts the zero-padded lanes`);
  }
}

/** out = x + c, elementwise. add()'s scalar path. */
function addScalar(x: Value, c: number): Value {
  if (c !== 0) refuseHwc4('add', x);
  return pending({ ...x.shape }, 'addScalar', [x], c);
}

/** out = x · c, elementwise. mul()'s scalar path. */
function mulScalar(x: Value, c: number): Value {
  return pending({ ...x.shape }, 'mulScalar', [x], c);
}

/** Fused clamped SwiGLU: g·σ(1.702·g)·(l+1), g = min(glu, 7),
 *  l = clamp(lin, ±7). glu and lin must share shape and dtype. */
export function swiglu(glu: Value, lin: Value): Value {
  if (glu.shape.elems !== lin.shape.elems || glu.shape.dtype !== lin.shape.dtype) {
    throw new Error(
      `swiglu: glu/lin must match — glu=(${glu.shape.elems},${glu.shape.dtype}) lin=(${lin.shape.elems},${lin.shape.dtype})`,
    );
  }
  if (glu.shape.dtype !== 'f32' && glu.shape.dtype !== 'f16') {
    throw new Error(`swiglu: needs a float dtype, got ${glu.shape.dtype}`);
  }
  return pending({ ...glu.shape }, 'swiglu', [glu, lin], undefined, [0]);
}

/** Fused SwiGLU chunk: x [t, 2F] to [t, F], hidden first and gate second in
 *  HF's order. Unclamped; `swiglu` is the clamped variant. */
export function swigluChunk(x: Value): Value {
  const [t, w, dtype] = want2d('swigluChunk', x, FLOAT_DTYPES);
  if (w % 2 !== 0) throw new Error(`swigluChunk: cols ${w} must be even (hidden|gate halves)`);
  const half = w / 2;
  return pending({ elems: t * half, dtype, dims: [t, half] }, 'swigluChunk', [x]);
}

/** out = 1/sqrt(x), elementwise. */
export function rsqrt(x: Value): Value {
  refuseHwc4('rsqrt', x);
  return pending({ ...x.shape }, 'rsqrt', [x]);
}

/** out = 1/(1+e^(−x)), elementwise. */
export function sigmoid(x: Value): Value {
  refuseHwc4('sigmoid', x);
  return pending({ ...x.shape }, 'sigmoid', [x]);
}

/** out = tanh(x), elementwise. */
export function tanh(x: Value): Value {
  return pending({ ...x.shape }, 'tanh', [x]);
}

/** out = x·Φ(x), elementwise. The exact-erf GELU. */
export function gelu(x: Value): Value {
  return pending({ ...x.shape }, 'gelu', [x]);
}

/** out = x·σ(x), elementwise. */
export function silu(x: Value): Value {
  return pending({ ...x.shape }, 'silu', [x]);
}

/** out = asinh(x) = ln(x + √(x²+1)), elementwise. */
export function asinh(x: Value): Value {
  return pending({ ...x.shape }, 'asinh', [x]);
}

/** out = min(max(x, lo), hi). Pass ±Infinity for a one-sided clamp. */
export function clamp(x: Value, lo: number, hi: number): Value {
  if (!(lo <= hi)) throw new Error(`clamp: lo (${lo}) must be <= hi (${hi})`);
  if (lo > 0 || hi < 0) refuseHwc4('clamp', x);
  return pending({ ...x.shape }, 'clampScalar', [x], undefined, [lo, hi]);
}
