/** Attention ops: banded, causal and cross sdpa with GQA and sinks, plus the
 *  fused rotary embedding. */

import { pending, Value } from '../value.ts';
import { FLOAT_DTYPES, want2d } from './shared.ts';

/** Finite stand-in for an unbounded window side; attrs never carry ±Inf. */
const WINDOW_INF = 0x3fffffff;

/** Scaled-dot-product attention over the band-mask family, with GQA and
 *  optional sinks. kvLen is independent of qLen.
 *
 *  Key j is visible to row i when qPos − windowLeft ≤ j ≤ qPos + windowRight,
 *  qPos = qPosOffset + i. Causal is (Infinity, 0), full and cross
 *  (Infinity, Infinity).
 *
 *  Optional `segments` clips each row to its own key range, which is how
 *  sequences packed into one tensor avoid attending across each other.
 *  `maxSegment` lets the evaluator size its key split from the longest segment
 *  rather than the whole pack. */
export function sdpa(
  q: Value,
  k: Value,
  v: Value,
  opts: {
    qHeads: number;
    kvHeads: number;
    headDim: number;
    windowLeft: number;
    windowRight: number;
    qPosOffset?: number;
    sinks?: Value;
    segments?: Value;
    maxSegment?: number;
  },
): Value {
  const { qHeads, kvHeads, headDim, sinks, segments } = opts;
  const qPosOffset = opts.qPosOffset ?? 0;
  const [qLen, qw, dtype] = want2d('sdpa', q, FLOAT_DTYPES);
  if (qw !== qHeads * headDim)
    throw new Error(`sdpa: q cols ${qw} != qHeads*headDim ${qHeads * headDim}`);
  if (kvHeads <= 0 || qHeads % kvHeads !== 0) {
    throw new Error(`sdpa: qHeads ${qHeads} not divisible by kvHeads ${kvHeads}`);
  }
  if (headDim > 128) throw new Error(`sdpa: headDim ${headDim} exceeds the kernel's max (128)`);
  const lowerWindow = (name: string, w: number): number => {
    if (w === Infinity) return WINDOW_INF;
    if (!Number.isInteger(w) || w < 0) {
      throw new Error(`sdpa: ${name} ${w} must be a non-negative integer or Infinity`);
    }
    return Math.min(w, WINDOW_INF);
  };
  const wl = lowerWindow('windowLeft', opts.windowLeft);
  const wr = lowerWindow('windowRight', opts.windowRight);
  if (!Number.isInteger(qPosOffset) || qPosOffset < 0) {
    throw new Error(`sdpa: qPosOffset ${qPosOffset} must be a non-negative integer`);
  }
  const kvw = kvHeads * headDim;
  const [kvLen, kw, kDtype] = want2d('sdpa.k', k, FLOAT_DTYPES);
  if (kw !== kvw) throw new Error(`sdpa: k cols ${kw} != kvHeads*headDim ${kvw}`);
  const [vt, vw, vDtype] = want2d('sdpa.v', v, FLOAT_DTYPES);
  // One element type per kernel. segments stays f32: it carries integer key
  // ranges that f16 stops representing exactly past 2048.
  if (kDtype !== dtype || vDtype !== dtype) {
    throw new Error(`sdpa: q, k and v must share a dtype, got ${dtype}/${kDtype}/${vDtype}`);
  }
  if (vt !== kvLen) throw new Error(`sdpa: v rows ${vt} != k rows ${kvLen}`);
  if (vw !== kvw) throw new Error(`sdpa: v cols ${vw} != kvHeads*headDim ${kvw}`);
  // Under a bounded right window, rows past kvLen mean a forgotten offset
  // rather than a mask. Unbounded-right is exempt.
  if (wr !== WINDOW_INF && qPosOffset + qLen > kvLen) {
    throw new Error(`sdpa: qPosOffset+qLen ${qPosOffset + qLen} exceeds key rows kvLen ${kvLen}`);
  }
  if (sinks && (sinks.shape.dtype !== 'f32' || sinks.shape.elems !== qHeads)) {
    throw new Error(`sdpa: sinks must be f32 with ${qHeads} elems, got ${sinks.shape.elems}`);
  }
  if (segments) {
    if (segments.shape.dtype !== 'f32' || segments.shape.elems !== 2 * qLen) {
      throw new Error(
        `sdpa: segments must be f32 [qLen, 2] = ${2 * qLen} elems, got (${segments.shape.elems},${segments.shape.dtype})`,
      );
    }
    // The qLen = 1 decode pair has no segment support, and a one-row pack has
    // nothing to mask.
    if (qLen === 1) throw new Error('sdpa: segments need qLen > 1 (decode has no segment route)');
  }
  const maxSegment = opts.maxSegment ?? 0;
  if (maxSegment !== 0) {
    if (!segments) throw new Error('sdpa: maxSegment needs segments');
    if (!Number.isInteger(maxSegment) || maxSegment < 1) {
      throw new Error(`sdpa: maxSegment ${maxSegment} must be a positive integer`);
    }
  }
  return pending(
    { elems: qLen * qw, dtype, dims: [qLen, qw] },
    'attn',
    [q, k, v, ...(sinks ? [sinks] : []), ...(segments ? [segments] : [])],
    undefined,
    [qHeads, kvHeads, headDim, wl, wr, qPosOffset, sinks ? 1 : 0, segments ? 1 : 0, maxSegment],
  );
}

/** sdpa over one fused qkv tensor read in place, with q, k and v as column
 *  blocks, so the three slice copies never run. Self-attention only, and no
 *  qLen = 1 route. */
export function sdpaPacked(
  qkv: Value,
  opts: {
    qHeads: number;
    kvHeads: number;
    headDim: number;
    windowLeft: number;
    windowRight: number;
    qPosOffset?: number;
    sinks?: Value;
    segments?: Value;
    maxSegment?: number;
  },
): Value {
  const { qHeads, kvHeads, headDim, sinks, segments } = opts;
  const qPosOffset = opts.qPosOffset ?? 0;
  const [t, w, dtype] = want2d('sdpaPacked', qkv, FLOAT_DTYPES);
  const qw = qHeads * headDim;
  const packedW = (qHeads + 2 * kvHeads) * headDim;
  if (w !== packedW) {
    throw new Error(`sdpaPacked: qkv cols ${w} != (qHeads + 2*kvHeads)*headDim = ${packedW}`);
  }
  if (kvHeads <= 0 || qHeads % kvHeads !== 0) {
    throw new Error(`sdpaPacked: qHeads ${qHeads} not divisible by kvHeads ${kvHeads}`);
  }
  if (headDim > 128) {
    throw new Error(`sdpaPacked: headDim ${headDim} exceeds the kernel's max (128)`);
  }
  if (t === 1) throw new Error('sdpaPacked: qLen must be > 1 (decode has no packed route)');
  const lowerWindow = (name: string, win: number): number => {
    if (win === Infinity) return WINDOW_INF;
    if (!Number.isInteger(win) || win < 0) {
      throw new Error(`sdpaPacked: ${name} ${win} must be a non-negative integer or Infinity`);
    }
    return Math.min(win, WINDOW_INF);
  };
  const wl = lowerWindow('windowLeft', opts.windowLeft);
  const wr = lowerWindow('windowRight', opts.windowRight);
  if (!Number.isInteger(qPosOffset) || qPosOffset < 0) {
    throw new Error(`sdpaPacked: qPosOffset ${qPosOffset} must be a non-negative integer`);
  }
  if (wr !== WINDOW_INF && qPosOffset + t > t) {
    throw new Error(`sdpaPacked: qPosOffset+qLen ${qPosOffset + t} exceeds key rows kvLen ${t}`);
  }
  if (sinks && (sinks.shape.dtype !== 'f32' || sinks.shape.elems !== qHeads)) {
    throw new Error(`sdpaPacked: sinks must be f32 with ${qHeads} elems, got ${sinks.shape.elems}`);
  }
  if (segments && (segments.shape.dtype !== 'f32' || segments.shape.elems !== 2 * t)) {
    throw new Error(
      `sdpaPacked: segments must be f32 [qLen, 2] = ${2 * t} elems, got (${segments.shape.elems},${segments.shape.dtype})`,
    );
  }
  const maxSegment = opts.maxSegment ?? 0;
  if (maxSegment !== 0) {
    if (!segments) throw new Error('sdpaPacked: maxSegment needs segments');
    if (!Number.isInteger(maxSegment) || maxSegment < 1) {
      throw new Error(`sdpaPacked: maxSegment ${maxSegment} must be a positive integer`);
    }
  }
  return pending(
    { elems: t * qw, dtype, dims: [t, qw] },
    'attn',
    [qkv, ...(sinks ? [sinks] : []), ...(segments ? [segments] : [])],
    undefined,
    [qHeads, kvHeads, headDim, wl, wr, qPosOffset, sinks ? 1 : 0, segments ? 1 : 0, maxSegment, 1],
  );
}

/** Rotary position embedding over every head section of x [T, W] in one
 *  kernel. Rotation is interleaved-pair, and cos/sin are pair-duplicated tables
 *  with any scale pre-folded.
 *
 *  srcStart and width optionally read a slice of a wider slab — the q or k
 *  section of a fused qkv, say — and rope it in the same dispatch. */
export function rope(
  x: Value,
  cos: Value,
  sin: Value,
  opts: { headDim: number; srcStart?: number; width?: number },
): Value {
  const { headDim } = opts;
  const [t, xw, dtype] = want2d('rope', x, FLOAT_DTYPES);
  const srcStart = opts.srcStart ?? 0;
  const w = opts.width ?? xw;
  if (!Number.isInteger(headDim) || headDim <= 0 || headDim % 2 !== 0) {
    throw new Error(`rope: headDim ${headDim} must be a positive even integer`);
  }
  if (!Number.isInteger(srcStart) || srcStart < 0 || srcStart % 2 !== 0) {
    throw new Error(`rope: srcStart ${srcStart} must be a non-negative even integer`);
  }
  if (!Number.isInteger(w) || w <= 0 || srcStart + w > xw) {
    throw new Error(`rope: window [${srcStart}, ${srcStart + w}) exceeds ${xw} input cols`);
  }
  if (w % headDim !== 0)
    throw new Error(`rope: width ${w} must be a multiple of headDim ${headDim}`);
  for (const [name, v] of [['cos', cos] as const, ['sin', sin] as const]) {
    if (v.shape.dtype !== 'f32' || v.shape.elems !== t * headDim) {
      throw new Error(
        `rope: ${name} must be f32 [T,headDim] = ${t * headDim} elems, got ${v.shape.elems}`,
      );
    }
  }
  return pending({ elems: t * w, dtype, dims: [t, w] }, 'rope', [x, cos, sin], undefined, [
    headDim,
    srcStart,
    w,
  ]);
}
