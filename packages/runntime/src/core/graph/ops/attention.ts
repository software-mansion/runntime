/** Attention ops: banded, causal and cross sdpa with GQA and sinks, local
 *  attention with a distance bias, and the fused rotary embedding. */

import type { KeyBias } from '../../kernels/attention/attn.ts';
import { RunntimeError } from '../../error.ts';
import { ATTN_LOCAL_MAX_WINDOW } from '../../kernels/attention/attnLocal.ts';
import { ROPE_LAYOUTS, type RopeLayout } from '../../kernels/attention/rope.ts';
import { pending, Value } from '../value.ts';
import { FLOAT_DTYPES, WINDOW_INF, want2d } from './shared.ts';

/** Widest head sdpa takes with more than one query row. */
const MAX_HEAD_DIM = 512;
/** Widest head the decode kernels (one query row) take. */
const MAX_DECODE_HEAD_DIM = 128;

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
 *  rather than the whole pack.
 *
 *  `scale` multiplies q·k before the softmax: 1/√headDim by default, as in
 *  PyTorch's `scaled_dot_product_attention`. Pass the model's own scale (1
 *  for q and k that are already normed or scaled). Sinks are not scaled.
 *
 *  Optional `keyBias` adds `value` to the scores of keys [start, end) in every
 *  row and head, after the scale. */
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
    scale?: number;
    sinks?: Value;
    segments?: Value;
    maxSegment?: number;
    keyBias?: KeyBias;
  },
): Value {
  const { qHeads, kvHeads, headDim, sinks, segments, keyBias } = opts;
  const qPosOffset = opts.qPosOffset ?? 0;
  const scale = opts.scale ?? 1 / Math.sqrt(headDim);
  const [qLen, qw, dtype] = want2d('sdpa', q, FLOAT_DTYPES);
  if (qw !== qHeads * headDim)
    throw new Error(`sdpa: q cols ${qw} != qHeads*headDim ${qHeads * headDim}`);
  if (kvHeads <= 0 || qHeads % kvHeads !== 0) {
    throw new Error(`sdpa: qHeads ${qHeads} not divisible by kvHeads ${kvHeads}`);
  }
  // Prefill kernels size their registers per head size; decode keeps a
  // fixed 128-wide accumulator.
  const maxHeadDim = qLen === 1 ? MAX_DECODE_HEAD_DIM : MAX_HEAD_DIM;
  if (headDim > maxHeadDim) {
    throw new Error(`sdpa: headDim ${headDim} exceeds the kernel's max (${maxHeadDim})`);
  }
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
  if (!Number.isFinite(scale)) throw new Error(`sdpa: scale ${scale} must be finite`);
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
  if (keyBias) {
    const { start, end, value } = keyBias;
    if (
      !Number.isInteger(start) ||
      !Number.isInteger(end) ||
      start < 0 ||
      start > end ||
      end > kvLen
    ) {
      throw new Error(`sdpa: keyBias keys [${start}, ${end}) must lie within the ${kvLen} keys`);
    }
    if (!Number.isFinite(value)) throw new Error(`sdpa: keyBias value ${value} must be finite`);
    // The qLen = 1 decode kernels have no key bias, as with segments.
    if (qLen === 1) throw new Error('sdpa: keyBias needs qLen > 1 (decode has no key bias route)');
  }
  return pending(
    { elems: qLen * qw, dtype, dims: [qLen, qw] },
    'attn',
    [q, k, v, ...(sinks ? [sinks] : []), ...(segments ? [segments] : [])],
    undefined,
    [
      qHeads,
      kvHeads,
      headDim,
      wl,
      wr,
      qPosOffset,
      sinks ? 1 : 0,
      segments ? 1 : 0,
      maxSegment,
      0, // q, k and v are separate tensors
      scale,
      ...(keyBias ? [1, keyBias.start, keyBias.end, keyBias.value] : []),
    ],
  );
}

/** sdpa over one fused qkv tensor read in place, with q, k and v as column
 *  blocks, so the three slice copies never run. Self-attention only, and no
 *  qLen = 1 route. `scale` is sdpa's: 1/√headDim by default. */
export function sdpaPacked(
  qkv: Value,
  opts: {
    qHeads: number;
    kvHeads: number;
    headDim: number;
    windowLeft: number;
    windowRight: number;
    qPosOffset?: number;
    scale?: number;
    sinks?: Value;
    segments?: Value;
    maxSegment?: number;
  },
): Value {
  const { qHeads, kvHeads, headDim, sinks, segments } = opts;
  const qPosOffset = opts.qPosOffset ?? 0;
  const scale = opts.scale ?? 1 / Math.sqrt(headDim);
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
  if (!Number.isFinite(scale)) throw new Error(`sdpaPacked: scale ${scale} must be finite`);
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
    [
      qHeads,
      kvHeads,
      headDim,
      wl,
      wr,
      qPosOffset,
      sinks ? 1 : 0,
      segments ? 1 : 0,
      maxSegment,
      1, // q, k and v are column blocks of qkv
      scale,
    ],
  );
}

/** Rotary position embedding over every head section of x [T, W] in one
 *  kernel. `layout` names the columns of a head that rotate together, as the
 *  checkpoint's q and k have them, and has no default: the wrong one gives
 *  wrong output without an error.
 *  - 'half': columns (p, p + headDim/2), HF's `rotate_half`.
 *  - 'interleaved': columns (2p, 2p + 1).
 *
 *  `ropesPerHead` (default 1) is for a head that holds several ropes side by
 *  side, each over its own headDim / ropesPerHead columns, like the x and y
 *  halves of a 2D rope: 'half' then pairs (p, p + headDim / ropesPerHead / 2)
 *  within each rope's columns. 'interleaved' pairs neighbours however a head
 *  is split, so its node carries 1.
 *
 *  cos and sin are f32 [T, headDim], one entry per column of a head, both
 *  columns of a pair holding the pair's value: `cat(freqs, freqs)` for
 *  'half', each frequency twice in a row for 'interleaved'. The score scale
 *  is not in the tables: sdpa applies it.
 *
 *  A pair with cos 1 and sin 0 passes through. The pairs are fixed by the
 *  layout, so what that expresses differs:
 *  - 'interleaved': identity on pairs k … headDim/2 − 1 leaves columns
 *    [2k, headDim) as they are. That is a partial rotary over the first 2k
 *    columns (Moonshine).
 *  - 'half': identity on those pairs leaves columns [k, headDim/2) and
 *    [headDim/2 + k, headDim). A checkpoint that applies rotate_half within
 *    its first rotaryDim columns pairs (p, p + rotaryDim/2), which these
 *    tables cannot express.
 *
 *  srcStart and width optionally read a slice of a wider slab — the q or k
 *  section of a fused qkv, say — and rope it in the same dispatch.
 *
 *  Throws an INVALID_ARGUMENT RunntimeError when layout is missing or not
 *  one of the two, or when ropesPerHead does not split a head into ropes of
 *  an even number of columns. */
export function rope(
  x: Value,
  cos: Value,
  sin: Value,
  opts: {
    headDim: number;
    layout: RopeLayout;
    ropesPerHead?: number;
    srcStart?: number;
    width?: number;
  },
): Value {
  const { headDim, ropesPerHead = 1 } = opts;
  const layout = ROPE_LAYOUTS.indexOf(opts.layout);
  if (layout < 0) {
    throw new RunntimeError(
      'INVALID_ARGUMENT',
      `rope: layout must be 'half' or 'interleaved', got ${String(opts.layout)}`,
    );
  }
  const [t, xw, dtype] = want2d('rope', x, FLOAT_DTYPES);
  const srcStart = opts.srcStart ?? 0;
  const w = opts.width ?? xw;
  if (!Number.isInteger(headDim) || headDim <= 0 || headDim % 2 !== 0) {
    throw new Error(`rope: headDim ${headDim} must be a positive even integer`);
  }
  if (!Number.isInteger(ropesPerHead) || ropesPerHead < 1 || headDim % (2 * ropesPerHead) !== 0) {
    throw new RunntimeError(
      'INVALID_ARGUMENT',
      `rope: ropesPerHead ${ropesPerHead} must split headDim ${headDim} into ropes of an even number of columns`,
    );
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
    layout,
    opts.layout === 'half' ? ropesPerHead : 1,
  ]);
}

/** Local attention over a fused qkv slab [T, 3·heads·headDim]: row t sees
 *  keys t − window + 1 … t. The score of the key j rows back is
 *  cap · tanh((q · qScale) · (k · kScale + rel[j]) / cap), with rel
 *  [window, heads·headDim] one f32 bias row per distance and qScale
 *  [headDim] f32 per column of q. Returns [T, heads·headDim]. */
export function localAttention(
  qkv: Value,
  rel: Value,
  qScale: Value,
  opts: { heads: number; headDim: number; window: number; kScale: number; softcap: number },
): Value {
  const { heads, headDim, window, kScale, softcap } = opts;
  const [t, w, dtype] = want2d('localAttention', qkv, FLOAT_DTYPES);
  if (!Number.isInteger(heads) || heads < 1 || !Number.isInteger(headDim) || headDim % 4 !== 0) {
    throw new Error(
      `localAttention: heads ${heads} must be a positive integer and headDim ${headDim} a multiple of 4`,
    );
  }
  const width = heads * headDim;
  if (w !== 3 * width) {
    throw new Error(`localAttention: qkv must have 3·heads·headDim = ${3 * width} cols, got ${w}`);
  }
  if (!Number.isInteger(window) || window < 1 || window > ATTN_LOCAL_MAX_WINDOW) {
    throw new Error(
      `localAttention: window ${window} must be an integer from 1 to ${ATTN_LOCAL_MAX_WINDOW}`,
    );
  }
  if (rel.shape.dtype !== 'f32' || rel.shape.elems !== window * width) {
    throw new Error(`localAttention: rel must be f32 [${window}, ${width}]`);
  }
  if (qScale.shape.dtype !== 'f32' || qScale.shape.elems !== headDim) {
    throw new Error(`localAttention: qScale must be f32 [${headDim}]`);
  }
  if (!(softcap > 0) || !Number.isFinite(kScale)) {
    throw new Error(
      `localAttention: softcap must be positive and kScale finite, got ${softcap}, ${kScale}`,
    );
  }
  return pending(
    { elems: t * width, dtype, dims: [t, width] },
    'localAttention',
    [qkv, rel, qScale],
    undefined,
    [heads, headDim, window, kScale, softcap],
  );
}
