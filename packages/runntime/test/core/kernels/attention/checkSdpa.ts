/** Shared by the attention kernel suites: runs one sdpa on the GPU and checks
 *  it against a plain CPU loop. */
import type { TgpuRoot } from 'typegpu';
import { toArray, type RunntimeExecutor } from '../../../../src/core/gpu/eval.ts';
import { sdpa, sdpaPacked } from '../../../../src/core/graph/ops.ts';
import { matrix } from '../../../../src/core/graph/value.ts';
import { expectClose, randomF32, roundF16, upload } from '../../../gpu.ts';

export interface SdpaCase {
  qLen: number;
  kvLen: number;
  qHeads: number;
  kvHeads: number;
  headDim: number;
  /** Keys before and after the query's position it sees. Default Infinity. */
  windowLeft?: number;
  windowRight?: number;
  qPosOffset?: number;
  /** A sink logit per query head. */
  sinks?: boolean;
  /** Packs two sequences: rows before qLen / 2 see the keys before
   *  kvLen / 2, the rest see the keys after. */
  segments?: boolean;
  maxSegment?: number;
  /** q, k and v as column blocks of one [T, (qHeads + 2·kvHeads)·headDim]
   *  tensor, through sdpaPacked. */
  packed?: boolean;
}

/** Every mask and input form, for a square self-attention shape with 4
 *  query heads and 2 kv heads. Every prefill route runs this list. */
export const SDPA_OPTIONS: readonly { name: string; opts: Partial<SdpaCase> }[] = [
  { name: 'full', opts: {} },
  { name: 'causal', opts: { windowRight: 0 } },
  { name: 'a sliding window', opts: { windowLeft: 3, windowRight: 0 } },
  { name: 'a two-sided window', opts: { windowLeft: 2, windowRight: 1 } },
  { name: 'sinks', opts: { sinks: true, windowRight: 0 } },
  { name: 'segments', opts: { segments: true } },
  { name: 'segments and causal', opts: { segments: true, windowRight: 0 } },
  { name: 'packed qkv', opts: { packed: true, windowRight: 0 } },
  { name: 'packed qkv, sinks and segments', opts: { packed: true, sinks: true, segments: true } },
  { name: 'every query head on one kv head', opts: { kvHeads: 1, windowRight: 0 } },
  { name: 'as many kv heads as query heads', opts: { kvHeads: 4 } },
];

/** softmax(q·kᵀ) v per query row and head, in f64, over the keys the window
 *  and the row's segment leave visible. q and k arrive pre-scaled. A sink
 *  adds exp(sink · ln 2) to the denominator alone. A row that sees no key
 *  comes out 0. */
export function sdpaRef(
  q: Float32Array,
  k: Float32Array,
  v: Float32Array,
  c: Required<Pick<SdpaCase, 'qLen' | 'kvLen' | 'qHeads' | 'kvHeads' | 'headDim'>> & {
    windowLeft: number;
    windowRight: number;
    qPosOffset: number;
    sinks?: Float32Array;
    segs?: Float32Array;
  },
): Float64Array {
  const { qLen, kvLen, qHeads, kvHeads, headDim: hd } = c;
  const out = new Float64Array(qLen * qHeads * hd);
  const group = qHeads / kvHeads;
  for (let i = 0; i < qLen; i++) {
    const qPos = c.qPosOffset + i;
    let lo = Math.max(0, qPos - c.windowLeft);
    let end = Math.min(kvLen - 1, qPos + c.windowRight);
    if (c.segs) {
      lo = Math.max(lo, c.segs[2 * i]!);
      end = Math.min(end, c.segs[2 * i + 1]! - 1);
    }
    for (let h = 0; h < qHeads; h++) {
      const kvh = Math.floor(h / group);
      const scores: number[] = [];
      for (let j = lo; j <= end; j++) {
        let s = 0;
        for (let x = 0; x < hd; x++) {
          s += q[(i * qHeads + h) * hd + x]! * k[(j * kvHeads + kvh) * hd + x]!;
        }
        scores.push(s);
      }
      const sink = c.sinks ? c.sinks[h]! * Math.LN2 : -Infinity;
      const max = Math.max(sink, ...scores);
      if (max === -Infinity) continue;
      let sum = Math.exp(sink - max);
      for (const s of scores) sum += Math.exp(s - max);
      for (let x = 0; x < hd; x++) {
        let acc = 0;
        for (let j = lo; j <= end; j++) {
          acc += Math.exp(scores[j - lo]! - max) * v[(j * kvHeads + kvh) * hd + x]!;
        }
        out[(i * qHeads + h) * hd + x] = acc / sum;
      }
    }
  }
  return out;
}

/** Lays the blocks side by side: row r of the result is row r of each block
 *  in turn. */
function packColumns(blocks: { data: Float32Array; cols: number }[], rows: number): Float32Array {
  const width = blocks.reduce((w, b) => w + b.cols, 0);
  const out = new Float32Array(rows * width);
  for (let r = 0; r < rows; r++) {
    let at = r * width;
    for (const b of blocks) {
      out.set(b.data.subarray(r * b.cols, (r + 1) * b.cols), at);
      at += b.cols;
    }
  }
  return out;
}

/** f32: a dot product per key and an online softmax over them. f16 output
 *  adds 2^-11 relative. */
const TOLERANCE = { f32: { atol: 2e-5, rtol: 1e-4 }, f16: { atol: 2e-3, rtol: 4e-3 } };

/** Runs one sdpa on random inputs on `dev` and checks it against sdpaRef. The
 *  route it takes is picked in gpu/dispatch/attention.ts. */
export async function checkSdpa(
  dev: { root: TgpuRoot; ex: RunntimeExecutor },
  c: SdpaCase,
  dtype: 'f32' | 'f16' = 'f32',
): Promise<void> {
  const { root, ex } = dev;
  const { qLen, kvLen, qHeads, kvHeads, headDim } = c;
  const windowLeft = c.windowLeft ?? Infinity;
  const windowRight = c.windowRight ?? Infinity;
  const qPosOffset = c.qPosOffset ?? 0;
  const round = dtype === 'f16' ? roundF16 : (x: Float32Array) => x;
  // Scaled by 2 / sqrt(headDim), so scores spread over a few units whatever
  // the head size and the softmax is neither flat nor one-hot.
  const scale = 2 / Math.sqrt(headDim);
  const q = round(randomF32(qLen * qHeads * headDim, 1).map((x) => x * scale));
  const k = round(randomF32(kvLen * kvHeads * headDim, 2));
  const v = round(randomF32(kvLen * kvHeads * headDim, 3));
  const sinks = c.sinks ? randomF32(qHeads, 4).map((x) => 2 * x) : undefined;
  let segs: Float32Array | undefined;
  if (c.segments) {
    segs = new Float32Array(2 * qLen);
    const qHalf = Math.floor(qLen / 2);
    const kvHalf = Math.floor(kvLen / 2);
    for (let r = 0; r < qLen; r++) {
      segs[2 * r] = r < qHalf ? 0 : kvHalf;
      segs[2 * r + 1] = r < qHalf ? kvHalf : kvLen;
    }
  }
  const meta = (rows: number, cols: number) => ({ ...matrix(rows, cols), dtype });
  const opts = {
    qHeads,
    kvHeads,
    headDim,
    windowLeft,
    windowRight,
    qPosOffset,
    sinks: sinks && upload(root, sinks, [1, qHeads]),
    segments: segs && upload(root, segs, [qLen, 2]),
    maxSegment: c.maxSegment,
  };
  let out;
  if (c.packed) {
    const blocks = [
      { data: q, cols: qHeads * headDim },
      { data: k, cols: kvHeads * headDim },
      { data: v, cols: kvHeads * headDim },
    ];
    const width = (qHeads + 2 * kvHeads) * headDim;
    out = sdpaPacked(upload(root, packColumns(blocks, qLen), meta(qLen, width)), opts);
  } else {
    out = sdpa(
      upload(root, q, meta(qLen, qHeads * headDim)),
      upload(root, k, meta(kvLen, kvHeads * headDim)),
      upload(root, v, meta(kvLen, kvHeads * headDim)),
      opts,
    );
  }
  const got = await toArray(out, ex);
  const want = sdpaRef(q, k, v, {
    qLen,
    kvLen,
    qHeads,
    kvHeads,
    headDim,
    windowLeft,
    windowRight,
    qPosOffset,
    sinks,
    segs,
  });
  expectClose(got, want, TOLERANCE[dtype]);
}
