/** Prefill attention (more than one query row) for head sizes that are a
 *  multiple of 16. Needs subgroups.
 *
 *  A workgroup handles 16 rows of one head, or of two heads that share a kv
 *  head. It loads a block of keys and values into workgroup memory once, and
 *  all its rows use it. Each row is
 *  split across 4 or 8 lanes, which add up q·k with subgroup shuffles.
 *  Windows, segments, sinks and the key bias work as in the rows kernel.
 *
 *  Based on webgpu-kernels/com.microsoft.MultiHeadAttention (attn-flash-prefill-cluster). */

import tgpu, { d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import {
  type F32Buffer,
  f32Filler,
  type FloatBuffer,
  type KernelHandle,
  makeHandle,
} from '../common.ts';
import { cachedBindGroup, cachedUniform } from '../../gpu/dispatchCache.ts';
import { type AttnGeoCfg, type KeyBias, keyBiasAt, keyBiasFields } from './attn.ts';
import { type Elem, F32_ELEM } from '../elem.ts';

const LN2 = Math.LN2;
/** Start of the running max, as in the rows kernel. Finite, so exp() never sees Inf. */
const NEG_SEED = -1e30;
/** Score of a key the row may not see. Below NEG_SEED, so it never becomes the max. */
const MASKED = -3e38;
/** Rows per workgroup. */
const TILE_Q = 16;

const Dims = d.struct({
  qLen: d.u32,
  kvLen: d.u32,
  qPosOffset: d.u32,
  biasStart: d.u32,
  biasEnd: d.u32,
  bias: d.f32,
});

const makeLayout = (elem: Elem) =>
  tgpu.bindGroupLayout({
    q: { storage: d.arrayOf(elem.vec4), access: 'readonly' },
    k: { storage: d.arrayOf(elem.vec4), access: 'readonly' },
    v: { storage: d.arrayOf(elem.vec4), access: 'readonly' },
    sinks: { storage: d.arrayOf(d.f32), access: 'readonly' }, // [qHeads], log2-space; dead when hasSinks=0
    segs: { storage: d.arrayOf(d.f32), access: 'readonly' }, // [qLen, 2]; dead when hasSegs=0
    out: { storage: d.arrayOf(elem.vec4), access: 'mutable' }, // [qLen, qHeads·hd/4]
    dims: { uniform: Dims },
  });

const layouts = new Map<string, ReturnType<typeof makeLayout>>();
function layoutFor(elem: Elem): ReturnType<typeof makeLayout> {
  let l = layouts.get(elem.key);
  if (!l) {
    l = makeLayout(elem);
    layouts.set(elem.key, l);
  }
  return l;
}

/** Lanes per row: 8 if hd % 32 == 0, 4 if hd % 16 == 0, else 0 (no flash route). */
export function attnFlashLanes(headDim: number): number {
  return headDim % 32 === 0 ? 8 : headDim % 16 === 0 ? 4 : 0;
}

/** Widest head whose q, output and scores stay in registers. */
const REGISTER_HEAD_DIM = 128;

/** Bytes of one key or value tile for heads past REGISTER_HEAD_DIM. */
const TILE_BYTES = 8192;

/** Keys per block: 8 with 8 lanes, which keeps the scores in registers, and
 *  32 with 4 lanes, 16 for f32 heads wider than 32. Wider heads take fewer,
 *  so the key and value tiles fit in workgroup memory. */
function tileKeys(headDim: number, f16: boolean): number {
  if (headDim > REGISTER_HEAD_DIM) {
    return Math.max(1, Math.floor(TILE_BYTES / (headDim * (f16 ? 2 : 4))));
  }
  if (attnFlashLanes(headDim) === 8) return 8;
  return !f16 && headDim > 32 ? 16 : 32;
}

/** q heads per workgroup: 2 when they come in pairs that share a kv head, so
 *  every key and value read from workgroup memory feeds both. Only with 8
 *  lanes and heads up to REGISTER_HEAD_DIM, where both fit in registers. */
function headsPerWorkgroup(qHeads: number, kvHeads: number, headDim: number): number {
  const pairs = Math.floor(qHeads / kvHeads) % 2 === 0;
  return pairs && attnFlashLanes(headDim) === 8 && headDim <= REGISTER_HEAD_DIM ? 2 : 1;
}

function makeAttnFlashKernel(cfg: AttnGeoCfg, elem: Elem) {
  const { qHeads, kvHeads, headDim: hd, windowLeft, windowRight, hasSinks, hasSegs } = cfg;
  const hasKeyBias = cfg.hasKeyBias ?? 0;
  const L = layoutFor(elem);
  const f16 = elem.key === 'f16';
  const lpq = attnFlashLanes(hd);
  const slice = hd / (4 * lpq);
  const hd4 = hd / 4;
  const tileK = tileKeys(hd, f16);
  const lanes = TILE_Q * lpq;
  const group = Math.floor(qHeads / kvHeads);
  const hp = headsPerWorkgroup(qHeads, kvHeads, hd);
  const packed = (cfg.qkvPacked ?? 0) > 0;
  const qStride4 = packed ? (qHeads + 2 * kvHeads) * hd4 : qHeads * hd4;
  const kvStride4 = packed ? qStride4 : kvHeads * hd4;
  const kOff4 = packed ? qHeads * hd4 : 0;
  const vOff4 = packed ? (qHeads + kvHeads) * hd4 : 0;
  const kTile = tgpu.workgroupVar(d.arrayOf(elem.vec4, tileK * hd4));
  const vTile = tgpu.workgroupVar(d.arrayOf(elem.vec4, tileK * hd4));
  // Keys [x, y) a row may see, by the rows kernel's rule.
  const bandOf = (row: number): d.v2u => {
    'use gpu';
    const D = L.$.dims;
    const qPos = D.qPosOffset + row;
    let lo = std.max(qPos, d.u32(windowLeft)) - windowLeft;
    let stop = std.min(qPos + windowRight, D.kvLen - 1) + 1;
    if (hasSegs > 0) {
      lo = std.max(lo, d.u32(L.$.segs[2 * row]!));
      stop = std.min(stop, d.u32(L.$.segs[2 * row + 1]!));
    }
    return d.vec2u(lo, stop);
  };
  return tgpu.computeFn({
    in: { lid: d.builtin.localInvocationIndex, wid: d.builtin.workgroupId },
    workgroupSize: [lanes],
  })(({ lid, wid }) => {
    'use gpu';
    const D = L.$.dims;
    const h0 = wid.y * hp;
    const kvh = std.intdiv(h0, group);
    const qSub = std.intdiv(lid, lpq);
    const lane = lid % lpq;
    const row = wid.x * TILE_Q + qSub;
    const rowValid = row < D.qLen;
    const rowC = std.min(row, D.qLen - 1);
    // Rows past qLen in the last workgroup see no keys.
    let band = bandOf(rowC);
    if (!rowValid) {
      band = d.vec2u(0, 0);
    }
    // Loop over every key some row of the workgroup may see. Each thread works
    // this out itself: the loop has barriers, so WGSL needs uniform bounds.
    let kStart = d.u32(0xffffffff);
    let kStop = d.u32(0);
    for (const r of tgpu.unroll(std.range(TILE_Q))) {
      const rr = wid.x * TILE_Q + r;
      if (rr < D.qLen) {
        const b = bandOf(rr);
        if (b.x < b.y) {
          kStart = std.min(kStart, b.x);
          kStop = std.max(kStop, b.y);
        }
      }
    }

    // Head j of the workgroup holds q, out and scores at j·slice and j·tileK.
    const qBase4 = rowC * qStride4 + h0 * hd4 + lane * slice;
    const qr = d.arrayOf(d.vec4f, hp * slice)();
    const o = d.arrayOf(d.vec4f, hp * slice)();
    const m = d.arrayOf(d.f32, hp)();
    const l = d.arrayOf(d.f32, hp)();
    for (const j of tgpu.unroll(std.range(hp))) {
      for (const c of tgpu.unroll(std.range(slice))) {
        qr[j * slice + c] = d.vec4f(L.$.q[qBase4 + j * hd4 + c]!);
      }
      m[j] = d.f32(NEG_SEED);
      if (hasSinks > 0) {
        m[j] = L.$.sinks[h0 + j]! * LN2;
        l[j] = d.f32(1);
      }
    }

    const s = d.arrayOf(d.f32, hp * tileK)();
    for (let t0 = kStart; t0 < kStop; t0 += tileK) {
      std.workgroupBarrier();
      for (let i = lid; i < tileK * hd4; i += lanes) {
        const kj = t0 + std.intdiv(i, hd4);
        const col = i % hd4;
        let kv = d.vec4f();
        let vv = d.vec4f();
        if (kj < kStop) {
          const base4 = kj * kvStride4 + kvh * hd4 + col;
          kv = d.vec4f(L.$.k[base4 + kOff4]!);
          vv = d.vec4f(L.$.v[base4 + vOff4]!);
        }
        // Elem types f16 storage as vec4f, so TypeScript needs the casts; the shader stores vec4h.
        if (f16) {
          kTile.$[i] = d.vec4h(kv) as unknown as d.v4f;
          vTile.$[i] = d.vec4h(vv) as unknown as d.v4f;
        } else {
          kTile.$[i] = d.vec4f(kv);
          vTile.$[i] = d.vec4f(vv);
        }
      }
      std.workgroupBarrier();

      // Each key and value read from workgroup memory feeds every head.
      const part = d.arrayOf(d.f32, hp)();
      const tileMax = d.arrayOf(d.f32, hp)();
      for (const j of tgpu.unroll(std.range(hp))) {
        tileMax[j] = d.f32(MASKED);
      }
      for (const kk of tgpu.unroll(std.range(tileK))) {
        for (const j of tgpu.unroll(std.range(hp))) {
          part[j] = d.f32(0);
        }
        for (const c of tgpu.unroll(std.range(slice))) {
          const kv = d.vec4f(kTile.$[kk * hd4 + lane * slice + c]!);
          for (const j of tgpu.unroll(std.range(hp))) {
            part[j] = part[j]! + std.dot(qr[j * slice + c]!, kv);
          }
        }
        const kj = t0 + kk;
        const seen = kj >= band.x && kj < band.y;
        for (const j of tgpu.unroll(std.range(hp))) {
          let x = part[j]!;
          x += std.subgroupShuffleXor(x, 1);
          x += std.subgroupShuffleXor(x, 2);
          if (lpq > 4) {
            x += std.subgroupShuffleXor(x, 4);
          }
          if (hasKeyBias > 0) {
            x += keyBiasAt(kj, D.biasStart, D.biasEnd, D.bias);
          }
          s[j * tileK + kk] = std.select(d.f32(MASKED), x, seen);
          tileMax[j] = std.max(tileMax[j]!, s[j * tileK + kk]!);
        }
      }
      const corr = d.arrayOf(d.f32, hp)();
      for (const j of tgpu.unroll(std.range(hp))) {
        const mNew = std.max(m[j]!, tileMax[j]!);
        corr[j] = std.exp(m[j]! - mNew);
        let pSum = d.f32(0);
        for (const kk of tgpu.unroll(std.range(tileK))) {
          const sk = s[j * tileK + kk]!;
          const p = std.select(d.f32(0), std.exp(sk - mNew), sk > d.f32(MASKED));
          s[j * tileK + kk] = p;
          pSum += p;
        }
        l[j] = l[j]! * corr[j]! + pSum;
        m[j] = mNew;
      }
      for (const c of tgpu.unroll(std.range(slice))) {
        const acc = d.arrayOf(d.vec4f, hp)();
        for (const j of tgpu.unroll(std.range(hp))) {
          acc[j] = o[j * slice + c]! * corr[j]!;
        }
        for (const kk of tgpu.unroll(std.range(tileK))) {
          const vv = d.vec4f(vTile.$[kk * hd4 + lane * slice + c]!);
          for (const j of tgpu.unroll(std.range(hp))) {
            acc[j] = acc[j]! + s[j * tileK + kk]! * vv;
          }
        }
        for (const j of tgpu.unroll(std.range(hp))) {
          o[j * slice + c] = d.vec4f(acc[j]!);
        }
      }
    }

    if (rowValid) {
      const oBase4 = row * qHeads * hd4 + h0 * hd4 + lane * slice;
      for (const j of tgpu.unroll(std.range(hp))) {
        const invL = std.select(d.f32(0), 1 / l[j]!, l[j]! > 0);
        for (const c of tgpu.unroll(std.range(slice))) {
          if (f16) {
            L.$.out[oBase4 + j * hd4 + c] = d.vec4h(o[j * slice + c]! * invL) as unknown as d.v4f;
          } else {
            L.$.out[oBase4 + j * hd4 + c] = d.vec4f(o[j * slice + c]! * invL);
          }
        }
      }
    }
  });
}

/** One pipeline per layer config and dtype. */
export function createAttnFlashPipeline(root: TgpuRoot, cfg: AttnGeoCfg, elem: Elem = F32_ELEM) {
  return root.createComputePipeline({ compute: makeAttnFlashKernel(cfg, elem) });
}

/** One workgroup per 16 rows of one head, or of two heads that share a kv head. */
export function attnFlashHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createAttnFlashPipeline>,
  args: {
    qLen: number;
    kvLen: number;
    qPosOffset: number;
    qHeads: number;
    kvHeads: number;
    headDim: number;
    keyBias?: KeyBias;
  },
  buffers: {
    q: FloatBuffer;
    k: FloatBuffer;
    v: FloatBuffer;
    out: FloatBuffer;
    sinks?: F32Buffer;
    segs?: F32Buffer;
    dummyF32?: F32Buffer;
  },
  elem: Elem = F32_ELEM,
): KernelHandle {
  const dims = cachedUniform(root, Dims, {
    qLen: args.qLen,
    kvLen: args.kvLen,
    qPosOffset: args.qPosOffset,
    ...keyBiasFields(args.keyBias),
  });
  const filler = f32Filler(buffers, elem);
  const { dummyF32: _dummy, ...bound } = buffers;
  const bindGroup = cachedBindGroup(root, layoutFor(elem), {
    ...bound,
    sinks: buffers.sinks ?? filler,
    segs: buffers.segs ?? filler,
    dims,
  });
  return makeHandle(pipeline, 'attnFlash', bindGroup, [
    Math.ceil(args.qLen / TILE_Q),
    args.qHeads / headsPerWorkgroup(args.qHeads, args.kvHeads, args.headDim),
  ]);
}
