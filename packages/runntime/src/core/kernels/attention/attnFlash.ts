/** Prefill attention (more than one query row) for head sizes that are a
 *  multiple of 16. Needs subgroups.
 *
 *  A workgroup handles 16 rows of one head. It loads a block of keys and
 *  values into workgroup memory once, and all 16 rows use it. Each row is
 *  split across 4 or 8 lanes, which add up q·k with subgroup shuffles.
 *  Windows, segments and sinks work as in the rows kernel.
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
import type { AttnGeoCfg } from './attn.ts';
import { type Elem, F32_ELEM } from '../elem.ts';

const LN2 = Math.LN2;
/** Start of the running max, as in the rows kernel. Finite, so exp() never sees Inf. */
const NEG_SEED = -1e30;
/** Score of a key the row may not see. Below NEG_SEED, so it never becomes the max. */
const MASKED = -3e38;
/** Rows per workgroup. */
const TILE_Q = 16;

const Dims = d.struct({ qLen: d.u32, kvLen: d.u32, qPosOffset: d.u32 });

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

/** Keys per block. f32 heads wider than 32 take 16 to fit in 16 KiB of workgroup memory. */
function tileKeys(headDim: number, f16: boolean): number {
  return !f16 && headDim > 32 ? 16 : 32;
}

function makeAttnFlashKernel(cfg: AttnGeoCfg, elem: Elem) {
  const { qHeads, kvHeads, headDim: hd, windowLeft, windowRight, hasSinks, hasSegs } = cfg;
  const L = layoutFor(elem);
  const f16 = elem.key === 'f16';
  const lpq = attnFlashLanes(hd);
  const slice = hd / (4 * lpq);
  const hd4 = hd / 4;
  const tileK = tileKeys(hd, f16);
  const lanes = TILE_Q * lpq;
  const group = Math.floor(qHeads / kvHeads);
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
    const h = wid.y;
    const kvh = d.u32(h / group);
    const qSub = d.u32(lid / lpq);
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

    const qBase4 = rowC * qStride4 + h * hd4 + lane * slice;
    const qr = d.arrayOf(d.vec4f, slice)();
    const o = d.arrayOf(d.vec4f, slice)();
    for (const c of tgpu.unroll(std.range(slice))) {
      qr[c] = d.vec4f(L.$.q[qBase4 + c]!);
    }

    let m = d.f32(NEG_SEED);
    let l = d.f32(0);
    if (hasSinks > 0) {
      m = L.$.sinks[h]! * LN2;
      l = d.f32(1);
    }

    const s = d.arrayOf(d.f32, tileK)();
    for (let t0 = kStart; t0 < kStop; t0 += tileK) {
      std.workgroupBarrier();
      for (let i = lid; i < tileK * hd4; i += lanes) {
        const kj = t0 + d.u32(i / hd4);
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

      let tileMax = d.f32(MASKED);
      for (const kk of tgpu.unroll(std.range(tileK))) {
        let part = d.f32(0);
        for (const c of tgpu.unroll(std.range(slice))) {
          part += std.dot(qr[c]!, d.vec4f(kTile.$[kk * hd4 + lane * slice + c]!));
        }
        part += std.subgroupShuffleXor(part, 1);
        part += std.subgroupShuffleXor(part, 2);
        if (lpq > 4) {
          part += std.subgroupShuffleXor(part, 4);
        }
        const kj = t0 + kk;
        s[kk] = std.select(d.f32(MASKED), part, kj >= band.x && kj < band.y);
        tileMax = std.max(tileMax, s[kk]!);
      }
      const mNew = std.max(m, tileMax);
      const corr = std.exp(m - mNew);
      let pSum = d.f32(0);
      for (const kk of tgpu.unroll(std.range(tileK))) {
        const p = std.select(d.f32(0), std.exp(s[kk]! - mNew), s[kk]! > d.f32(MASKED));
        s[kk] = p;
        pSum += p;
      }
      l = l * corr + pSum;
      m = mNew;
      for (const c of tgpu.unroll(std.range(slice))) {
        let acc = o[c]! * corr;
        for (const kk of tgpu.unroll(std.range(tileK))) {
          acc += s[kk]! * d.vec4f(vTile.$[kk * hd4 + lane * slice + c]!);
        }
        o[c] = d.vec4f(acc);
      }
    }

    if (rowValid) {
      const invL = std.select(d.f32(0), 1 / l, l > 0);
      const oBase4 = row * qHeads * hd4 + h * hd4 + lane * slice;
      for (const c of tgpu.unroll(std.range(slice))) {
        if (f16) {
          L.$.out[oBase4 + c] = d.vec4h(o[c]! * invL) as unknown as d.v4f;
        } else {
          L.$.out[oBase4 + c] = d.vec4f(o[c]! * invL);
        }
      }
    }
  });
}

/** One pipeline per layer config and dtype. */
export function createAttnFlashPipeline(root: TgpuRoot, cfg: AttnGeoCfg, elem: Elem = F32_ELEM) {
  return root.createComputePipeline({ compute: makeAttnFlashKernel(cfg, elem) });
}

/** One workgroup per 16 rows of one head. */
export function attnFlashHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createAttnFlashPipeline>,
  args: { qLen: number; kvLen: number; qPosOffset: number; qHeads: number },
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
  });
  const filler = f32Filler(buffers, elem);
  const { dummyF32: _dummy, ...bound } = buffers;
  const bindGroup = cachedBindGroup(root, layoutFor(elem), {
    ...bound,
    sinks: buffers.sinks ?? filler,
    segs: buffers.segs ?? filler,
    dims,
  });
  return makeHandle(pipeline, 'attnFlash', bindGroup, [Math.ceil(args.qLen / TILE_Q), args.qHeads]);
}
