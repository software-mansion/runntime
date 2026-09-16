import tgpu, { d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import {
  type F32Buffer,
  f32Filler,
  flatWorkgroupId,
  type FloatBuffer,
  type KernelHandle,
  makeHandle,
  WORKGROUP_SIZE,
} from '../common.ts';
import { type Elem, F32_ELEM } from '../elem.ts';

/** Split-K attention for qLen = 1: the same banded GQA SDPA as attn.ts,
 *  parallelized for decode.
 *
 *  One workgroup per (q row, head), its threads striding the key range and
 *  merging their online-softmax partials through workgroup memory. Only the
 *  summation order differs from attn.ts, so results agree to ~1e-6, not
 *  bitwise. */

const LN2 = Math.LN2;
const MAX_HEAD_DIM = 128;
const NEG_SEED = -1e30;
const TILE = 32;

const Dims = d.struct({ qLen: d.u32, kvLen: d.u32, qPosOffset: d.u32 });

export interface AttnDecodeCfg {
  qHeads: number;
  kvHeads: number;
  headDim: number;
  windowLeft: number;
  windowRight: number;
  hasSinks: number; // 0 | 1 — selects the softmax seed
}

const makeDecodeLayout = (elem: Elem) =>
  tgpu.bindGroupLayout({
    q: { storage: d.arrayOf(elem.scalar), access: 'readonly' }, // [qLen, qHeads·headDim]
    k: { storage: d.arrayOf(elem.scalar), access: 'readonly' }, // [kvLen, kvHeads·headDim]
    v: { storage: d.arrayOf(elem.scalar), access: 'readonly' }, // [kvLen, kvHeads·headDim]
    sinks: { storage: d.arrayOf(d.f32), access: 'readonly' }, // [qHeads], log2-space; dead when hasSinks=0
    out: { storage: d.arrayOf(elem.scalar), access: 'mutable' }, // [qLen, qHeads·headDim]
    dims: { uniform: Dims },
  });

const decodeLayouts = new Map<string, ReturnType<typeof makeDecodeLayout>>();
export function attnDecodeLayoutFor(elem: Elem): ReturnType<typeof makeDecodeLayout> {
  let l = decodeLayouts.get(elem.key);
  if (!l) {
    l = makeDecodeLayout(elem);
    decodeLayouts.set(elem.key, l);
  }
  return l;
}

const partialM = tgpu.workgroupVar(d.arrayOf(d.f32, WORKGROUP_SIZE));
const partialL = tgpu.workgroupVar(d.arrayOf(d.f32, WORKGROUP_SIZE));

const scratchOf = (hd: number) => ({
  acc: tgpu.privateVar(d.arrayOf(d.f32, paddedFor(hd))),
  tile: tgpu.workgroupVar(d.arrayOf(d.f32, WORKGROUP_SIZE * tileFor(hd))),
});

const MIN_EXACT_TILE = 8;

function tileFor(hd: number): number {
  for (let t = Math.min(TILE, hd); t >= MIN_EXACT_TILE; t--) {
    if (hd % t === 0) return t;
  }
  return Math.min(TILE, hd);
}

function paddedFor(hd: number): number {
  const t = tileFor(hd);
  return Math.ceil(hd / t) * t;
}

function makeAttnDecodeKernel(cfg: AttnDecodeCfg, elem: Elem) {
  const layout = attnDecodeLayoutFor(elem);
  const storeScalar = elem.scalar;
  const { qHeads, kvHeads, headDim: hd, windowLeft, windowRight, hasSinks } = cfg;
  const group = Math.floor(qHeads / kvHeads);
  const qWidth = qHeads * hd;
  const kvWidth = kvHeads * hd;
  const tile = tileFor(hd);
  const padded = paddedFor(hd);
  // False whenever the tile divides the head, so the guard folds away and
  // those kernels keep the exact-divisor merge.
  const needsTailGuard = padded !== hd;
  const { acc: accReg, tile: partialAcc } = scratchOf(hd);
  return tgpu.computeFn({
    in: { wid: d.builtin.workgroupId, lid: d.builtin.localInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ wid, lid }) => {
    'use gpu';
    const D = layout.$.dims;
    const tid = lid.x;
    const slot = flatWorkgroupId(wid);
    const h = slot % qHeads;
    const row = d.u32(slot / qHeads);
    const kvh = d.u32(h / group);
    const qBase = row * qWidth + h * hd;
    const qPos = D.qPosOffset + row;

    // Padded tail lanes must read as 0.
    for (const z of std.range(padded)) {
      accReg.$[z] = d.f32(0);
    }

    // One thread carries the sink term; the merge absorbs it.
    let m = d.f32(NEG_SEED);
    let l = d.f32(0);
    if (hasSinks > 0) {
      if (tid === 0) {
        m = layout.$.sinks[h]! * LN2;
        l = d.f32(1);
      }
    }

    // Strided across the workgroup; max avoids u32 underflow.
    const lo = std.max(qPos, d.u32(windowLeft)) - windowLeft;
    const end = std.min(qPos + windowRight, D.kvLen - 1);
    for (let j = lo + tid; j <= end; j += WORKGROUP_SIZE) {
      const kBase = j * kvWidth + kvh * hd;
      let score = d.f32(0);
      for (const kk of std.range(hd)) {
        score += d.f32(layout.$.q[qBase + kk]!) * d.f32(layout.$.k[kBase + kk]!);
      }

      const mNew = std.max(m, score);
      const correction = std.exp(m - mNew);
      const p = std.exp(score - mNew);
      l = l * correction + p;
      m = mNew;

      const vBase = j * kvWidth + kvh * hd;
      for (const nn of std.range(hd)) {
        accReg.$[nn] = accReg.$[nn]! * correction + d.f32(layout.$.v[vBase + nn]!) * p;
      }
    }

    // Every thread scans, which leaves M uniform without a second barrier.
    partialM.$[tid] = m;
    std.workgroupBarrier();
    let M = d.f32(NEG_SEED);
    for (const i of std.range(WORKGROUP_SIZE)) {
      M = std.max(M, partialM.$[i]!);
    }

    // Rescale to the workgroup max, then sum the l's.
    const scale = std.exp(m - M);
    partialL.$[tid] = l * scale;
    std.workgroupBarrier();
    let L = d.f32(0);
    for (const i of std.range(WORKGROUP_SIZE)) {
      L += partialL.$[i]!;
    }

    for (const c of std.range(hd)) {
      accReg.$[c] = accReg.$[c]! * scale;
    }

    // Empty band: L = 0, so the row is zeros.
    const invL = std.select(0, 1 / L, L > 0);
    const oBase = row * qWidth + h * hd;

    // Column-sum one tile at a time; the barriers stay in uniform control
    // flow.
    for (const base of std.range(0, padded, tile)) {
      std.workgroupBarrier(); // previous round's reads finish before overwrite
      for (const t of std.range(tile)) {
        partialAcc.$[tid * tile + t] = accReg.$[base + t]!;
      }
      std.workgroupBarrier();
      if (tid < tile && (!needsTailGuard || base + tid < hd)) {
        let sum = d.f32(0);
        for (const r of std.range(WORKGROUP_SIZE)) {
          sum += partialAcc.$[r * tile + tid]!;
        }
        layout.$.out[oBase + base + tid] = storeScalar(sum * invL);
      }
    }
  });
}

export function createAttnDecodePipeline(
  root: TgpuRoot,
  cfg: AttnDecodeCfg,
  elem: Elem = F32_ELEM,
) {
  return root.createComputePipeline({ compute: makeAttnDecodeKernel(cfg, elem) });
}

export function attnDecodeHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createAttnDecodePipeline>,
  args: { qLen: number; kvLen: number; qPosOffset: number; qHeads: number },
  buffers: {
    q: FloatBuffer;
    k: FloatBuffer;
    v: FloatBuffer;
    out: FloatBuffer;
    sinks?: F32Buffer;
    dummyF32?: F32Buffer;
  },
  elem: Elem = F32_ELEM,
): KernelHandle {
  const dims = root
    .createBuffer(Dims, { qLen: args.qLen, kvLen: args.kvLen, qPosOffset: args.qPosOffset })
    .$usage('uniform');
  const filler = f32Filler(buffers, elem);
  const { dummyF32: _dummy, ...bound } = buffers;
  const bindGroup = root.createBindGroup(attnDecodeLayoutFor(elem), {
    ...bound,
    sinks: buffers.sinks ?? filler,
    dims,
  });
  // One workgroup per (q row, q head), not one thread.
  return makeHandle(pipeline, 'attnDecode', bindGroup, args.qLen * args.qHeads);
}

// ── Key-split pair (long key ranges) ─────────────────────────────────────
// One workgroup per head leaves most of the GPU idle on a long key range, so
// past ATTN_DECODE_CHUNK keys the executor splits into two dispatches: a
// per-(head, chunk) scan writing unnormalized partials, then a per-head merge.
//
// All split pairs in a submit share one `partials` buffer, which is safe
// because dispatches run in encode order. The sink term enters on chunk 0
// thread 0, so the merge needs no special case.

export const ATTN_DECODE_CHUNK = 256;
export const ATTN_DECODE_MAX_CHUNKS = 8;
export const ATTN_SPLIT_SCRATCH_ELEMS = 64 * ATTN_DECODE_MAX_CHUNKS * (2 + MAX_HEAD_DIM);

const SplitDims = d.struct({ kvLen: d.u32, qPosOffset: d.u32, chunks: d.u32 });

const makeSplitLayout = (elem: Elem) =>
  tgpu.bindGroupLayout({
    q: { storage: d.arrayOf(elem.scalar), access: 'readonly' }, // [1, qHeads·headDim]
    k: { storage: d.arrayOf(elem.scalar), access: 'readonly' }, // [kvLen, kvHeads·headDim]
    v: { storage: d.arrayOf(elem.scalar), access: 'readonly' }, // [kvLen, kvHeads·headDim]
    sinks: { storage: d.arrayOf(d.f32), access: 'readonly' }, // [qHeads]; dead when hasSinks=0
    partials: { storage: d.arrayOf(d.f32), access: 'mutable' }, // [(h·MAX_CHUNKS+c)·(2+hd)]
    dims: { uniform: SplitDims },
  });

const splitLayouts = new Map<string, ReturnType<typeof makeSplitLayout>>();
export function attnDecodeSplitLayoutFor(elem: Elem): ReturnType<typeof makeSplitLayout> {
  let l = splitLayouts.get(elem.key);
  if (!l) {
    l = makeSplitLayout(elem);
    splitLayouts.set(elem.key, l);
  }
  return l;
}

function makeAttnDecodeSplitKernel(cfg: AttnDecodeCfg, elem: Elem) {
  const layout = attnDecodeSplitLayoutFor(elem);
  const { qHeads, kvHeads, headDim: hd, windowLeft, windowRight, hasSinks } = cfg;
  const group = Math.floor(qHeads / kvHeads);
  const kvWidth = kvHeads * hd;
  const tile = tileFor(hd);
  const padded = paddedFor(hd);
  // False whenever the tile divides the head, so the guard folds away and
  // those kernels keep the exact-divisor merge.
  const needsTailGuard = padded !== hd;
  const { acc: accReg, tile: partialAcc } = scratchOf(hd);
  return tgpu.computeFn({
    in: { wid: d.builtin.workgroupId, lid: d.builtin.localInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ wid, lid }) => {
    'use gpu';
    const D = layout.$.dims;
    const tid = lid.x;
    const chunkSlot = flatWorkgroupId(wid);
    const chunk = chunkSlot % D.chunks;
    const h = d.u32(chunkSlot / D.chunks);
    const kvh = d.u32(h / group);
    const qBase = h * hd; // qLen = 1 → row 0
    const qPos = D.qPosOffset;

    // Padded: the last merge round's tail lanes must read as 0.
    for (const z of std.range(padded)) {
      accReg.$[z] = d.f32(0);
    }

    let m = d.f32(NEG_SEED);
    let l = d.f32(0);
    if (hasSinks > 0) {
      if (chunk === 0) {
        if (tid === 0) {
          m = layout.$.sinks[h]! * LN2;
          l = d.f32(1);
        }
      }
    }

    // Sliced into `chunks` even pieces; this workgroup strides its own.
    const lo = std.max(qPos, d.u32(windowLeft)) - windowLeft;
    const end = std.min(qPos + windowRight, D.kvLen - 1);
    const range = end + 1 - std.min(lo, end + 1); // 0 when the band is empty
    // A floor of 1 keeps `myLo + chunkLen − 1` from wrapping.
    const chunkLen = std.max(d.u32((range + D.chunks - 1) / D.chunks), 1);
    const myLo = lo + chunk * chunkLen;
    const myEnd = std.min(myLo + chunkLen - 1, end);
    for (let j = myLo + tid; j <= myEnd; j += WORKGROUP_SIZE) {
      const kBase = j * kvWidth + kvh * hd;
      let score = d.f32(0);
      for (const kk of std.range(hd)) {
        score += d.f32(layout.$.q[qBase + kk]!) * d.f32(layout.$.k[kBase + kk]!);
      }
      const mNew = std.max(m, score);
      const correction = std.exp(m - mNew);
      const p = std.exp(score - mNew);
      l = l * correction + p;
      m = mNew;
      const vBase = j * kvWidth + kvh * hd;
      for (const nn of std.range(hd)) {
        accReg.$[nn] = accReg.$[nn]! * correction + d.f32(layout.$.v[vBase + nn]!) * p;
      }
    }

    // As the single-workgroup kernel, but unnormalized into a scratch slot.
    partialM.$[tid] = m;
    std.workgroupBarrier();
    let M = d.f32(NEG_SEED);
    for (const i of std.range(WORKGROUP_SIZE)) {
      M = std.max(M, partialM.$[i]!);
    }
    const scale = std.exp(m - M);
    partialL.$[tid] = l * scale;
    std.workgroupBarrier();
    let L = d.f32(0);
    for (const i of std.range(WORKGROUP_SIZE)) {
      L += partialL.$[i]!;
    }
    for (const c of std.range(hd)) {
      accReg.$[c] = accReg.$[c]! * scale;
    }

    const slot = (h * ATTN_DECODE_MAX_CHUNKS + chunk) * (2 + hd);
    if (tid === 0) {
      layout.$.partials[slot] = M;
      layout.$.partials[slot + 1] = L;
    }
    for (const base of std.range(0, padded, tile)) {
      std.workgroupBarrier(); // previous round's reads finish before overwrite
      for (const t of std.range(tile)) {
        partialAcc.$[tid * tile + t] = accReg.$[base + t]!;
      }
      std.workgroupBarrier();
      if (tid < tile && (!needsTailGuard || base + tid < hd)) {
        let sum = d.f32(0);
        for (const r of std.range(WORKGROUP_SIZE)) {
          sum += partialAcc.$[r * tile + tid]!;
        }
        layout.$.partials[slot + 2 + base + tid] = sum;
      }
    }
  });
}

const MergeDims = d.struct({ chunks: d.u32 });

const makeMergeLayout = (elem: Elem) =>
  tgpu.bindGroupLayout({
    partials: { storage: d.arrayOf(d.f32), access: 'readonly' },
    out: { storage: d.arrayOf(elem.scalar), access: 'mutable' }, // [1, qHeads·headDim]
    dims: { uniform: MergeDims },
  });

const mergeLayouts = new Map<string, ReturnType<typeof makeMergeLayout>>();
export function attnDecodeMergeLayoutFor(elem: Elem): ReturnType<typeof makeMergeLayout> {
  let l = mergeLayouts.get(elem.key);
  if (!l) {
    l = makeMergeLayout(elem);
    mergeLayouts.set(elem.key, l);
  }
  return l;
}

function makeAttnDecodeMergeKernel(cfg: AttnDecodeCfg, elem: Elem) {
  const layout = attnDecodeMergeLayoutFor(elem);
  const storeScalar = elem.scalar;
  const { qHeads, headDim: hd } = cfg;
  const stride = 2 + hd;
  return tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const h = gid.x; // one thread per head — qHeads is far below the grid cap
    if (h >= qHeads) {
      return;
    }
    const chunks = layout.$.dims.chunks;
    const base = h * ATTN_DECODE_MAX_CHUNKS * stride;

    let M = d.f32(NEG_SEED);
    for (let c = d.u32(0); c < chunks; c++) {
      M = std.max(M, layout.$.partials[base + c * stride]!);
    }
    let L = d.f32(0);
    for (let c = d.u32(0); c < chunks; c++) {
      const mc = layout.$.partials[base + c * stride]!;
      L += layout.$.partials[base + c * stride + 1]! * std.exp(mc - M);
    }
    const invL = std.select(0, 1 / L, L > 0);
    for (const dcol of std.range(hd)) {
      let sum = d.f32(0);
      for (let c = d.u32(0); c < chunks; c++) {
        const mc = layout.$.partials[base + c * stride]!;
        sum += layout.$.partials[base + c * stride + 2 + dcol]! * std.exp(mc - M);
      }
      layout.$.out[h * hd + dcol] = storeScalar(sum * invL);
    }
  });
}

export function createAttnDecodeSplitPipeline(
  root: TgpuRoot,
  cfg: AttnDecodeCfg,
  elem: Elem = F32_ELEM,
) {
  return root.createComputePipeline({ compute: makeAttnDecodeSplitKernel(cfg, elem) });
}

export function createAttnDecodeMergePipeline(
  root: TgpuRoot,
  cfg: AttnDecodeCfg,
  elem: Elem = F32_ELEM,
) {
  return root.createComputePipeline({ compute: makeAttnDecodeMergeKernel(cfg, elem) });
}

export function attnDecodeSplitHandles(
  root: TgpuRoot,
  partialPipeline: ReturnType<typeof createAttnDecodeSplitPipeline>,
  mergePipeline: ReturnType<typeof createAttnDecodeMergePipeline>,
  args: { kvLen: number; qPosOffset: number; qHeads: number; chunks: number },
  buffers: {
    q: FloatBuffer;
    k: FloatBuffer;
    v: FloatBuffer;
    out: FloatBuffer;
    partials: F32Buffer;
    sinks?: F32Buffer;
    dummyF32?: F32Buffer;
  },
  elem: Elem = F32_ELEM,
): [KernelHandle, KernelHandle] {
  const dims = root
    .createBuffer(SplitDims, {
      kvLen: args.kvLen,
      qPosOffset: args.qPosOffset,
      chunks: args.chunks,
    })
    .$usage('uniform');
  const partialBind = root.createBindGroup(attnDecodeSplitLayoutFor(elem), {
    q: buffers.q,
    k: buffers.k,
    v: buffers.v,
    sinks: buffers.sinks ?? f32Filler(buffers, elem),
    partials: buffers.partials,
    dims,
  });
  const mergeDims = root.createBuffer(MergeDims, { chunks: args.chunks }).$usage('uniform');
  const mergeBind = root.createBindGroup(attnDecodeMergeLayoutFor(elem), {
    partials: buffers.partials,
    out: buffers.out,
    dims: mergeDims,
  });
  return [
    makeHandle(partialPipeline, 'attnDecodeSplit', partialBind, args.qHeads * args.chunks),
    makeHandle(mergePipeline, 'attnDecodeMerge', mergeBind, 1),
  ];
}
