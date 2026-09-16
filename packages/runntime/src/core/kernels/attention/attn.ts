import tgpu, { d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import {
  flatIndex,
  type F32Buffer,
  f32Filler,
  type FloatBuffer,
  type KernelHandle,
  makeHandle,
  WORKGROUP_SIZE,
} from '../common.ts';
import { type Elem, F32_ELEM } from '../elem.ts';

/** Scaled-dot-product attention, one thread per (q row, q head).
 *
 *  Serial in the key count, so the executor prefers the row kernels below and
 *  falls back here past the dispatch limit. q and k arrive pre-scaled. */

const LN2 = Math.LN2;
const MAX_HEAD_DIM = 128;
const NEG_SEED = -1e30;

const Dims = d.struct({ qLen: d.u32, kvLen: d.u32, qPosOffset: d.u32 });

const Config = d.struct({
  qHeads: d.u32,
  kvHeads: d.u32,
  headDim: d.u32,
  windowLeft: d.u32,
  windowRight: d.u32,
  hasSinks: d.u32,
  hasSegs: d.u32,
  // 0: q/k/v are three separate tensors. 1: all three bindings point at one
  // [T, (qHeads+2·kvHeads)·headDim] tensor whose columns are the q|k|v
  // blocks — the kernel reads them by column offset, no slice copies.
  qkvPacked: d.u32,
});
const config = tgpu.accessor(Config, {
  qHeads: 1,
  kvHeads: 1,
  headDim: 2,
  windowLeft: 0,
  windowRight: 0,
  hasSinks: 0,
  hasSegs: 0,
  qkvPacked: 0,
});

const makeAttnLayout = (elem: Elem) =>
  tgpu.bindGroupLayout({
    q: { storage: d.arrayOf(elem.scalar), access: 'readonly' }, // [qLen, qHeads·headDim]
    k: { storage: d.arrayOf(elem.scalar), access: 'readonly' }, // [kvLen, kvHeads·headDim]
    v: { storage: d.arrayOf(elem.scalar), access: 'readonly' }, // [kvLen, kvHeads·headDim]
    sinks: { storage: d.arrayOf(d.f32), access: 'readonly' }, // [qHeads], log2-space; dead when hasSinks=0
    segs: { storage: d.arrayOf(d.f32), access: 'readonly' }, // [qLen, 2] row's key range; dead when hasSegs=0
    out: { storage: d.arrayOf(elem.scalar), access: 'mutable' }, // [qLen, qHeads·headDim]
    dims: { uniform: Dims },
  });

const attnLayouts = new Map<string, ReturnType<typeof makeAttnLayout>>();
export function attnLayoutFor(elem: Elem): ReturnType<typeof makeAttnLayout> {
  let l = attnLayouts.get(elem.key);
  if (!l) {
    l = makeAttnLayout(elem);
    attnLayouts.set(elem.key, l);
  }
  return l;
}

export const attnLayout = attnLayoutFor(F32_ELEM);

const accReg = tgpu.privateVar(d.arrayOf(d.f32, MAX_HEAD_DIM));

function makeAttnKernel(elem: Elem = F32_ELEM) {
  const f16 = elem.key === 'f16';
  const A = attnLayoutFor(elem);
  return tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const D = A.$.dims;
    const C = config.$;
    const idx = flatIndex(gid);
    if (idx >= D.qLen * C.qHeads) {
      return;
    }
    const h = idx % C.qHeads;
    const row = d.u32(idx / C.qHeads); // explicit integer division
    const kvh = d.u32(h / d.u32(C.qHeads / C.kvHeads));
    const hd = C.headDim;
    const qWidth = C.qHeads * hd;
    const kvWidth = C.kvHeads * hd;
    let qStride = qWidth;
    let kvStride = kvWidth;
    let kOff = d.u32(0);
    let vOff = d.u32(0);
    if (C.qkvPacked > 0) {
      qStride = qWidth + 2 * kvWidth;
      kvStride = qStride;
      kOff = qWidth;
      vOff = qWidth + kvWidth;
    }
    const qBase = row * qStride + h * hd;
    const qPos = D.qPosOffset + row;

    for (let z = d.u32(0); z < hd; z++) {
      accReg.$[z] = d.f32(0);
    }

    // -1e30 rather than -Inf: buffers never hold ±Inf.
    let m = d.f32(NEG_SEED);
    let l = d.f32(0);
    if (C.hasSinks > 0) {
      m = A.$.sinks[h]! * LN2;
      l = d.f32(1);
    }

    // max(qPos, WL) − WL avoids u32 underflow.
    let lo = std.max(qPos, C.windowLeft) - C.windowLeft;
    let end = std.min(qPos + C.windowRight, D.kvLen - 1);
    if (C.hasSegs > 0) {
      lo = std.max(lo, d.u32(A.$.segs[2 * row]!));
      end = std.min(end, d.u32(A.$.segs[2 * row + 1]!) - 1);
    }

    for (let j = lo; j <= end; j++) {
      const kBase = j * kvStride + kOff + kvh * hd;
      let score = d.f32(0);
      for (let kk = d.u32(0); kk < hd; kk++) {
        if (f16) {
          score += d.f32(A.$.q[qBase + kk]!) * d.f32(A.$.k[kBase + kk]!);
        } else {
          score += A.$.q[qBase + kk]! * A.$.k[kBase + kk]!;
        }
      }

      const mNew = std.max(m, score);
      const correction = std.exp(m - mNew);
      const p = std.exp(score - mNew);
      l = l * correction + p;
      m = mNew;

      const vBase = j * kvStride + vOff + kvh * hd;
      for (let nn = d.u32(0); nn < hd; nn++) {
        if (f16) {
          accReg.$[nn] = accReg.$[nn]! * correction + d.f32(A.$.v[vBase + nn]!) * p;
        } else {
          accReg.$[nn] = accReg.$[nn]! * correction + A.$.v[vBase + nn]! * p;
        }
      }
    }

    // Empty band writes zeros rather than dividing by 0.
    const invL = std.select(0, 1 / l, l > 0);
    const oBase = row * qWidth + h * hd;
    for (let o = d.u32(0); o < hd; o++) {
      if (f16) {
        A.$.out[oBase + o] = d.f16(accReg.$[o]! * invL);
      } else {
        A.$.out[oBase + o] = accReg.$[o]! * invL;
      }
    }
  });
}

const attnKernels = new Map<string, ReturnType<typeof makeAttnKernel>>();
function attnKernelFor(elem: Elem) {
  let k = attnKernels.get(elem.key);
  if (!k) {
    k = makeAttnKernel(elem);
    attnKernels.set(elem.key, k);
  }
  return k;
}

export function createAttnPipeline(
  root: TgpuRoot,
  cfg: {
    qHeads: number;
    kvHeads: number;
    headDim: number;
    windowLeft: number;
    windowRight: number;
    hasSinks: number; // 0 | 1 — comptime; selects the softmax seed
    hasSegs: number; // 0 | 1 — comptime; clips each row's keys to its segment
    qkvPacked?: number; // 0 | 1 — comptime; q/k/v as column blocks of one tensor
  },
  elem: Elem = F32_ELEM,
) {
  return root
    .with(config, { ...cfg, qkvPacked: cfg.qkvPacked ?? 0 })
    .createComputePipeline({ compute: attnKernelFor(elem) });
}

export function attnHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createAttnPipeline>,
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
  const dims = root
    .createBuffer(Dims, { qLen: args.qLen, kvLen: args.kvLen, qPosOffset: args.qPosOffset })
    .$usage('uniform');
  const filler = f32Filler(buffers, elem);
  const { dummyF32: _dummy, ...bound } = buffers;
  const bindGroup = root.createBindGroup(attnLayoutFor(elem), {
    ...bound,
    sinks: buffers.sinks ?? filler,
    segs: buffers.segs ?? filler,
    dims,
  });
  return makeHandle(
    pipeline,
    'attn',
    bindGroup,
    Math.ceil((args.qLen * args.qHeads) / WORKGROUP_SIZE),
  );
}

// ── Register-resident row kernels (qLen > 1) ──────────────────────────────
// Built per pipeline config, so headDim enters as a literal and each thread's
// accumulator is an exactly-sized array that stays in registers.
//
// attnRows runs one thread per (row, head). attnRowsSplit + attnRowsMerge
// slice each row's band into chunks that merge through a scratch buffer of
// `chunks` slots per (row, head); chunk 0 carries the sink seed.
// attnRowsSubSplit is the split partial with 32 subgroup lanes per
// (row, head, chunk) instead of one thread.

export const ATTN_ROWS_CHUNK = 96;
export const ATTN_ROWS_MAX_CHUNKS = 16;

const SplitDims = d.struct({ qLen: d.u32, kvLen: d.u32, qPosOffset: d.u32, chunks: d.u32 });

const makeAttnSplitLayout = (elem: Elem) =>
  tgpu.bindGroupLayout({
    q: { storage: d.arrayOf(elem.scalar), access: 'readonly' }, // [qLen, qHeads·headDim]
    k: { storage: d.arrayOf(elem.scalar), access: 'readonly' }, // [kvLen, kvHeads·headDim]
    v: { storage: d.arrayOf(elem.scalar), access: 'readonly' }, // [kvLen, kvHeads·headDim]
    sinks: { storage: d.arrayOf(d.f32), access: 'readonly' }, // [qHeads]; dead when hasSinks=0
    segs: { storage: d.arrayOf(d.f32), access: 'readonly' }, // [qLen, 2]; dead when hasSegs=0
    partials: { storage: d.arrayOf(d.f32), access: 'mutable' }, // [qLen·qHeads·MAX_CHUNKS, 2+hd]
    dims: { uniform: SplitDims },
  });

const attnSplitLayouts = new Map<string, ReturnType<typeof makeAttnSplitLayout>>();
export function attnSplitLayoutFor(elem: Elem): ReturnType<typeof makeAttnSplitLayout> {
  let l = attnSplitLayouts.get(elem.key);
  if (!l) {
    l = makeAttnSplitLayout(elem);
    attnSplitLayouts.set(elem.key, l);
  }
  return l;
}

export const attnRowsSplitLayout = attnSplitLayoutFor(F32_ELEM);

interface AttnGeoCfg {
  qHeads: number;
  kvHeads: number;
  headDim: number;
  windowLeft: number;
  windowRight: number;
  hasSinks: number; // 0 | 1 — selects the softmax seed
  hasSegs: number; // 0 | 1 — clips each row's keys to its own segment range
  qkvPacked?: number; // 0 | 1 — q/k/v as column blocks of one tensor
}

function makeAttnRowsKernel(cfg: AttnGeoCfg, elem: Elem = F32_ELEM) {
  const { qHeads, kvHeads, headDim: hd, windowLeft, windowRight, hasSinks, hasSegs } = cfg;
  const f16 = elem.key === 'f16';
  const L = attnLayoutFor(elem);
  const group = Math.floor(qHeads / kvHeads);
  const qStride = (cfg.qkvPacked ?? 0) > 0 ? (qHeads + 2 * kvHeads) * hd : qHeads * hd;
  const kvStride = (cfg.qkvPacked ?? 0) > 0 ? qStride : kvHeads * hd;
  const kOff = (cfg.qkvPacked ?? 0) > 0 ? qHeads * hd : 0;
  const vOff = (cfg.qkvPacked ?? 0) > 0 ? (qHeads + kvHeads) * hd : 0;
  const qReg = tgpu.privateVar(d.arrayOf(d.f32, hd));
  const acc = tgpu.privateVar(d.arrayOf(d.f32, hd));
  return tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const D = L.$.dims;
    if (gid.x >= D.qLen * qHeads) {
      return;
    }
    const h = gid.x % qHeads;
    const row = d.u32(gid.x / qHeads); // explicit integer division
    const kvh = d.u32(h / group);
    const qBase = row * qStride + h * hd;
    const qPos = D.qPosOffset + row;

    // Softmax state stays f32 whatever the storage dtype is.
    for (const z of std.range(hd)) {
      if (f16) {
        qReg.$[z] = d.f32(L.$.q[qBase + z]!);
      } else {
        qReg.$[z] = L.$.q[qBase + z]!;
      }
      acc.$[z] = d.f32(0);
    }

    let m = d.f32(NEG_SEED);
    let l = d.f32(0);
    if (hasSinks > 0) {
      m = L.$.sinks[h]! * LN2;
      l = d.f32(1);
    }

    // max avoids u32 underflow.
    let lo = std.max(qPos, d.u32(windowLeft)) - windowLeft;
    let end = std.min(qPos + windowRight, D.kvLen - 1);
    if (hasSegs > 0) {
      lo = std.max(lo, d.u32(L.$.segs[2 * row]!));
      end = std.min(end, d.u32(L.$.segs[2 * row + 1]!) - 1);
    }
    for (let j = lo; j <= end; j++) {
      const kBase = j * kvStride + kOff + kvh * hd;
      const vBase = j * kvStride + vOff + kvh * hd;
      let score = d.f32(0);
      if (f16) {
        for (const kk of std.range(hd)) {
          score += qReg.$[kk]! * d.f32(L.$.k[kBase + kk]!);
        }
      } else {
        for (const kk of std.range(hd)) {
          score += qReg.$[kk]! * L.$.k[kBase + kk]!;
        }
      }
      const mNew = std.max(m, score);
      const correction = std.exp(m - mNew);
      const p = std.exp(score - mNew);
      l = l * correction + p;
      m = mNew;
      if (f16) {
        for (const nn of std.range(hd)) {
          acc.$[nn] = acc.$[nn]! * correction + d.f32(L.$.v[vBase + nn]!) * p;
        }
      } else {
        for (const nn of std.range(hd)) {
          acc.$[nn] = acc.$[nn]! * correction + L.$.v[vBase + nn]! * p;
        }
      }
    }

    const invL = std.select(0, 1 / l, l > 0);
    const oBase = row * qHeads * hd + h * hd;
    for (const o of std.range(hd)) {
      if (f16) {
        L.$.out[oBase + o] = d.f16(acc.$[o]! * invL);
      } else {
        L.$.out[oBase + o] = acc.$[o]! * invL;
      }
    }
  });
}

function makeAttnRowsSplitKernel(cfg: AttnGeoCfg, elem: Elem = F32_ELEM) {
  const { qHeads, kvHeads, headDim: hd, windowLeft, windowRight, hasSinks, hasSegs } = cfg;
  const f16 = elem.key === 'f16';
  const S = attnSplitLayoutFor(elem);
  const group = Math.floor(qHeads / kvHeads);
  const qStride = (cfg.qkvPacked ?? 0) > 0 ? (qHeads + 2 * kvHeads) * hd : qHeads * hd;
  const kvStride = (cfg.qkvPacked ?? 0) > 0 ? qStride : kvHeads * hd;
  const kOff = (cfg.qkvPacked ?? 0) > 0 ? qHeads * hd : 0;
  const vOff = (cfg.qkvPacked ?? 0) > 0 ? (qHeads + kvHeads) * hd : 0;
  const qReg = tgpu.privateVar(d.arrayOf(d.f32, hd));
  const acc = tgpu.privateVar(d.arrayOf(d.f32, hd));
  return tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const D = S.$.dims;
    if (gid.x >= D.qLen * qHeads) {
      return;
    }
    const h = gid.x % qHeads;
    const row = d.u32(gid.x / qHeads); // explicit integer division
    const kvh = d.u32(h / group);
    const qBase = row * qStride + h * hd;
    const qPos = D.qPosOffset + row;

    for (const z of std.range(hd)) {
      if (f16) {
        qReg.$[z] = d.f32(S.$.q[qBase + z]!);
      } else {
        qReg.$[z] = S.$.q[qBase + z]!;
      }
      acc.$[z] = d.f32(0);
    }

    let m = d.f32(NEG_SEED);
    let l = d.f32(0);
    if (hasSinks > 0) {
      if (gid.y === 0) {
        m = S.$.sinks[h]! * LN2;
        l = d.f32(1);
      }
    }

    // An empty band would wrap the slice arithmetic below; publish the seed
    // state instead.
    let lo = std.max(qPos, d.u32(windowLeft)) - windowLeft;
    let end = std.min(qPos + windowRight, D.kvLen - 1);
    if (hasSegs > 0) {
      lo = std.max(lo, d.u32(S.$.segs[2 * row]!));
      end = std.min(end, d.u32(S.$.segs[2 * row + 1]!) - 1);
    }
    if (lo > end) {
      const slot = ((row * qHeads + h) * D.chunks + gid.y) * (2 + hd);
      S.$.partials[slot] = m;
      S.$.partials[slot + 1] = l;
      for (const w of std.range(hd)) {
        S.$.partials[slot + 2 + w] = d.f32(0);
      }
      return;
    }

    // The max guard keeps chunkLen ≥ 1, so the arithmetic cannot wrap.
    const range = end - lo + 1;
    const chunkLen = std.max(d.u32((range + D.chunks - 1) / D.chunks), 1);
    const sliceLo = lo + gid.y * chunkLen;
    const sliceEnd = std.min(sliceLo + chunkLen - 1, end);

    for (let j = sliceLo; j <= sliceEnd; j++) {
      const kBase = j * kvStride + kOff + kvh * hd;
      const vBase = j * kvStride + vOff + kvh * hd;
      let score = d.f32(0);
      for (const kk of std.range(hd)) {
        if (f16) {
          score += qReg.$[kk]! * d.f32(S.$.k[kBase + kk]!);
        } else {
          score += qReg.$[kk]! * S.$.k[kBase + kk]!;
        }
      }
      const mNew = std.max(m, score);
      const correction = std.exp(m - mNew);
      const p = std.exp(score - mNew);
      l = l * correction + p;
      m = mNew;
      for (const nn of std.range(hd)) {
        if (f16) {
          acc.$[nn] = acc.$[nn]! * correction + d.f32(S.$.v[vBase + nn]!) * p;
        } else {
          acc.$[nn] = acc.$[nn]! * correction + S.$.v[vBase + nn]! * p;
        }
      }
    }

    const slot = ((row * qHeads + h) * D.chunks + gid.y) * (2 + hd);
    S.$.partials[slot] = m;
    S.$.partials[slot + 1] = l;
    for (const w of std.range(hd)) {
      S.$.partials[slot + 2 + w] = acc.$[w]!;
    }
  });
}

function makeAttnRowsSubgroupSplitKernel(cfg: AttnGeoCfg, elem: Elem = F32_ELEM) {
  const { qHeads, kvHeads, headDim: hd, windowLeft, windowRight, hasSinks, hasSegs } = cfg;
  const f16 = elem.key === 'f16';
  const S = attnSplitLayoutFor(elem);
  const group = Math.floor(qHeads / kvHeads);
  const qStride = (cfg.qkvPacked ?? 0) > 0 ? (qHeads + 2 * kvHeads) * hd : qHeads * hd;
  const kvStride = (cfg.qkvPacked ?? 0) > 0 ? qStride : kvHeads * hd;
  const kOff = (cfg.qkvPacked ?? 0) > 0 ? qHeads * hd : 0;
  const vOff = (cfg.qkvPacked ?? 0) > 0 ? (qHeads + kvHeads) * hd : 0;
  const LANES = 32;
  const cpl = Math.ceil(hd / LANES); // lane-owned columns
  const qv = tgpu.privateVar(d.arrayOf(d.f32, cpl));
  const acc = tgpu.privateVar(d.arrayOf(d.f32, cpl));
  return tgpu.computeFn({
    in: { wid: d.builtin.workgroupId, lid: d.builtin.localInvocationId },
    workgroupSize: [LANES],
  })(({ wid, lid }) => {
    'use gpu';
    const D = S.$.dims;
    const lane = lid.x;
    const h = wid.x % qHeads;
    const row = d.u32(wid.x / qHeads); // explicit integer division
    const kvh = d.u32(h / group);
    const qBase = row * qStride + h * hd;
    const qPos = D.qPosOffset + row;

    // Columns past hd read a clamped index but hold 0, so no per-key branch.
    for (const c of tgpu.unroll(std.range(cpl))) {
      const kk = lane + c * LANES;
      const idx = std.min(kk, d.u32(hd - 1));
      if (f16) {
        qv.$[c] = std.select(d.f32(0), d.f32(S.$.q[qBase + idx]!), kk < hd);
      } else {
        qv.$[c] = std.select(d.f32(0), S.$.q[qBase + idx]!, kk < hd);
      }
      acc.$[c] = d.f32(0);
    }

    // Replicated per lane: every lane tracks identical m/l.
    let m = d.f32(NEG_SEED);
    let l = d.f32(0);
    if (hasSinks > 0) {
      if (wid.y === 0) {
        m = S.$.sinks[h]! * LN2;
        l = d.f32(1);
      }
    }

    const slot = ((row * qHeads + h) * D.chunks + wid.y) * (2 + hd);

    // Row-uniform, so the whole workgroup leaves together.
    let lo = std.max(qPos, d.u32(windowLeft)) - windowLeft;
    let end = std.min(qPos + windowRight, D.kvLen - 1);
    if (hasSegs > 0) {
      lo = std.max(lo, d.u32(S.$.segs[2 * row]!));
      end = std.min(end, d.u32(S.$.segs[2 * row + 1]!) - 1);
    }
    if (lo > end) {
      if (lane === 0) {
        S.$.partials[slot] = m;
        S.$.partials[slot + 1] = l;
      }
      for (const w of tgpu.unroll(std.range(cpl))) {
        const kk = lane + w * LANES;
        if (kk < hd) {
          S.$.partials[slot + 2 + kk] = d.f32(0);
        }
      }
      return;
    }

    const range = end - lo + 1;
    const chunkLen = std.max(d.u32((range + D.chunks - 1) / D.chunks), 1);
    const sliceLo = lo + wid.y * chunkLen;
    const sliceEnd = std.min(sliceLo + chunkLen - 1, end);

    for (let j = sliceLo; j <= sliceEnd; j++) {
      const kBase = j * kvStride + kOff + kvh * hd;
      const vBase = j * kvStride + vOff + kvh * hd;
      let partial = d.f32(0);
      for (const cc of tgpu.unroll(std.range(cpl))) {
        const kk = lane + cc * LANES;
        const idx = std.min(kk, d.u32(hd - 1));
        if (f16) {
          partial += qv.$[cc]! * d.f32(S.$.k[kBase + idx]!);
        } else {
          partial += qv.$[cc]! * S.$.k[kBase + idx]!;
        }
      }
      const score = std.subgroupAdd(partial);
      const mNew = std.max(m, score);
      const correction = std.exp(m - mNew);
      const p = std.exp(score - mNew);
      l = l * correction + p;
      m = mNew;
      for (const nn of tgpu.unroll(std.range(cpl))) {
        const kk = lane + nn * LANES;
        const idx = std.min(kk, d.u32(hd - 1));
        if (f16) {
          acc.$[nn] = acc.$[nn]! * correction + d.f32(S.$.v[vBase + idx]!) * p;
        } else {
          acc.$[nn] = acc.$[nn]! * correction + S.$.v[vBase + idx]! * p;
        }
      }
    }

    if (lane === 0) {
      S.$.partials[slot] = m;
      S.$.partials[slot + 1] = l;
    }
    for (const w of tgpu.unroll(std.range(cpl))) {
      const kk = lane + w * LANES;
      if (kk < hd) {
        S.$.partials[slot + 2 + kk] = acc.$[w]!;
      }
    }
  });
}

const MergeDims = d.struct({ qLen: d.u32, chunks: d.u32 });

const makeAttnMergeLayout = (elem: Elem) =>
  tgpu.bindGroupLayout({
    partials: { storage: d.arrayOf(d.f32), access: 'readonly' }, // [qLen·qHeads·MAX_CHUNKS, 2+hd]
    out: { storage: d.arrayOf(elem.scalar), access: 'mutable' }, // [qLen, qHeads·headDim]
    dims: { uniform: MergeDims },
  });

const attnMergeLayouts = new Map<string, ReturnType<typeof makeAttnMergeLayout>>();
export function attnMergeLayoutFor(elem: Elem): ReturnType<typeof makeAttnMergeLayout> {
  let l = attnMergeLayouts.get(elem.key);
  if (!l) {
    l = makeAttnMergeLayout(elem);
    attnMergeLayouts.set(elem.key, l);
  }
  return l;
}

export const attnRowsMergeLayout = attnMergeLayoutFor(F32_ELEM);

function makeAttnMergeKernel(cfg: AttnGeoCfg, elem: Elem = F32_ELEM) {
  const { qHeads, headDim: hd } = cfg;
  const f16 = elem.key === 'f16';
  const MERGE = attnMergeLayoutFor(elem);
  return tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const D = MERGE.$.dims;
    if (gid.x >= D.qLen * qHeads) {
      return;
    }
    const h = gid.x % qHeads;
    const row = d.u32(gid.x / qHeads); // explicit integer division
    const stride = 2 + hd;
    const rowBase = (row * qHeads + h) * D.chunks;

    let M = d.f32(NEG_SEED);
    for (let c = d.u32(0); c < D.chunks; c++) {
      M = std.max(M, MERGE.$.partials[(rowBase + c) * stride]!);
    }

    let L = d.f32(0);
    for (let c = d.u32(0); c < D.chunks; c++) {
      const slot = (rowBase + c) * stride;
      L += MERGE.$.partials[slot + 1]! * std.exp(MERGE.$.partials[slot]! - M);
    }

    const invL = std.select(0, 1 / L, L > 0);
    const oBase = row * qHeads * hd + h * hd;
    for (const o of std.range(hd)) {
      let a = d.f32(0);
      for (let c = d.u32(0); c < D.chunks; c++) {
        const slot = (rowBase + c) * stride;
        a += MERGE.$.partials[slot + 2 + o]! * std.exp(MERGE.$.partials[slot]! - M);
      }
      if (f16) {
        MERGE.$.out[oBase + o] = d.f16(a * invL);
      } else {
        MERGE.$.out[oBase + o] = a * invL;
      }
    }
  });
}

export function createAttnRowsPipeline(root: TgpuRoot, cfg: AttnGeoCfg, elem: Elem = F32_ELEM) {
  return root.createComputePipeline({ compute: makeAttnRowsKernel(cfg, elem) });
}

export function createAttnRowsSplitPipeline(
  root: TgpuRoot,
  cfg: AttnGeoCfg,
  elem: Elem = F32_ELEM,
) {
  return root.createComputePipeline({ compute: makeAttnRowsSplitKernel(cfg, elem) });
}

export function createAttnRowsSubgroupSplitPipeline(
  root: TgpuRoot,
  cfg: AttnGeoCfg,
  elem: Elem = F32_ELEM,
) {
  return root.createComputePipeline({ compute: makeAttnRowsSubgroupSplitKernel(cfg, elem) });
}

export function createAttnRowsMergePipeline(
  root: TgpuRoot,
  cfg: AttnGeoCfg,
  elem: Elem = F32_ELEM,
) {
  return root.createComputePipeline({ compute: makeAttnMergeKernel(cfg, elem) });
}

export function attnRowsHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createAttnRowsPipeline>,
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
  const dims = root
    .createBuffer(Dims, { qLen: args.qLen, kvLen: args.kvLen, qPosOffset: args.qPosOffset })
    .$usage('uniform');
  const filler = f32Filler(buffers, elem);
  const { dummyF32: _dummy, ...bound } = buffers;
  const bindGroup = root.createBindGroup(attnLayoutFor(elem), {
    ...bound,
    sinks: buffers.sinks ?? filler,
    segs: buffers.segs ?? filler,
    dims,
  });
  // One thread per (q row, q head), register-resident accumulator.
  return makeHandle(
    pipeline,
    'attnRows',
    bindGroup,
    Math.ceil((args.qLen * args.qHeads) / WORKGROUP_SIZE),
  );
}

export function attnRowsSplitHandles(
  root: TgpuRoot,
  partial:
    | ReturnType<typeof createAttnRowsSplitPipeline>
    | ReturnType<typeof createAttnRowsSubgroupSplitPipeline>,
  merge: ReturnType<typeof createAttnRowsMergePipeline>,
  args: {
    qLen: number;
    kvLen: number;
    qPosOffset: number;
    qHeads: number;
    chunks: number;
    subgroup?: boolean;
  },
  buffers: {
    q: FloatBuffer;
    k: FloatBuffer;
    v: FloatBuffer;
    out: FloatBuffer;
    partials: F32Buffer;
    sinks?: F32Buffer;
    segs?: F32Buffer;
    dummyF32: F32Buffer;
  },
  elem: Elem = F32_ELEM,
): KernelHandle[] {
  const splitDims = root
    .createBuffer(SplitDims, {
      qLen: args.qLen,
      kvLen: args.kvLen,
      qPosOffset: args.qPosOffset,
      chunks: args.chunks,
    })
    .$usage('uniform');
  const splitBind = root.createBindGroup(attnSplitLayoutFor(elem), {
    q: buffers.q,
    k: buffers.k,
    v: buffers.v,
    sinks: buffers.sinks ?? buffers.dummyF32,
    segs: buffers.segs ?? buffers.dummyF32,
    partials: buffers.partials,
    dims: splitDims,
  });
  const mergeDims = root
    .createBuffer(MergeDims, { qLen: args.qLen, chunks: args.chunks })
    .$usage('uniform');
  const mergeBind = root.createBindGroup(attnMergeLayoutFor(elem), {
    partials: buffers.partials,
    out: buffers.out,
    dims: mergeDims,
  });
  return [
    // Grid y is the key chunk, x is (row, head).
    args.subgroup
      ? makeHandle(partial, 'attnRowsSubSplit', splitBind, [args.qLen * args.qHeads, args.chunks])
      : makeHandle(partial, 'attnRowsSplit', splitBind, [
          Math.ceil((args.qLen * args.qHeads) / WORKGROUP_SIZE),
          args.chunks,
        ]),
    makeHandle(
      merge,
      'attnRowsMerge',
      mergeBind,
      Math.ceil((args.qLen * args.qHeads) / WORKGROUP_SIZE),
    ),
  ];
}
