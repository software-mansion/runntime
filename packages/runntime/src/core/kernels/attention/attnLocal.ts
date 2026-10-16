/** Local attention over a fused qkv slab [T, 3·heads·headDim]: query t of
 *  each head sees keys t − window + 1 … t, nothing ahead.
 *
 *      score(t, j) = cap · tanh((q_t · qScale) · (k_{t−j} · kScale + rel_j) / cap)
 *
 *  for distance j, then softmax over the keys and a weighted sum of their
 *  values. rel [window, heads·headDim] holds one bias row per distance, and
 *  qScale [headDim] scales each column of q.
 *
 *  One workgroup per (row, head). Each lane takes vec4 words of the head and
 *  adds its share of every score to workgroup memory; one lane per key then
 *  adds up the shares. Sums in f32. */

import tgpu, { d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import {
  type F32Buffer,
  type FloatBuffer,
  flatWorkgroupId,
  type KernelHandle,
  makeHandle,
} from '../common.ts';
import { cachedBindGroup, cachedUniform } from '../../gpu/dispatchCache.ts';
import { type Elem, F32_ELEM } from '../elem.ts';

/** Lanes per workgroup. */
const LANES = 32;
/** Most keys a query sees: one lane adds up each key's score. */
export const ATTN_LOCAL_MAX_WINDOW = LANES;

const Dims = d.struct({ rows: d.u32, kScale: d.f32, softcap: d.f32 });

const makeLayout = (elem: Elem) =>
  tgpu.bindGroupLayout({
    qkv: { storage: d.arrayOf(elem.vec4), access: 'readonly' },
    rel: { storage: d.arrayOf(d.vec4f), access: 'readonly' },
    qScale: { storage: d.arrayOf(d.vec4f), access: 'readonly' },
    out: { storage: d.arrayOf(elem.vec4), access: 'mutable' },
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

export interface AttnLocalCfg {
  heads: number;
  headDim: number;
  window: number;
}

function makeKernel(cfg: AttnLocalCfg, elem: Elem) {
  const { heads, window } = cfg;
  const L = layoutFor(elem);
  const store = elem.vec4;
  const words = cfg.headDim / 4;
  const width = heads * words;
  const rowWords = 3 * width;
  const shares = tgpu.workgroupVar(d.arrayOf(d.f32, window * LANES));
  const scores = tgpu.workgroupVar(d.arrayOf(d.f32, window));
  return tgpu.computeFn({
    in: { wid: d.builtin.workgroupId, lid: d.builtin.localInvocationId },
    workgroupSize: [LANES],
  })(({ wid, lid }) => {
    'use gpu';
    const D = L.$.dims;
    const g = flatWorkgroupId(wid);
    // Past 65535 groups the grid is 2-D, and its last row runs past the end.
    if (g >= D.rows * heads) return;
    const t = std.intdiv(g, d.u32(heads));
    const h = g % heads;
    const tid = lid.x;
    const qBase = t * rowWords + h * words;

    for (let j = d.u32(0); j < window; j++) {
      let s = d.f32(0);
      if (j <= t) {
        const kBase = (t - j) * rowWords + width + h * words;
        const rBase = j * width + h * words;
        for (let w = tid; w < words; w += LANES) {
          const q = d.vec4f(L.$.qkv[qBase + w]!) * L.$.qScale[w]!;
          const k = d.vec4f(L.$.qkv[kBase + w]!) * D.kScale + L.$.rel[rBase + w]!;
          s += std.dot(q, k);
        }
      }
      shares.$[j * LANES + tid] = s;
    }
    std.workgroupBarrier();
    if (tid < window) {
      let s = d.f32(0);
      for (let i = d.u32(0); i < LANES; i++) {
        s += shares.$[tid * LANES + i]!;
      }
      scores.$[tid] = D.softcap * std.tanh(s / D.softcap);
    }
    std.workgroupBarrier();

    // Key t − j exists for j <= t, so distance 0 always does.
    const n = std.min(d.u32(window), t + 1);
    let m = scores.$[0]!;
    for (let j = d.u32(1); j < n; j++) {
      m = std.max(m, scores.$[j]!);
    }
    let sum = d.f32(0);
    for (let j = d.u32(0); j < n; j++) {
      sum += std.exp(scores.$[j]! - m);
    }
    for (let w = tid; w < words; w += LANES) {
      let acc = d.vec4f();
      for (let j = d.u32(0); j < n; j++) {
        const v = d.vec4f(L.$.qkv[(t - j) * rowWords + 2 * width + h * words + w]!);
        acc += std.exp(scores.$[j]! - m) * v;
      }
      L.$.out[t * width + h * words + w] = store(acc / sum);
    }
  });
}

const kernels = new Map<string, ReturnType<typeof makeKernel>>();
function kernelFor(cfg: AttnLocalCfg, elem: Elem): ReturnType<typeof makeKernel> {
  const key = `${elem.key}:${cfg.heads}:${cfg.headDim}:${cfg.window}`;
  let k = kernels.get(key);
  if (!k) {
    k = makeKernel(cfg, elem);
    kernels.set(key, k);
  }
  return k;
}

export function createAttnLocalPipeline(root: TgpuRoot, cfg: AttnLocalCfg, elem: Elem = F32_ELEM) {
  return root.createComputePipeline({ compute: kernelFor(cfg, elem) });
}

export function attnLocalHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createAttnLocalPipeline>,
  args: { rows: number; heads: number; kScale: number; softcap: number },
  buffers: { qkv: FloatBuffer; rel: F32Buffer; qScale: F32Buffer; out: FloatBuffer },
  elem: Elem = F32_ELEM,
): KernelHandle {
  const { rows, kScale, softcap } = args;
  const dims = cachedUniform(root, Dims, { rows, kScale, softcap });
  const bindGroup = cachedBindGroup(root, layoutFor(elem), { ...buffers, dims });
  // One workgroup per (row, head).
  return makeHandle(pipeline, 'attnLocal', bindGroup, rows * args.heads);
}
