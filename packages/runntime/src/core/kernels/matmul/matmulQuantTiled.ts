/** Matmul with a quantW weight: out = act(a·W + bias) + addend, in 32×32
 *  output tiles. W holds int8 or int4 values in the blocked-unit layout, times
 *  f32 group scales. a, bias, addend and out are f16 or f32; sums are f32.
 *
 *  Each step unpacks a slice of W into workgroup memory as floats, so the
 *  inner loop is a plain float matmul with matmulTiledVec4's tiling. A
 *  loader reads a and another reads the scales, so conv1d runs on this kernel
 *  too, its scales where its torch weight has them. By default row k of W
 *  uses scale row k / groupSize. */

import tgpu, { d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import { type KernelHandle, makeHandle } from '../common.ts';
import { cachedBindGroup, cachedUniform } from '../../gpu/dispatchCache.ts';
import { activationFor, actSlot } from '../activations.ts';
import { type Elem, F32_ELEM } from '../elem.ts';
import { type LoadA, stridedLoadA } from './matmulTiledVec4.ts';
import {
  columnWord,
  deqRow,
  type MatmulQuantBuffers,
  type MatmulQuantCfg,
} from '../quantCommon.ts';

const Dims = d.struct({ m: d.u32, aRows: d.u32 });

function makeLayout(elem: Elem) {
  return tgpu.bindGroupLayout({
    a: { storage: d.arrayOf(elem.vec4), access: 'readonly' },
    w: { storage: d.arrayOf(d.u32), access: 'readonly' },
    scales: { storage: d.arrayOf(d.vec4f), access: 'readonly' }, // read through scaleAt
    bias: { storage: d.arrayOf(elem.vec4), access: 'readonly' }, // [N/4]; dead when hasBias=0
    addend: { storage: d.arrayOf(elem.vec4), access: 'readonly' }, // [M,N/4]; dead when hasAdd=0
    out: { storage: d.arrayOf(elem.vec4), access: 'mutable' },
    dims: { uniform: Dims },
  });
}

const layouts = new Map<string, ReturnType<typeof makeLayout>>();
function layoutFor(elem: Elem): ReturnType<typeof makeLayout> {
  let l = layouts.get(elem.key);
  if (!l) {
    l = makeLayout(elem);
    layouts.set(elem.key, l);
  }
  return l;
}

const loadASlot = tgpu.slot<LoadA>();

/** The scales of row `row` of W in columns 4·col4 to 4·col4 + 3. */
export type ScaleAt = (row: number, col4: number) => d.v4f;
const scaleAtSlot = tgpu.slot<ScaleAt>();

/** What a scale loader reads: the scales, bound as vec4s. */
export interface ScaleSource {
  readonly $: { readonly scales: d.v4f[] };
}

/** The layout every pipeline of this kernel binds, so a loader can read a,
 *  aRows and the scales. */
export function matmulQuantTiledLayout(elem: Elem): ReturnType<typeof makeLayout> {
  return layoutFor(elem);
}

export interface MatmulQuantTiledCfg extends MatmulQuantCfg {
  /** Distance between rows of a, in values. K when omitted. */
  aStride?: number;
}

/** Scales [K / groupSize, N]: one per group of groupSize rows in each column. */
function groupedScales(elem: Elem, n: number, groupSize: number): ScaleAt {
  const layout = layoutFor(elem);
  const n4 = n / 4;
  return (row: number, col4: number) => {
    'use gpu';
    return d.vec4f(layout.$.scales[std.intdiv(row, groupSize) * n4 + col4]!);
  };
}

// Each workgroup computes a BM×BN tile of out, each thread TM rows × TN
// groups of 4 columns.
const BM = 32;
const BN = 32;
const TM = 4;
const TN = 1;

/** K per step of the tile loop: K must be a multiple of it. */
export const MATMUL_QUANT_TILED_BK = 8;
const BK = MATMUL_QUANT_TILED_BK;

/** Whether a [M,K]·[K,N] quantW matmul can use this kernel. groupSize also
 *  checks the default scale groups. */
export function matmulQuantTiledEligible(k: number, n: number, groupSize?: number): boolean {
  const grouped = groupSize === undefined || (groupSize % 4 === 0 && k % groupSize === 0);
  return k % BK === 0 && n % 4 === 0 && grouped;
}

function makeMatmulQuantTiledKernel(cfg: MatmulQuantTiledCfg, elem: Elem) {
  const { k, n, bits, hasBias, hasAdd } = cfg;
  const layout = layoutFor(elem);
  const storeVec4 = elem.vec4;
  const bk4 = BK / 4;
  const bn4 = BN / 4;
  // Threads per row of the tile, each owning TN units spaced colThreads apart.
  const colThreads = bn4 / TN;
  const lanes = colThreads * (BM / TM);
  const k4 = k / 4;
  const n4 = n / 4;
  const tiles = k / BK;
  if (BM * bk4 !== lanes || BK * bn4 !== lanes) {
    throw new Error('matmulQuantTiled: each tile needs one vec4 per thread');
  }
  // Words per packed 4×4 block of W.
  const tileWords = bits / 2;
  const cw1 = columnWord(1, bits);
  const cw2 = columnWord(2, bits);
  const cw3 = columnWord(3, bits);
  const tileA = tgpu.workgroupVar(d.arrayOf(elem.vec4, BM * bk4));
  const tileB = tgpu.workgroupVar(d.arrayOf(elem.vec4, BK * bn4));
  return tgpu.computeFn({
    in: { lid: d.builtin.localInvocationIndex, wid: d.builtin.workgroupId },
    workgroupSize: [lanes],
  })(({ lid, wid }) => {
    'use gpu';
    const m = layout.$.dims.m;
    const tx = lid % colThreads;
    const ty = std.intdiv(lid, colThreads);
    const mBase = wid.y * BM;
    const nBase4 = wid.x * bn4;

    const accs = d.arrayOf(d.vec4f, TM * TN)();
    const bv = d.arrayOf(d.vec4f, 4 * TN)();
    // Thread lid fills slot lid of each tile.
    const am = mBase + std.intdiv(lid, bk4);
    const aCol = lid % bk4;
    const bRow = std.intdiv(lid, bn4);
    const bc4 = nBase4 + (lid % bn4);
    let ra = storeVec4();
    let w0 = d.u32(0);
    let w1 = d.u32(0);
    let w2 = d.u32(0);
    let w3 = d.u32(0);
    let scale = d.vec4f();
    if (am < m) {
      ra = storeVec4(loadASlot.$(am, aCol));
    }
    if (bc4 < n4) {
      const wb = (bc4 * k4 + std.intdiv(bRow, 4)) * tileWords;
      w0 = layout.$.w[wb]!;
      w1 = layout.$.w[wb + cw1]!;
      w2 = layout.$.w[wb + cw2]!;
      w3 = layout.$.w[wb + cw3]!;
      scale = scaleAtSlot.$(bRow, bc4);
    }
    for (const t of std.range(tiles)) {
      tileA.$[lid] = storeVec4(ra);
      tileB.$[lid] = storeVec4(deqRow(w0, w1, w2, w3, bRow % 4, bits) * scale);
      std.workgroupBarrier();
      // The next tile's loads are in flight while this one computes.
      const tn = t + 1;
      if (tn < tiles) {
        ra = storeVec4();
        if (am < m) {
          ra = storeVec4(loadASlot.$(am, tn * bk4 + aCol));
        }
        if (bc4 < n4) {
          const bk = tn * BK + bRow;
          const wb = (bc4 * k4 + std.intdiv(bk, 4)) * tileWords;
          w0 = layout.$.w[wb]!;
          w1 = layout.$.w[wb + cw1]!;
          w2 = layout.$.w[wb + cw2]!;
          w3 = layout.$.w[wb + cw3]!;
          scale = scaleAtSlot.$(bk, bc4);
        }
      }

      for (const kv of tgpu.unroll(std.range(bk4))) {
        for (const j of tgpu.unroll(std.range(TN))) {
          for (const r of tgpu.unroll(std.range(4))) {
            bv[j * 4 + r] = d.vec4f(tileB.$[(kv * 4 + r) * bn4 + tx + j * colThreads]!);
          }
        }
        for (const i of tgpu.unroll(std.range(TM))) {
          const av = d.vec4f(tileA.$[(ty * TM + i) * bk4 + kv]!);
          for (const j of tgpu.unroll(std.range(TN))) {
            accs[i * TN + j] =
              accs[i * TN + j]! +
              av.x * bv[j * 4]! +
              av.y * bv[j * 4 + 1]! +
              av.z * bv[j * 4 + 2]! +
              av.w * bv[j * 4 + 3]!;
          }
        }
      }
      std.workgroupBarrier();
    }

    for (const j of tgpu.unroll(std.range(TN))) {
      const c4 = nBase4 + tx + j * colThreads;
      if (c4 < n4) {
        let bias = d.vec4f();
        if (hasBias > 0) {
          bias = d.vec4f(layout.$.bias[c4]!);
        }
        for (const i of tgpu.unroll(std.range(TM))) {
          const row = mBase + ty * TM + i;
          if (row < m) {
            let v = actSlot.$(accs[i * TN + j]! + bias);
            if (hasAdd > 0) {
              v += d.vec4f(layout.$.addend[row * n4 + c4]!);
            }
            layout.$.out[row * n4 + c4] = storeVec4(v);
          }
        }
      }
    }
  });
}

/** One pipeline per dtype, K, N, bits, group size, loaders, epilogue and
 *  activation. `loadA` reads a; by default it reads rows aStride apart.
 *  `scaleAt` reads the scales; by default [K / groupSize, N]. */
export function createMatmulQuantTiledPipeline(
  root: TgpuRoot,
  cfg: MatmulQuantTiledCfg,
  elem: Elem = F32_ELEM,
  loadA: LoadA = stridedLoadA(elem, cfg.aStride ?? cfg.k, layoutFor(elem)),
  scaleAt: ScaleAt = groupedScales(elem, cfg.n, cfg.groupSize),
) {
  return root
    .with(actSlot, activationFor(cfg.act))
    .with(loadASlot, loadA)
    .with(scaleAtSlot, scaleAt)
    .createComputePipeline({ compute: makeMatmulQuantTiledKernel(cfg, elem) });
}

/** One workgroup per 32×32 output tile. aRows is how many rows a's buffer
 *  holds, for loaders that bound-check it. */
export function matmulQuantTiledHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createMatmulQuantTiledPipeline>,
  m: number,
  n: number,
  buffers: MatmulQuantBuffers,
  elem: Elem = F32_ELEM,
  aRows = m,
): KernelHandle {
  const dims = cachedUniform(root, Dims, { m, aRows });
  // Absent epilogue operands alias `a`; those pipelines never read them.
  const bindGroup = cachedBindGroup(root, layoutFor(elem), {
    ...buffers,
    bias: buffers.bias ?? buffers.a,
    addend: buffers.addend ?? buffers.a,
    dims,
  });
  return makeHandle(pipeline, 'matmulQuantTiled', bindGroup, [
    Math.ceil(n / BN),
    Math.ceil(m / BM),
  ]);
}
