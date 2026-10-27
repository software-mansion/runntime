/** Matmul with a quantW weight: out = act(a·W + bias) + addend, in 32×32
 *  output tiles. W holds int8 or int4 values in the blocked-unit layout, times
 *  f32 group scales. a, bias, addend and out are f16 or f32; sums are f32.
 *
 *  Each step unpacks a slice of W into workgroup memory as floats, so the
 *  inner loop is a plain float matmul with matmulTiledVec4's tiling. */

import tgpu, { d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import {
  type F32Buffer,
  type FloatBuffer,
  type KernelHandle,
  makeHandle,
  type U32Buffer,
} from '../common.ts';
import { cachedBindGroup, cachedUniform } from '../../gpu/dispatchCache.ts';
import { activationFor, actSlot } from '../activations.ts';
import { type Elem, F32_ELEM } from '../elem.ts';
import { columnWord, deqRow } from '../quantCommon.ts';

const Dims = d.struct({ m: d.u32 });

function makeLayout(elem: Elem) {
  return tgpu.bindGroupLayout({
    a: { storage: d.arrayOf(elem.vec4), access: 'readonly' },
    w: { storage: d.arrayOf(d.u32), access: 'readonly' },
    scales: { storage: d.arrayOf(d.vec4f), access: 'readonly' }, // [K/group, N/4]
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

// Each workgroup computes a BM×BN tile of out, each thread TM rows × TN
// groups of 4 columns.
const BM = 32;
const BN = 32;
const TM = 4;
const TN = 1;

/** K per step of the tile loop: K must be a multiple of it. */
export const MATMUL_QUANT_TILED_BK = 8;
const BK = MATMUL_QUANT_TILED_BK;

/** Whether a [M,K]·[K,N] quantW matmul can use this kernel. */
export function matmulQuantTiledEligible(k: number, n: number, groupSize: number): boolean {
  return k % BK === 0 && n % 4 === 0 && groupSize % 4 === 0 && k % groupSize === 0;
}

export interface MatmulQuantTiledCfg {
  k: number;
  n: number;
  bits: number;
  groupSize: number;
  hasBias: number;
  hasAdd: number;
  act?: number;
}

function makeMatmulQuantTiledKernel(cfg: MatmulQuantTiledCfg, elem: Elem) {
  const { k, n, bits, groupSize, hasBias, hasAdd } = cfg;
  const layout = layoutFor(elem);
  const storeVec4 = elem.vec4;
  const bk4 = BK / 4;
  const bn4 = BN / 4;
  // Threads per row of the tile, each owning TN units spaced colThreads apart.
  const colThreads = bn4 / TN;
  const lanes = colThreads * (BM / TM);
  const k4 = k / 4;
  const n4 = n / 4;
  const aVecs = Math.ceil((BM * bk4) / lanes);
  const bUnits = Math.ceil((BK * bn4) / lanes);
  const tiles = k / BK;
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
    for (const t of std.range(tiles)) {
      for (const i of tgpu.unroll(std.range(aVecs))) {
        const e = lid + i * lanes;
        if (e < BM * bk4) {
          const am = mBase + std.intdiv(e, bk4);
          let v = storeVec4();
          if (am < m) {
            v = storeVec4(layout.$.a[am * k4 + t * bk4 + (e % bk4)]!);
          }
          tileA.$[e] = storeVec4(v);
        }
      }
      for (const i of tgpu.unroll(std.range(bUnits))) {
        const e = lid + i * lanes;
        if (e < BK * bn4) {
          const bk = t * BK + std.intdiv(e, bn4);
          const bc4 = nBase4 + (e % bn4);
          let wv = storeVec4();
          if (bc4 < n4) {
            const wb = (bc4 * k4 + std.intdiv(bk, 4)) * tileWords;
            const q = deqRow(
              layout.$.w[wb]!,
              layout.$.w[wb + cw1]!,
              layout.$.w[wb + cw2]!,
              layout.$.w[wb + cw3]!,
              bk % 4,
              bits,
            );
            wv = storeVec4(q * layout.$.scales[std.intdiv(bk, groupSize) * n4 + bc4]!);
          }
          tileB.$[e] = storeVec4(wv);
        }
      }
      std.workgroupBarrier();

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

/** One pipeline per dtype, K, N, bits, group size, epilogue and activation. */
export function createMatmulQuantTiledPipeline(
  root: TgpuRoot,
  cfg: MatmulQuantTiledCfg,
  elem: Elem = F32_ELEM,
) {
  return root
    .with(actSlot, activationFor(cfg.act))
    .createComputePipeline({ compute: makeMatmulQuantTiledKernel(cfg, elem) });
}

/** One workgroup per 32×32 output tile. */
export function matmulQuantTiledHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createMatmulQuantTiledPipeline>,
  m: number,
  n: number,
  buffers: {
    a: FloatBuffer;
    w: U32Buffer;
    scales: F32Buffer;
    out: FloatBuffer;
    bias?: FloatBuffer;
    addend?: FloatBuffer;
  },
  elem: Elem = F32_ELEM,
): KernelHandle {
  const dims = cachedUniform(root, Dims, { m });
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
