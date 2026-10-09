/** MoE expert matmuls on quantW weights. The routing comes straight from
 *  topk's packed output [T, 2·SLOTS]: expert ids, then gate weights.
 *
 *  moeUp runs both input projections and the clamped SwiGLU for every
 *  (token, slot) in one dispatch. moeDown runs the output projection for
 *  every slot, mixes the slots by gate weight and adds the residual.
 *
 *  A workgroup computes UNITS groups of 4 output columns for one row. K is
 *  split across K_LANES lanes, so neighbouring lanes read neighbouring weight
 *  tiles, and a fixed tree in workgroup memory adds the lanes up. Sums in f32.
 *
 *  Based on webgpu-kernels/com.microsoft.QMoE (qmoe-fc1-activation-gemv). */

import tgpu, { d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import {
  type F32Buffer,
  type KernelHandle,
  makeHandle,
  MAX_WORKGROUPS_PER_DIM,
  type U32Buffer,
} from '../common.ts';
import { cachedBindGroup, cachedUniform } from '../../gpu/dispatchCache.ts';
import { deqVec4 } from '../quantCommon.ts';

/** Experts per token: the k of the topk that feeds these kernels. */
export const MOE_SLOTS = 4;

const K_LANES = 32;
const UNITS = 8;
const WG = K_LANES * UNITS;

const SWIGLU_LIMIT = 7.0;
const SWIGLU_ALPHA = 1.702;

const LANE_STRIDES = Array.from(
  { length: Math.log2(K_LANES) },
  (_, level) => K_LANES >> (level + 1),
);

const Dims = d.struct({ tokens: d.u32 });

const Config = d.struct({
  k: d.u32,
  n: d.u32,
  n4: d.u32, // n/4 — 4-column groups per row
  k4count: d.u32, // k/4 — 4-row chunks per column
  tileWords: d.u32, // bits/2 — words per 16-value tile
  g4: d.u32, // groupSize/4 — chunks per scale group
  kg: d.u32, // k/groupSize — scale rows per expert
  bits: d.u32,
  experts: d.u32,
});
// Degenerate defaults, so an unconfigured pipeline fails a correctness check.
const config = tgpu.accessor(Config, {
  k: 0,
  n: 0,
  n4: 0,
  k4count: 0,
  tileWords: 0,
  g4: 1,
  kg: 0,
  bits: 0,
  experts: 0,
});

const partialA = tgpu.workgroupVar(d.arrayOf(d.vec4f, WG));
const partialB = tgpu.workgroupVar(d.arrayOf(d.vec4f, WG));

/** Dots x with each of the 4 columns of one 16-value weight tile. */
const tileDots = (w0: number, w1: number, w2: number, w3: number, x: d.v4f) => {
  'use gpu';
  const C = config.$;
  const b4 = C.bits * 4;
  return d.vec4f(
    std.dot(x, deqVec4(w0, 0, C.bits)),
    std.dot(x, deqVec4(w1, b4 % 32, C.bits)),
    std.dot(x, deqVec4(w2, (2 * b4) % 32, C.bits)),
    std.dot(x, deqVec4(w3, (3 * b4) % 32, C.bits)),
  );
};

/** Word offset of column c inside a weight tile. */
const colWord = (c: number) => {
  'use gpu';
  return d.u32((c * config.$.bits * 4) / 32);
};

/** Adds partialA (and partialB) over the K lanes of each unit, so lane 0
 *  holds the total. It has barriers: every thread must call it. */
const mergeLanes = (lid: number, lane: number, both: boolean) => {
  'use gpu';
  std.workgroupBarrier();
  for (const stride of tgpu.unroll(LANE_STRIDES)) {
    if (lane < stride) {
      partialA.$[lid] = partialA.$[lid]! + partialA.$[lid + stride]!;
      if (both) {
        partialB.$[lid] = partialB.$[lid]! + partialB.$[lid + stride]!;
      }
    }
    std.workgroupBarrier();
  }
};

// Float buffers are read as vec4: K and N are multiples of 4.
export const moeUpLayout = tgpu.bindGroupLayout({
  x: { storage: d.arrayOf(d.vec4f), access: 'readonly' }, // [T, K]
  route: { storage: d.arrayOf(d.f32), access: 'readonly' }, // [T, 2·SLOTS]
  gluW: { storage: d.arrayOf(d.u32), access: 'readonly' },
  gluScales: { storage: d.arrayOf(d.vec4f), access: 'readonly' }, // [E·KG, N]
  linW: { storage: d.arrayOf(d.u32), access: 'readonly' },
  linScales: { storage: d.arrayOf(d.vec4f), access: 'readonly' },
  // glu rows, then lin rows: [2E, N]. One buffer instead of two keeps the
  // kernel at 8 storage buffers, the WebGPU default limit.
  bias: { storage: d.arrayOf(d.vec4f), access: 'readonly' },
  out: { storage: d.arrayOf(d.vec4f), access: 'mutable' }, // [T·SLOTS, N]
  dims: { uniform: Dims },
});

const moeUpKernel = tgpu.computeFn({
  in: { wid: d.builtin.workgroupId, lid: d.builtin.localInvocationIndex },
  workgroupSize: [WG],
})(({ wid, lid }) => {
  'use gpu';
  const C = config.$;
  const lane = lid % K_LANES;
  const unit = wid.x * UNITS + d.u32(lid / K_LANES);
  // The grid is at most 65535 rows high; longer inputs loop.
  const rows = moeUpLayout.$.dims.tokens * MOE_SLOTS;
  for (let row = wid.y; row < rows; row += MAX_WORKGROUPS_PER_DIM) {
    const t = d.u32(row / MOE_SLOTS);
    const expert = d.u32(moeUpLayout.$.route[t * 2 * MOE_SLOTS + (row % MOE_SLOTS)]!);

    let glu = d.vec4f();
    let lin = d.vec4f();
    if (unit < C.n4) {
      const wBase = (expert * C.n4 + unit) * C.k4count * C.tileWords;
      const sRow = expert * C.kg;
      for (let k4 = lane; k4 < C.k4count; k4 += K_LANES) {
        const x = moeUpLayout.$.x[t * C.k4count + k4]!;
        const s = (sRow + d.u32(k4 / C.g4)) * C.n4 + unit;
        const tb = wBase + k4 * C.tileWords;
        const g = tileDots(
          moeUpLayout.$.gluW[tb]!,
          moeUpLayout.$.gluW[tb + colWord(1)]!,
          moeUpLayout.$.gluW[tb + colWord(2)]!,
          moeUpLayout.$.gluW[tb + colWord(3)]!,
          x,
        );
        const l = tileDots(
          moeUpLayout.$.linW[tb]!,
          moeUpLayout.$.linW[tb + colWord(1)]!,
          moeUpLayout.$.linW[tb + colWord(2)]!,
          moeUpLayout.$.linW[tb + colWord(3)]!,
          x,
        );
        glu += g * moeUpLayout.$.gluScales[s]!;
        lin += l * moeUpLayout.$.linScales[s]!;
      }
    }
    partialA.$[lid] = d.vec4f(glu);
    partialB.$[lid] = d.vec4f(lin);
    mergeLanes(lid, lane, true);

    if (lane === 0 && unit < C.n4) {
      const b = expert * C.n4 + unit;
      const g = std.min(partialA.$[lid]! + moeUpLayout.$.bias[b]!, d.vec4f(SWIGLU_LIMIT));
      const l = std.clamp(
        partialB.$[lid]! + moeUpLayout.$.bias[C.experts * C.n4 + b]!,
        d.vec4f(-SWIGLU_LIMIT),
        d.vec4f(SWIGLU_LIMIT),
      );
      moeUpLayout.$.out[row * C.n4 + unit] = (g / (1 + std.exp(-SWIGLU_ALPHA * g))) * (l + 1);
    }
  }
});

export const moeDownLayout = tgpu.bindGroupLayout({
  act: { storage: d.arrayOf(d.vec4f), access: 'readonly' }, // [T·SLOTS, K]
  route: { storage: d.arrayOf(d.f32), access: 'readonly' }, // [T, 2·SLOTS]
  w: { storage: d.arrayOf(d.u32), access: 'readonly' },
  scales: { storage: d.arrayOf(d.vec4f), access: 'readonly' }, // [E·KG, N]
  bias: { storage: d.arrayOf(d.vec4f), access: 'readonly' }, // [E, N]
  residual: { storage: d.arrayOf(d.vec4f), access: 'readonly' }, // [T, N]
  out: { storage: d.arrayOf(d.vec4f), access: 'mutable' }, // [T, N]
  dims: { uniform: Dims },
});

const moeDownKernel = tgpu.computeFn({
  in: { wid: d.builtin.workgroupId, lid: d.builtin.localInvocationIndex },
  workgroupSize: [WG],
})(({ wid, lid }) => {
  'use gpu';
  const C = config.$;
  const lane = lid % K_LANES;
  const unit = wid.x * UNITS + d.u32(lid / K_LANES);
  // The grid is at most 65535 rows high; longer inputs loop.
  for (let t = wid.y; t < moeDownLayout.$.dims.tokens; t += MAX_WORKGROUPS_PER_DIM) {
    const r = t * 2 * MOE_SLOTS;

    let acc = d.vec4f();
    if (unit < C.n4) {
      for (const s of tgpu.unroll(std.range(MOE_SLOTS))) {
        const expert = d.u32(moeDownLayout.$.route[r + s]!);
        const gate = moeDownLayout.$.route[r + MOE_SLOTS + s]!;
        const wBase = (expert * C.n4 + unit) * C.k4count * C.tileWords;
        const sRow = expert * C.kg;
        const aRow = (t * MOE_SLOTS + s) * C.k4count;
        let dot = d.vec4f();
        for (let k4 = lane; k4 < C.k4count; k4 += K_LANES) {
          const tb = wBase + k4 * C.tileWords;
          const sc = (sRow + d.u32(k4 / C.g4)) * C.n4 + unit;
          const v = tileDots(
            moeDownLayout.$.w[tb]!,
            moeDownLayout.$.w[tb + colWord(1)]!,
            moeDownLayout.$.w[tb + colWord(2)]!,
            moeDownLayout.$.w[tb + colWord(3)]!,
            moeDownLayout.$.act[aRow + k4]!,
          );
          dot += v * moeDownLayout.$.scales[sc]!;
        }
        acc += dot * gate;
      }
    }
    partialA.$[lid] = d.vec4f(acc);
    mergeLanes(lid, lane, false);

    if (lane === 0 && unit < C.n4) {
      let sum = partialA.$[lid]! + moeDownLayout.$.residual[t * C.n4 + unit]!;
      for (const s of tgpu.unroll(std.range(MOE_SLOTS))) {
        const expert = d.u32(moeDownLayout.$.route[r + s]!);
        sum +=
          moeDownLayout.$.bias[expert * C.n4 + unit]! * moeDownLayout.$.route[r + MOE_SLOTS + s]!;
      }
      moeDownLayout.$.out[t * C.n4 + unit] = d.vec4f(sum);
    }
  }
});

/** Load-time shape of a routed expert matmul: [K] in, [N] out per row. */
export interface MoeQuantCfg {
  k: number;
  n: number;
  bits: number;
  groupSize: number;
  experts: number;
}

const unitGroups = (n: number) => Math.ceil(n / 4 / UNITS);

const withConfig = (root: TgpuRoot, cfg: MoeQuantCfg) =>
  root.with(config, {
    k: cfg.k,
    n: cfg.n,
    n4: cfg.n / 4,
    k4count: cfg.k / 4,
    tileWords: cfg.bits / 2,
    g4: cfg.groupSize / 4,
    kg: cfg.k / cfg.groupSize,
    bits: cfg.bits,
    experts: cfg.experts,
  });

export function createMoeUpPipeline(root: TgpuRoot, cfg: MoeQuantCfg) {
  return withConfig(root, cfg).createComputePipeline({ compute: moeUpKernel });
}

export function createMoeDownPipeline(root: TgpuRoot, cfg: MoeQuantCfg) {
  return withConfig(root, cfg).createComputePipeline({ compute: moeDownKernel });
}

export function moeUpHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createMoeUpPipeline>,
  args: { tokens: number; n: number },
  buffers: {
    x: F32Buffer;
    route: F32Buffer;
    gluW: U32Buffer;
    gluScales: F32Buffer;
    linW: U32Buffer;
    linScales: F32Buffer;
    bias: F32Buffer;
    out: F32Buffer;
  },
): KernelHandle {
  const dims = cachedUniform(root, Dims, { tokens: args.tokens });
  const bindGroup = cachedBindGroup(root, moeUpLayout, { ...buffers, dims });
  return makeHandle(pipeline, 'moeUp', bindGroup, [
    unitGroups(args.n),
    Math.min(args.tokens * MOE_SLOTS, MAX_WORKGROUPS_PER_DIM),
  ]);
}

export function moeDownHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createMoeDownPipeline>,
  args: { tokens: number; n: number },
  buffers: {
    act: F32Buffer;
    route: F32Buffer;
    w: U32Buffer;
    scales: F32Buffer;
    bias: F32Buffer;
    residual: F32Buffer;
    out: F32Buffer;
  },
): KernelHandle {
  const dims = cachedUniform(root, Dims, { tokens: args.tokens });
  const bindGroup = cachedBindGroup(root, moeDownLayout, { ...buffers, dims });
  return makeHandle(pipeline, 'moeDown', bindGroup, [
    unitGroups(args.n),
    Math.min(args.tokens, MAX_WORKGROUPS_PER_DIM),
  ]);
}
