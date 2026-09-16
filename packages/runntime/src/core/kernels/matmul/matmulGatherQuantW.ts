import tgpu, { d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import {
  type F32Buffer,
  type KernelHandle,
  flatIndex,
  makeHandle,
  type U32Buffer,
  WORKGROUP_SIZE,
} from '../common.ts';
import { cachedBindGroup, cachedUniform } from '../../gpu/dispatchCache.ts';
import { deqVec4 } from '../quantCommon.ts';

/** Indexed quantized matmul plus bias:
 *  out[i] = a[i] · dequant(W[expert_i]) + bias[expert_i], where expert_i comes
 *  from expertIdx[i], read per row from a GPU buffer as f32 since it arrives
 *  straight from the topk packed output. Each token therefore runs against its
 *  own expert's weights without the routing decision leaving the device.
 *
 *  W is the fused [E·K, N] quant weight in the blocked-unit layout, scales are
 *  [E·KG, N] f32 and bias is [E, N] f32.
 *
 *  Every load-time constant — K, N, bits, groupSize and the counts derived from
 *  them — is baked into the pipeline through a tgpu.accessor, so the emitted
 *  WGSL carries literal loop bounds and offsets that the driver unrolls and
 *  folds. Only `m`, the runtime token count, is a uniform, so there is one
 *  pipeline per (op, dtype, config) and a new token count never compiles a
 *  shader. */

const Dims = d.struct({ m: d.u32 });

const Config = d.struct({
  k: d.u32,
  n: d.u32,
  n4: d.u32, // n/4 — col units per row
  k4count: d.u32, // k/4 — 4-row chunks per column
  tileWords: d.u32, // bits/2 — words per 16-value tile
  g4: d.u32, // groupSize/4 — chunks per scale group
  kg: d.u32, // k/groupSize — scale rows per expert
  bits: d.u32,
});
// Degenerate defaults, so an unconfigured pipeline fails a correctness check
// rather than silently running a plausible config.
const config = tgpu.accessor(Config, {
  k: 0,
  n: 0,
  n4: 0,
  k4count: 0,
  tileWords: 0,
  g4: 1,
  kg: 0,
  bits: 0,
});

export const matmulGatherQuantWLayout = tgpu.bindGroupLayout({
  a: { storage: d.arrayOf(d.f32), access: 'readonly' },
  w: { storage: d.arrayOf(d.u32), access: 'readonly' },
  scales: { storage: d.arrayOf(d.f32), access: 'readonly' },
  bias: { storage: d.arrayOf(d.f32), access: 'readonly' }, // [E, N]
  expertIdx: { storage: d.arrayOf(d.f32), access: 'readonly' }, // [m], f32 from topk
  out: { storage: d.arrayOf(d.f32), access: 'mutable' },
  dims: { uniform: Dims },
});

export const matmulGatherQuantWKernel = tgpu.computeFn({
  in: { gid: d.builtin.globalInvocationId },
  workgroupSize: [WORKGROUP_SIZE],
})(({ gid }) => {
  'use gpu';
  const C = config.$;
  const idx = flatIndex(gid);
  if (idx >= matmulGatherQuantWLayout.$.dims.m * C.n4) {
    return;
  }
  const row = d.u32(idx / C.n4);
  const unit = idx % C.n4; // this thread's 4-col unit
  const col0 = unit * 4;

  // Per-row expert (f32 index from topk → u32) → weight / scale / bias bases.
  const expert = d.u32(matmulGatherQuantWLayout.$.expertIdx[row]!);
  const wBase = (expert * C.n4 + unit) * C.k4count * C.tileWords;
  const sRow = expert * C.kg; // row into scales [E·KG, N]
  const b4 = C.bits * 4; // tile bits per column

  let sv = d.vec4f(
    matmulGatherQuantWLayout.$.scales[sRow * C.n + col0]!,
    matmulGatherQuantWLayout.$.scales[sRow * C.n + col0 + 1]!,
    matmulGatherQuantWLayout.$.scales[sRow * C.n + col0 + 2]!,
    matmulGatherQuantWLayout.$.scales[sRow * C.n + col0 + 3]!,
  );
  let acc = d.vec4f(); // scaled total
  let raw = d.vec4f(); // current group's unscaled dots
  let curG = d.u32(0);
  for (let k4 = d.u32(0); k4 < C.k4count; k4++) {
    const g = d.u32(k4 / C.g4);
    if (g !== curG) {
      acc += raw * sv;
      raw = d.vec4f();
      curG = g;
      const s = (sRow + g) * C.n + col0;
      sv = d.vec4f(
        matmulGatherQuantWLayout.$.scales[s]!,
        matmulGatherQuantWLayout.$.scales[s + 1]!,
        matmulGatherQuantWLayout.$.scales[s + 2]!,
        matmulGatherQuantWLayout.$.scales[s + 3]!,
      );
    }
    const xb = row * C.k + k4 * 4;
    const x = d.vec4f(
      matmulGatherQuantWLayout.$.a[xb]!,
      matmulGatherQuantWLayout.$.a[xb + 1]!,
      matmulGatherQuantWLayout.$.a[xb + 2]!,
      matmulGatherQuantWLayout.$.a[xb + 3]!,
    );
    const tb = wBase + k4 * C.tileWords;
    raw += d.vec4f(
      std.dot(x, deqVec4(matmulGatherQuantWLayout.$.w[tb]!, 0, C.bits)),
      std.dot(x, deqVec4(matmulGatherQuantWLayout.$.w[tb + d.u32(b4 / 32)]!, b4 % 32, C.bits)),
      std.dot(
        x,
        deqVec4(matmulGatherQuantWLayout.$.w[tb + d.u32((2 * b4) / 32)]!, (2 * b4) % 32, C.bits),
      ),
      std.dot(
        x,
        deqVec4(matmulGatherQuantWLayout.$.w[tb + d.u32((3 * b4) / 32)]!, (3 * b4) % 32, C.bits),
      ),
    );
  }
  acc += raw * sv;
  // Fold in the gathered per-expert bias.
  const bBase = expert * C.n + col0;
  acc += d.vec4f(
    matmulGatherQuantWLayout.$.bias[bBase]!,
    matmulGatherQuantWLayout.$.bias[bBase + 1]!,
    matmulGatherQuantWLayout.$.bias[bBase + 2]!,
    matmulGatherQuantWLayout.$.bias[bBase + 3]!,
  );
  const oBase = row * C.n + col0;
  matmulGatherQuantWLayout.$.out[oBase] = acc.x;
  matmulGatherQuantWLayout.$.out[oBase + 1] = acc.y;
  matmulGatherQuantWLayout.$.out[oBase + 2] = acc.z;
  matmulGatherQuantWLayout.$.out[oBase + 3] = acc.w;
});

export interface GatherQuantConfig {
  k: number;
  n: number;
  bits: number;
  groupSize: number;
}

export function createMatmulGatherQuantWPipeline(root: TgpuRoot, cfg: GatherQuantConfig) {
  return root
    .with(config, {
      k: cfg.k,
      n: cfg.n,
      n4: cfg.n / 4,
      k4count: cfg.k / 4,
      tileWords: cfg.bits / 2,
      g4: cfg.groupSize / 4,
      kg: cfg.k / cfg.groupSize,
      bits: cfg.bits,
    })
    .createComputePipeline({ compute: matmulGatherQuantWKernel });
}

export function matmulGatherQuantWHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createMatmulGatherQuantWPipeline>,
  args: { m: number; n: number },
  buffers: {
    a: F32Buffer;
    w: U32Buffer;
    scales: F32Buffer;
    bias: F32Buffer;
    expertIdx: F32Buffer;
    out: F32Buffer;
  },
): KernelHandle {
  const dims = cachedUniform(root, Dims, { m: args.m });
  const bindGroup = cachedBindGroup(root, matmulGatherQuantWLayout, { ...buffers, dims });
  return makeHandle(
    pipeline,
    'matmulGatherQuantW',
    bindGroup,
    Math.ceil((args.m * (args.n / 4)) / WORKGROUP_SIZE),
  );
}
