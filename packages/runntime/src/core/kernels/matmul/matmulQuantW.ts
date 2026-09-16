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

/** a[M,K] · dequant(view(w)[baseRow..+K, N]) giving out[M,N], where w is a
 *  quantW weight in the blocked-unit layout.
 *
 *  Tiles are expert-granular, so baseRow must be a K-aligned expert boundary,
 *  which the op enforces. scales are indexed [scaleBase + kGroup*N + col].
 *
 *  One thread per 4-column unit streams its K-contiguous tile run and dots each
 *  x vec4 against 4 words. Same inner loop as matmulGatherQuantW, without the
 *  per-row expert lookup and the bias. */
const Dims = d.struct({
  m: d.u32,
  baseRow: d.u32, // per-expert view offset — varies per CALL, so stays a uniform
  scaleBase: d.u32,
});

const Config = d.struct({
  k: d.u32,
  n: d.u32,
  n4: d.u32,
  k4count: d.u32,
  tileWords: d.u32,
  g4: d.u32,
  bits: d.u32,
});
const config = tgpu.accessor(Config, {
  k: 0,
  n: 0,
  n4: 0,
  k4count: 0,
  tileWords: 0,
  g4: 1,
  bits: 0,
});

export const matmulQuantWLayout = tgpu.bindGroupLayout({
  a: { storage: d.arrayOf(d.f32), access: 'readonly' },
  w: { storage: d.arrayOf(d.u32), access: 'readonly' },
  scales: { storage: d.arrayOf(d.f32), access: 'readonly' },
  out: { storage: d.arrayOf(d.f32), access: 'mutable' },
  dims: { uniform: Dims },
});

export const matmulQuantWKernel = tgpu.computeFn({
  in: { gid: d.builtin.globalInvocationId },
  workgroupSize: [WORKGROUP_SIZE],
})(({ gid }) => {
  'use gpu';
  const D = matmulQuantWLayout.$.dims;
  const C = config.$;
  const idx = flatIndex(gid);
  if (idx >= D.m * C.n4) {
    return;
  }
  const row = d.u32(idx / C.n4);
  const unit = idx % C.n4; // this thread's 4-col unit
  const col0 = unit * 4;

  const expert = d.u32(D.baseRow / C.k); // blocked tiles are expert-granular
  const wBase = (expert * C.n4 + unit) * C.k4count * C.tileWords;
  const b4 = C.bits * 4; // tile bits per column

  let sv = d.vec4f(
    matmulQuantWLayout.$.scales[D.scaleBase + col0]!,
    matmulQuantWLayout.$.scales[D.scaleBase + col0 + 1]!,
    matmulQuantWLayout.$.scales[D.scaleBase + col0 + 2]!,
    matmulQuantWLayout.$.scales[D.scaleBase + col0 + 3]!,
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
      const s = D.scaleBase + g * C.n + col0;
      sv = d.vec4f(
        matmulQuantWLayout.$.scales[s]!,
        matmulQuantWLayout.$.scales[s + 1]!,
        matmulQuantWLayout.$.scales[s + 2]!,
        matmulQuantWLayout.$.scales[s + 3]!,
      );
    }
    const xb = row * C.k + k4 * 4;
    const x = d.vec4f(
      matmulQuantWLayout.$.a[xb]!,
      matmulQuantWLayout.$.a[xb + 1]!,
      matmulQuantWLayout.$.a[xb + 2]!,
      matmulQuantWLayout.$.a[xb + 3]!,
    );
    const tb = wBase + k4 * C.tileWords;
    raw += d.vec4f(
      std.dot(x, deqVec4(matmulQuantWLayout.$.w[tb]!, 0, C.bits)),
      std.dot(x, deqVec4(matmulQuantWLayout.$.w[tb + d.u32(b4 / 32)]!, b4 % 32, C.bits)),
      std.dot(
        x,
        deqVec4(matmulQuantWLayout.$.w[tb + d.u32((2 * b4) / 32)]!, (2 * b4) % 32, C.bits),
      ),
      std.dot(
        x,
        deqVec4(matmulQuantWLayout.$.w[tb + d.u32((3 * b4) / 32)]!, (3 * b4) % 32, C.bits),
      ),
    );
  }
  acc += raw * sv;
  const oBase = row * C.n + col0;
  matmulQuantWLayout.$.out[oBase] = acc.x;
  matmulQuantWLayout.$.out[oBase + 1] = acc.y;
  matmulQuantWLayout.$.out[oBase + 2] = acc.z;
  matmulQuantWLayout.$.out[oBase + 3] = acc.w;
});

export function createMatmulQuantWPipeline(
  root: TgpuRoot,
  cfg: { k: number; n: number; bits: number; groupSize: number },
) {
  return root
    .with(config, {
      k: cfg.k,
      n: cfg.n,
      n4: cfg.n / 4,
      k4count: cfg.k / 4,
      tileWords: cfg.bits / 2,
      g4: cfg.groupSize / 4,
      bits: cfg.bits,
    })
    .createComputePipeline({ compute: matmulQuantWKernel });
}

export function matmulQuantWHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createMatmulQuantWPipeline>,
  args: { m: number; n: number; baseRow: number; scaleBase: number },
  buffers: { a: F32Buffer; w: U32Buffer; scales: F32Buffer; out: F32Buffer },
): KernelHandle {
  const dims = cachedUniform(root, Dims, {
    m: args.m,
    baseRow: args.baseRow,
    scaleBase: args.scaleBase,
  });
  const bindGroup = cachedBindGroup(root, matmulQuantWLayout, { ...buffers, dims });
  return makeHandle(
    pipeline,
    'matmulQuantW',
    bindGroup,
    Math.ceil((args.m * (args.n / 4)) / WORKGROUP_SIZE),
  );
}
