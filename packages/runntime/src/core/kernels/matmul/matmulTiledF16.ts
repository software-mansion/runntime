/** f16 matmul with 32×32 tiles: out = act(a·b + bias) + addend, for K and N
 *  multiples of 4. A workgroup loads 32×32 blocks of a and b into workgroup
 *  memory and every thread reuses them. Sums in f32.
 *
 *  Based on webgpu-kernels/ai.onnx.MatMul (matmul-notrans-vec4-tiled-reg). */

import tgpu, { d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import { type FloatBuffer, type KernelHandle, makeHandle } from '../common.ts';
import { cachedBindGroup, cachedUniform } from '../../gpu/dispatchCache.ts';
import { activationFor, actSlot } from '../activations.ts';

const Dims = d.struct({ m: d.u32 });

const layout = tgpu.bindGroupLayout({
  a: { storage: d.arrayOf(d.vec4h), access: 'readonly' },
  b: { storage: d.arrayOf(d.vec4h), access: 'readonly' },
  bias: { storage: d.arrayOf(d.vec4h), access: 'readonly' }, // [N/4]; dead when hasBias=0
  addend: { storage: d.arrayOf(d.vec4h), access: 'readonly' }, // [M,N/4]; dead when hasAdd=0
  out: { storage: d.arrayOf(d.vec4h), access: 'mutable' },
  dims: { uniform: Dims },
});

// 32×32 output tile per workgroup, 8×8 threads, 32 K per block.
const BM = 32;
const BN = 32;
const WGC = 8;
const WGR = 8;
const BK = 32;

/** At K = 32 there is one block and nothing to reuse. */
const MIN_K = 64;

/** Whether an f16 [M,K]·[K,N] matmul can use this kernel. */
export function matmulTiledF16Eligible(k: number, n: number): boolean {
  return k >= MIN_K && k % 4 === 0 && n % 4 === 0;
}

/** K and N are built into the shader; M comes from the uniform. */
export interface MatmulTiledF16Cfg {
  k: number;
  n: number;
  hasBias: number;
  hasAdd: number;
  act?: number;
}

function makeMatmulTiledF16Kernel(cfg: MatmulTiledF16Cfg) {
  const { k, n, hasBias, hasAdd } = cfg;
  const lanes = WGC * WGR;
  const tm = BM / WGR;
  const bk4 = BK / 4;
  const bn4 = BN / 4;
  const k4 = k / 4;
  const n4 = n / 4;
  const aVecs = (BM * bk4) / lanes;
  const bVecs = (BK * bn4) / lanes;
  const tiles = Math.ceil(k / BK);
  const tileA = tgpu.workgroupVar(d.arrayOf(d.vec4h, BM * bk4));
  const tileB = tgpu.workgroupVar(d.arrayOf(d.vec4h, BK * bn4));
  return tgpu.computeFn({
    in: { lid: d.builtin.localInvocationIndex, wid: d.builtin.workgroupId },
    workgroupSize: [lanes],
  })(({ lid, wid }) => {
    'use gpu';
    const m = layout.$.dims.m;
    const tx = lid % WGC;
    const ty = d.u32(lid / WGC);
    const mBase = wid.y * BM;
    const nBase4 = wid.x * bn4;

    const accs = d.arrayOf(d.vec4f, tm)();
    for (const t of std.range(tiles)) {
      // Neighbouring threads load neighbouring vec4s.
      for (const i of tgpu.unroll(std.range(aVecs))) {
        const e = lid + i * lanes;
        const am = mBase + d.u32(e / bk4);
        const ak4 = t * bk4 + (e % bk4);
        let v = d.vec4h();
        if (am < m && ak4 < k4) {
          v = d.vec4h(layout.$.a[am * k4 + ak4]!);
        }
        tileA.$[e] = d.vec4h(v);
      }
      for (const i of tgpu.unroll(std.range(bVecs))) {
        const e = lid + i * lanes;
        const bk = t * BK + d.u32(e / bn4);
        const bc4 = nBase4 + (e % bn4);
        let v = d.vec4h();
        if (bk < k && bc4 < n4) {
          v = d.vec4h(layout.$.b[bk * n4 + bc4]!);
        }
        tileB.$[e] = d.vec4h(v);
      }
      std.workgroupBarrier();

      for (const kv of tgpu.unroll(std.range(bk4))) {
        const b0 = d.vec4f(tileB.$[kv * 4 * bn4 + tx]!);
        const b1 = d.vec4f(tileB.$[(kv * 4 + 1) * bn4 + tx]!);
        const b2 = d.vec4f(tileB.$[(kv * 4 + 2) * bn4 + tx]!);
        const b3 = d.vec4f(tileB.$[(kv * 4 + 3) * bn4 + tx]!);
        for (const i of tgpu.unroll(std.range(tm))) {
          const av = d.vec4f(tileA.$[(ty * tm + i) * bk4 + kv]!);
          accs[i] = accs[i]! + av.x * b0 + av.y * b1 + av.z * b2 + av.w * b3;
        }
      }
      std.workgroupBarrier();
    }

    const c4 = nBase4 + tx;
    if (c4 < n4) {
      let bias = d.vec4f();
      if (hasBias > 0) {
        bias = d.vec4f(layout.$.bias[c4]!);
      }
      for (const i of tgpu.unroll(std.range(tm))) {
        const row = mBase + ty * tm + i;
        if (row < m) {
          let v = actSlot.$(accs[i]! + bias);
          if (hasAdd > 0) {
            v += d.vec4f(layout.$.addend[row * n4 + c4]!);
          }
          layout.$.out[row * n4 + c4] = d.vec4h(v);
        }
      }
    }
  });
}

/** One pipeline per K, N, epilogue and activation. */
export function createMatmulTiledF16Pipeline(root: TgpuRoot, cfg: MatmulTiledF16Cfg) {
  return root
    .with(actSlot, activationFor(cfg.act))
    .createComputePipeline({ compute: makeMatmulTiledF16Kernel(cfg) });
}

/** One workgroup per 32×32 output tile. */
export function matmulTiledF16Handle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createMatmulTiledF16Pipeline>,
  m: number,
  n: number,
  buffers: {
    a: FloatBuffer;
    b: FloatBuffer;
    out: FloatBuffer;
    bias?: FloatBuffer;
    addend?: FloatBuffer;
  },
): KernelHandle {
  const dims = cachedUniform(root, Dims, { m });
  // Absent epilogue operands alias `a`; those pipelines never read them.
  const bindGroup = cachedBindGroup(root, layout, {
    ...buffers,
    bias: buffers.bias ?? buffers.a,
    addend: buffers.addend ?? buffers.a,
    dims,
  });
  return makeHandle(pipeline, 'matmulTiledF16', bindGroup, [Math.ceil(n / BN), Math.ceil(m / BM)]);
}
