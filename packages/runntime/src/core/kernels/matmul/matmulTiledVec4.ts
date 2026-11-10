/** Matmul with 32×32 tiles: out = act(a·b + bias) + addend, for K and N
 *  multiples of 4, in f16 or f32. A workgroup loads blocks of a and b, 8 K
 *  at a time, into workgroup memory and every thread reuses them. Sums in
 *  f32.
 *
 *  Rows of a can be further apart than K, which lets conv1d run on this kernel.
 *  A pipeline can also plug in its own loader for a, which reads each vec4 of
 *  a's logical [M, K] from wherever it lives.
 *
 *  Based on webgpu-kernels/ai.onnx.MatMul (matmul-notrans-vec4-tiled-reg). */

import tgpu, { d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import { type FloatBuffer, type KernelHandle, makeHandle } from '../common.ts';
import { cachedBindGroup, cachedUniform } from '../../gpu/dispatchCache.ts';
import { activationFor, actSlot } from '../activations.ts';
import { type Elem, F32_ELEM } from '../elem.ts';

const Dims = d.struct({ m: d.u32, aRows: d.u32 });

function makeLayout(elem: Elem) {
  return tgpu.bindGroupLayout({
    a: { storage: d.arrayOf(elem.vec4), access: 'readonly' },
    b: { storage: d.arrayOf(elem.vec4), access: 'readonly' },
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

// 32×32 output tile per workgroup, 8×8 threads, 8 K per block.
const BM = 32;
const BN = 32;
const WGC = 8;
const WGR = 8;
const BK = 8;

/** Reads vec4 k4 of row `row` of a. The kernel calls it only inside [M, K]. */
export type LoadA = (row: number, k4: number) => d.v4f;

const loadASlot = tgpu.slot<LoadA>();

/** Below this K there is too little to reuse. */
const MIN_K = 64;

/** Whether a [M,K]·[K,N] matmul can use this kernel. */
export function matmulTiledVec4Eligible(k: number, n: number): boolean {
  return k >= MIN_K && k % 4 === 0 && n % 4 === 0;
}

/** K, N and the row stride of a are built into the shader; M comes from the
 *  uniform. */
export interface MatmulTiledVec4Cfg {
  k: number;
  n: number;
  hasBias: number;
  hasAdd: number;
  act?: number;
  /** Distance between rows of a, in values. K when omitted. */
  aStride?: number;
}

/** What a loader reads: a, and how many rows a's buffer holds. Every kernel
 *  that takes a loader binds both. */
export interface ASource {
  readonly $: { readonly a: d.v4f[]; readonly dims: { readonly aRows: number } };
}

/** The layout every pipeline of this kernel binds, so a loader can read a and
 *  aRows. */
export function matmulTiledVec4Layout(elem: Elem): ReturnType<typeof makeLayout> {
  return layoutFor(elem);
}

/** Reads row `row` of a at aStride. */
export function stridedLoadA(
  elem: Elem,
  aStride: number,
  source: ASource = layoutFor(elem),
): LoadA {
  const storeVec4 = elem.vec4;
  const aStride4 = aStride / 4;
  return (row: number, k4: number) => {
    'use gpu';
    return storeVec4(source.$.a[row * aStride4 + k4]!);
  };
}

function makeMatmulTiledVec4Kernel(cfg: MatmulTiledVec4Cfg, elem: Elem) {
  const { k, n, hasBias, hasAdd } = cfg;
  const layout = layoutFor(elem);
  const storeVec4 = elem.vec4;
  const lanes = WGC * WGR;
  const tm = BM / WGR;
  const bk4 = BK / 4;
  const bn4 = BN / 4;
  const k4 = k / 4;
  const n4 = n / 4;
  const tiles = Math.ceil(k / BK);
  if (BM * bk4 !== lanes || BK * bn4 !== lanes) {
    throw new Error('matmulTiledVec4: each tile needs one vec4 per thread');
  }
  const tileA = tgpu.workgroupVar(d.arrayOf(elem.vec4, BM * bk4));
  const tileB = tgpu.workgroupVar(d.arrayOf(elem.vec4, BK * bn4));
  return tgpu.computeFn({
    in: { lid: d.builtin.localInvocationIndex, wid: d.builtin.workgroupId },
    workgroupSize: [lanes],
  })(({ lid, wid }) => {
    'use gpu';
    const m = layout.$.dims.m;
    const tx = lid % WGC;
    const ty = std.intdiv(lid, WGC);
    const mBase = wid.y * BM;
    const nBase4 = wid.x * bn4;

    const accs = d.arrayOf(d.vec4f, tm)();
    // Thread lid fills slot lid of each tile.
    const am = mBase + std.intdiv(lid, bk4);
    const aCol = lid % bk4;
    const bRow = std.intdiv(lid, bn4);
    const bc4 = nBase4 + (lid % bn4);
    let ra = storeVec4();
    let rb = storeVec4();
    if (am < m && aCol < k4) {
      ra = storeVec4(loadASlot.$(am, aCol));
    }
    if (bRow < k && bc4 < n4) {
      rb = storeVec4(layout.$.b[bRow * n4 + bc4]!);
    }
    for (const t of std.range(tiles)) {
      tileA.$[lid] = storeVec4(ra);
      tileB.$[lid] = storeVec4(rb);
      std.workgroupBarrier();
      // The next tile's loads are in flight while this one computes.
      const tn = t + 1;
      if (tn < tiles) {
        const ak4 = tn * bk4 + aCol;
        ra = storeVec4();
        if (am < m && ak4 < k4) {
          ra = storeVec4(loadASlot.$(am, ak4));
        }
        const bk = tn * BK + bRow;
        rb = storeVec4();
        if (bk < k && bc4 < n4) {
          rb = storeVec4(layout.$.b[bk * n4 + bc4]!);
        }
      }
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
          layout.$.out[row * n4 + c4] = storeVec4(v);
        }
      }
    }
  });
}

/** One pipeline per dtype, K, N, loader, epilogue and activation. `loadA`
 *  replaces the strided read of a. */
export function createMatmulTiledVec4Pipeline(
  root: TgpuRoot,
  cfg: MatmulTiledVec4Cfg,
  elem: Elem = F32_ELEM,
  loadA: LoadA = stridedLoadA(elem, cfg.aStride ?? cfg.k),
) {
  return root
    .with(actSlot, activationFor(cfg.act))
    .with(loadASlot, loadA)
    .createComputePipeline({ compute: makeMatmulTiledVec4Kernel(cfg, elem) });
}

/** One workgroup per 32×32 output tile. aRows is how many rows a's buffer
 *  holds, for loaders that bound-check it. */
export function matmulTiledVec4Handle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createMatmulTiledVec4Pipeline>,
  m: number,
  n: number,
  buffers: {
    a: FloatBuffer;
    b: FloatBuffer;
    out: FloatBuffer;
    bias?: FloatBuffer;
    addend?: FloatBuffer;
  },
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
  return makeHandle(pipeline, 'matmulTiledVec4', bindGroup, [Math.ceil(n / BN), Math.ceil(m / BM)]);
}
