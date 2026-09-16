import tgpu, { d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import {
  type FloatBuffer,
  type KernelHandle,
  flatIndex,
  makeHandle,
  type F16Buffer,
  WORKGROUP_SIZE,
} from '../common.ts';
import { cachedBindGroup, cachedUniform } from '../../gpu/dispatchCache.ts';
import type { Elem } from '../elem.ts';

/** Runtime-variable dims: input spatial size only. */
const Dims = d.struct({ h: d.u32, w: d.u32 });

export interface ConvTranspose2dCfg {
  cIn: number;
  cOut: number;
  k: number;
  hasBias: number; // 0 | 1
}

const makeLayout = (elem: Elem) =>
  tgpu.bindGroupLayout({
    x: { storage: d.arrayOf(elem.scalar), access: 'readonly' },
    w: { storage: d.arrayOf(d.u32), access: 'readonly' },
    bias: { storage: d.arrayOf(elem.scalar), access: 'readonly' },
    out: { storage: d.arrayOf(elem.scalar), access: 'mutable' },
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

function makeConvTranspose2dKernel(cfg: ConvTranspose2dCfg, elem: Elem) {
  const { cIn, cOut, k, hasBias } = cfg;
  const paired = cIn % 2 === 0;
  const layout = layoutFor(elem);
  const storeScalar = elem.scalar;
  return tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const dims = layout.$.dims;
    const hOut = dims.h * k;
    const wOut = dims.w * k;
    const idx = flatIndex(gid);
    if (idx >= cOut * hOut * wOut) {
      return;
    }
    const wo = idx % wOut;
    const ho = d.u32(idx / wOut) % hOut;
    const co = d.u32(idx / (wOut * hOut));
    const hi = d.u32(ho / k); // explicit integer division (codebase convention)
    const wi = d.u32(wo / k);
    const di = ho % k;
    const dj = wo % k;
    const rowBase = (co * k * k + di * k + dj) * cIn;
    const xPix = hi * dims.w + wi;
    const plane = dims.h * dims.w;
    let acc = d.f32(0);
    if (paired) {
      const wordBase = d.u32(rowBase / 2); // rowBase is even when cIn is
      for (const c2 of std.range(cIn / 2)) {
        const pair = std.unpack2x16float(layout.$.w[wordBase + c2]!);
        const xb = 2 * c2 * plane + xPix;
        acc += pair.x * d.f32(layout.$.x[xb]!) + pair.y * d.f32(layout.$.x[xb + plane]!);
      }
    } else {
      for (const ci of std.range(cIn)) {
        const flat = rowBase + ci;
        const pair = std.unpack2x16float(layout.$.w[d.u32(flat / 2)]!);
        const wv = std.select(pair.x, pair.y, flat % 2 === 1);
        acc += wv * d.f32(layout.$.x[ci * plane + xPix]!);
      }
    }
    if (hasBias === 1) {
      acc += d.f32(layout.$.bias[co]!);
    }
    layout.$.out[idx] = storeScalar(acc);
  });
}

export function createConvTranspose2dPipeline(root: TgpuRoot, cfg: ConvTranspose2dCfg, elem: Elem) {
  return root.createComputePipeline({ compute: makeConvTranspose2dKernel(cfg, elem) });
}

export function convTranspose2dHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createConvTranspose2dPipeline>,
  cfg: ConvTranspose2dCfg,
  args: { h: number; w: number },
  buffers: { x: FloatBuffer; w: F16Buffer; bias: FloatBuffer; out: FloatBuffer },
  elem: Elem,
): KernelHandle {
  const { h, w } = args;
  const dims = cachedUniform(root, Dims, { h, w });
  const bindGroup = cachedBindGroup(root, layoutFor(elem), { ...buffers, dims });
  const total = cfg.cOut * h * cfg.k * w * cfg.k;
  return makeHandle(pipeline, 'convTranspose2d', bindGroup, Math.ceil(total / WORKGROUP_SIZE));
}
