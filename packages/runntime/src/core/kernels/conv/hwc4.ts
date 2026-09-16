/** The HWC4 kernel family: conv, pool, upsample, channel-block copy and the
 *  CHW ↔ HWC4 boundary conversions.
 *
 *  Layout contract, shared with nn.Conv2d's weight repack and the hwc4 ops.
 *  Activations are `array<vec4<f16>>` at element (y·W + x)·C4 + cB, pixel-major
 *  with channels zero-padded to a multiple of 4. Conv weights are mat4 tiles of
 *  4 vec4<f16>, tile (oB, iB, tap) at ((oB·CI4 + iB)·kH·kW + tap), entry iL
 *  holding the output lanes for input lane iL; depthwise weights are one vec4
 *  per (cB, tap). Padded weight lanes are zero, so padded activation lanes stay
 *  0 through the epilogue.
 *
 *  Kernels are factories over plain JS numbers, so geometry reaches WGSL as
 *  literals and branches prune. Input h and w stay a runtime uniform. */

import tgpu, { d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import {
  type F16Buffer,
  type F32Buffer,
  flatIndex,
  type KernelHandle,
  makeFlatIndex,
  makeHandle,
  WORKGROUP_SIZE,
} from '../common.ts';
import { ACT_CODE, gelu } from '../activations.ts';
import { cachedBindGroup, cachedUniform } from '../../gpu/dispatchCache.ts';

export const WG_1X1 = 256;
const flatIndex256 = makeFlatIndex(WG_1X1);

const ConvDims = d.struct({ h: d.u32, w: d.u32, outC4: d.u32, outOffB: d.u32 });

const convLayout = tgpu.bindGroupLayout({
  x: { storage: d.arrayOf(d.vec4h), access: 'readonly' },
  w: { storage: d.arrayOf(d.vec4h), access: 'readonly' },
  bias: { storage: d.arrayOf(d.f16), access: 'readonly' },
  res: { storage: d.arrayOf(d.vec4h), access: 'readonly' }, // residual addend (dummy when hasAdd=0)
  out: { storage: d.arrayOf(d.vec4h), access: 'mutable' },
  dims: { uniform: ConvDims },
});

export interface Conv2dHwc4Cfg {
  cIn: number;
  cOut: number;
  kH: number;
  kW: number;
  stride: number;
  padding: number;
  hasBias: number;
  hasAct: number;
  hasAdd: number;
  kind: number;
  pxT?: number;
  ks?: number;
}

const makeEpilogue = (cfg: Conv2dHwc4Cfg) => {
  const { cOut, hasBias, hasAct, hasAdd } = cfg;
  const co4 = Math.ceil(cOut / 4);
  const aligned = cOut % 4 === 0;
  return (acc: d.v4h, oB: number, pxOut: number): d.v4h => {
    'use gpu';
    let r = acc * d.f16(1);
    if (hasBias === 1) {
      const c0 = oB * 4;
      if (aligned) {
        r += d.vec4h(
          convLayout.$.bias[c0]!,
          convLayout.$.bias[c0 + 1]!,
          convLayout.$.bias[c0 + 2]!,
          convLayout.$.bias[c0 + 3]!,
        );
      }
      if (!aligned) {
        const zero = d.f16(0);
        r += d.vec4h(
          std.select(zero, convLayout.$.bias[std.min(c0, cOut - 1)]!, c0 < cOut),
          std.select(zero, convLayout.$.bias[std.min(c0 + 1, cOut - 1)]!, c0 + 1 < cOut),
          std.select(zero, convLayout.$.bias[std.min(c0 + 2, cOut - 1)]!, c0 + 2 < cOut),
          std.select(zero, convLayout.$.bias[std.min(c0 + 3, cOut - 1)]!, c0 + 3 < cOut),
        );
      }
    }
    // Re-implemented in f16 here; ACTIVATIONS' bodies are all f32.
    if (hasAct === ACT_CODE.silu) {
      r = r / (d.vec4h(1) + std.exp(d.vec4h(0) - r));
    }
    if (hasAct === ACT_CODE.gelu) {
      r = d.vec4h(gelu(d.vec4f(r)));
    }
    if (hasAct === ACT_CODE.relu) {
      r = std.max(r, d.vec4h(0));
    }
    if (hasAdd === 1) {
      // torch's x + cv(x) order: the addend lands after bias and act.
      r += d.vec4h(convLayout.$.res[pxOut * co4 + oB]!);
    }
    return r;
  };
};

export const makeConvGenericKernel = (cfg: Conv2dHwc4Cfg) => {
  const { kH, kW, stride, padding } = cfg;
  const ci4 = Math.ceil(cfg.cIn / 4);
  const co4 = Math.ceil(cfg.cOut / 4);
  const kk = kH * kW;
  const epilogue = makeEpilogue(cfg);
  return tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const dims = convLayout.$.dims;
    const hOut = d.u32((dims.h + 2 * padding - kH) / stride) + 1;
    const wOut = d.u32((dims.w + 2 * padding - kW) / stride) + 1;
    const idx = flatIndex(gid);
    if (idx >= co4 * hOut * wOut) {
      return;
    }
    const ox = idx % wOut;
    const oy = d.u32(idx / wOut) % hOut;
    const oB = d.u32(idx / (wOut * hOut));

    let acc = d.vec4h();
    for (const ky of tgpu.unroll(std.range(kH))) {
      const iyP = oy * stride + ky;
      if (iyP >= padding && iyP < dims.h + padding) {
        const iy = iyP - padding;
        for (const kx of tgpu.unroll(std.range(kW))) {
          const ixP = ox * stride + kx;
          if (ixP >= padding && ixP < dims.w + padding) {
            const ix = ixP - padding;
            const pIn = (iy * dims.w + ix) * ci4;
            // u32-wrapped: a bare JS int would drag the index math to i32.
            const tap = d.u32(ky * kW + kx);
            for (let iB = d.u32(0); iB < ci4; iB++) {
              const xv = convLayout.$.x[pIn + iB]!;
              const tb = ((oB * ci4 + iB) * kk + tap) * 4;
              acc += xv.x * convLayout.$.w[tb]!;
              acc += xv.y * convLayout.$.w[tb + 1]!;
              acc += xv.z * convLayout.$.w[tb + 2]!;
              acc += xv.w * convLayout.$.w[tb + 3]!;
            }
          }
        }
      }
    }
    const pxOut = oy * wOut + ox;
    convLayout.$.out[pxOut * dims.outC4 + dims.outOffB + oB] = epilogue(acc, oB, pxOut);
  });
};

export const makeConv1x1Kernel = (cfg: Conv2dHwc4Cfg) => {
  const ci4 = Math.ceil(cfg.cIn / 4);
  const co4 = Math.ceil(cfg.cOut / 4);
  const pxT = cfg.pxT === 8 ? 8 : 4;
  const epilogue = makeEpilogue(cfg);
  return tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WG_1X1],
  })(({ gid }) => {
    'use gpu';
    const dims = convLayout.$.dims;
    const pT = d.u32((dims.h * dims.w) / pxT);
    const idx = flatIndex256(gid);
    if (idx >= pT * co4) {
      return;
    }
    const p0 = d.u32(idx / co4) * pxT;
    const oB = idx % co4;

    const accs = d.arrayOf(d.vec4h, pxT)();
    for (let iB = d.u32(0); iB < ci4; iB++) {
      const tb = (oB * ci4 + iB) * 4;
      const w0 = convLayout.$.w[tb]!;
      const w1 = convLayout.$.w[tb + 1]!;
      const w2 = convLayout.$.w[tb + 2]!;
      const w3 = convLayout.$.w[tb + 3]!;
      for (const k of tgpu.unroll(std.range(pxT))) {
        const xv = convLayout.$.x[(p0 + k) * ci4 + iB]!;
        accs[k] = accs[k]! + xv.x * w0 + xv.y * w1 + xv.z * w2 + xv.w * w3;
      }
    }
    const ob0 = dims.outOffB + oB;
    for (const k of tgpu.unroll(std.range(pxT))) {
      convLayout.$.out[(p0 + k) * dims.outC4 + ob0] = epilogue(accs[k]!, oB, p0 + k);
    }
  });
};

const SMALL_1X1_PXT = 4;
const smallPartial = tgpu.workgroupVar(d.arrayOf(d.vec4h, WG_1X1 * SMALL_1X1_PXT));

export function small1x1Lanes(cIn: number): number {
  const ci4 = Math.ceil(cIn / 4);
  const want = Math.max(2, Math.min(16, Math.floor(ci4 / 8)));
  return 2 ** Math.floor(Math.log2(want));
}

export const SMALL_1X1_MAX_TILE_THREADS = 2048;
export function conv1x1Kind(hw: number, cOut: number): 1 | 3 {
  const co4 = Math.ceil(cOut / 4);
  if (hw % 4 !== 0) return 3;
  return (hw / 4) * co4 < SMALL_1X1_MAX_TILE_THREADS ? 3 : 1;
}

export const makeConv1x1SmallKernel = (cfg: Conv2dHwc4Cfg) => {
  const ci4 = Math.ceil(cfg.cIn / 4);
  const co4 = Math.ceil(cfg.cOut / 4);
  const SMALL_1X1_KS = cfg.ks ?? small1x1Lanes(cfg.cIn);
  const SMALL_1X1_OL = WG_1X1 / SMALL_1X1_KS;
  const SMALL_1X1_LOG2_KS = Math.log2(SMALL_1X1_KS);
  const oGroups = Math.ceil(co4 / SMALL_1X1_OL);
  const epilogue = makeEpilogue(cfg);
  return tgpu.computeFn({
    in: { wid: d.builtin.workgroupId, lid: d.builtin.localInvocationIndex },
    workgroupSize: [WG_1X1],
  })(({ wid, lid }) => {
    'use gpu';
    const dims = convLayout.$.dims;
    const hw = dims.h * dims.w;
    const kLane = lid % SMALL_1X1_KS;
    const oLane = d.u32(lid / SMALL_1X1_KS);
    const oB = (wid.x % oGroups) * SMALL_1X1_OL + oLane;
    const p0 = d.u32(wid.x / oGroups) * SMALL_1X1_PXT;

    const accs = d.arrayOf(d.vec4h, SMALL_1X1_PXT)();
    if (oB < co4) {
      for (let iB = kLane; iB < ci4; iB += SMALL_1X1_KS) {
        const tb = (oB * ci4 + iB) * 4;
        const w0 = convLayout.$.w[tb]!;
        const w1 = convLayout.$.w[tb + 1]!;
        const w2 = convLayout.$.w[tb + 2]!;
        const w3 = convLayout.$.w[tb + 3]!;
        for (const k of tgpu.unroll(std.range(SMALL_1X1_PXT))) {
          if (p0 + k < hw) {
            const xv = convLayout.$.x[(p0 + k) * ci4 + iB]!;
            accs[k] = accs[k]! + xv.x * w0 + xv.y * w1 + xv.z * w2 + xv.w * w3;
          }
        }
      }
    }
    for (const k of tgpu.unroll(std.range(SMALL_1X1_PXT))) {
      smallPartial.$[lid * SMALL_1X1_PXT + k] = d.vec4h(accs[k]!);
    }
    std.workgroupBarrier();
    // Tree fold over kLane: (kLane, oLane) lives at tid = oLane·KS + kLane,
    // so partner lanes are `s` entries apart. Every thread hits every
    // barrier; only the adds are lane-gated.
    for (const i of tgpu.unroll(std.range(SMALL_1X1_LOG2_KS))) {
      const s = d.u32(SMALL_1X1_KS >> (i + 1));
      if (kLane < s) {
        for (const k of tgpu.unroll(std.range(SMALL_1X1_PXT))) {
          smallPartial.$[lid * SMALL_1X1_PXT + k] =
            smallPartial.$[lid * SMALL_1X1_PXT + k]! +
            smallPartial.$[(lid + s) * SMALL_1X1_PXT + k]!;
        }
      }
      std.workgroupBarrier();
    }
    if (kLane === 0 && oB < co4) {
      const ob0 = dims.outOffB + oB;
      for (const k of tgpu.unroll(std.range(SMALL_1X1_PXT))) {
        if (p0 + k < hw) {
          convLayout.$.out[(p0 + k) * dims.outC4 + ob0] = epilogue(
            smallPartial.$[lid * SMALL_1X1_PXT + k]!,
            oB,
            p0 + k,
          );
        }
      }
    }
  });
};

const makeConvDwKernel = (cfg: Conv2dHwc4Cfg) => {
  const { kH, kW, stride, padding } = cfg;
  const c4n = Math.ceil(cfg.cOut / 4);
  const kk = kH * kW;
  const epilogue = makeEpilogue(cfg);
  return tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const dims = convLayout.$.dims;
    const hOut = d.u32((dims.h + 2 * padding - kH) / stride) + 1;
    const wOut = d.u32((dims.w + 2 * padding - kW) / stride) + 1;
    const idx = flatIndex(gid);
    if (idx >= c4n * hOut * wOut) {
      return;
    }
    const ox = idx % wOut;
    const oy = d.u32(idx / wOut) % hOut;
    const cB = d.u32(idx / (wOut * hOut));

    let acc = d.vec4h();
    for (let ky = d.u32(0); ky < kH; ky++) {
      const iyP = oy * stride + ky;
      if (iyP >= padding && iyP < dims.h + padding) {
        const iy = iyP - padding;
        for (let kx = d.u32(0); kx < kW; kx++) {
          const ixP = ox * stride + kx;
          if (ixP >= padding && ixP < dims.w + padding) {
            const ix = ixP - padding;
            acc +=
              convLayout.$.w[cB * kk + ky * kW + kx]! *
              convLayout.$.x[(iy * dims.w + ix) * c4n + cB]!;
          }
        }
      }
    }
    const pxOut = oy * wOut + ox;
    convLayout.$.out[pxOut * dims.outC4 + dims.outOffB + cB] = epilogue(acc, cB, pxOut);
  });
};

const HWC4_ACTS: readonly number[] = [ACT_CODE.none, ACT_CODE.silu, ACT_CODE.gelu, ACT_CODE.relu];

export function createConv2dHwc4Pipeline(root: TgpuRoot, cfg: Conv2dHwc4Cfg) {
  if (!HWC4_ACTS.includes(cfg.hasAct)) {
    throw new Error(
      `conv2d hwc4: no f16 body for activation code ${cfg.hasAct} — this kernel fuses only none/silu/gelu/relu (ACT_CODE)`,
    );
  }
  // The split-K kernel's entrypoint takes workgroup builtins, the others a
  // global id; the two signatures do not unify in one expression.
  if (cfg.kind === 3) {
    return root.createComputePipeline({ compute: makeConv1x1SmallKernel(cfg) });
  }
  const kernel =
    cfg.kind === 1
      ? makeConv1x1Kernel(cfg)
      : cfg.kind === 2
        ? makeConvDwKernel(cfg)
        : makeConvGenericKernel(cfg);
  return root.createComputePipeline({ compute: kernel });
}

export function conv2dHwc4Handle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createConv2dHwc4Pipeline>,
  cfg: Conv2dHwc4Cfg,
  args: { h: number; w: number; outC4?: number; outOffB?: number },
  buffers: { x: F16Buffer; w: F16Buffer; bias: F16Buffer; res: F16Buffer; out: F16Buffer },
): KernelHandle {
  const { h, w } = args;
  const co4 = Math.ceil(cfg.cOut / 4);
  const dims = cachedUniform(root, ConvDims, {
    h,
    w,
    outC4: args.outC4 ?? co4,
    outOffB: args.outOffB ?? 0,
  });
  const bindGroup = cachedBindGroup(root, convLayout, { ...buffers, dims });
  const shape = `${cfg.cIn}>${cfg.cOut}@${h}x${w}`;
  if (cfg.kind === 1) {
    const threads = ((h * w) / (cfg.pxT === 8 ? 8 : 4)) * co4;
    return makeHandle(
      pipeline,
      'conv2d_hwc4_1x1',
      bindGroup,
      Math.ceil(threads / WG_1X1),
      `conv2d_hwc4_1x1 ${shape}`,
    );
  }
  if (cfg.kind === 3) {
    const oLanes = WG_1X1 / (cfg.ks ?? small1x1Lanes(cfg.cIn));
    const groups = Math.ceil(co4 / oLanes) * Math.ceil((h * w) / SMALL_1X1_PXT);
    return makeHandle(pipeline, 'conv2d_hwc4_1x1s', bindGroup, groups, `conv2d_hwc4_1x1s ${shape}`);
  }
  const hOut = Math.floor((h + 2 * cfg.padding - cfg.kH) / cfg.stride) + 1;
  const wOut = Math.floor((w + 2 * cfg.padding - cfg.kW) / cfg.stride) + 1;
  const name = cfg.kind === 2 ? 'conv2d_hwc4_dw' : 'conv2d_hwc4';
  return makeHandle(
    pipeline,
    name,
    bindGroup,
    Math.ceil((co4 * hOut * wOut) / WORKGROUP_SIZE),
    `${name} ${shape}`,
  );
}

// ---------------------------------------------------------------- pool / upsample

const MoveDims = d.struct({ c4: d.u32, h: d.u32, w: d.u32, outC4: d.u32, outOffB: d.u32 });

const moveLayout = tgpu.bindGroupLayout({
  x: { storage: d.arrayOf(d.vec4h), access: 'readonly' },
  out: { storage: d.arrayOf(d.vec4h), access: 'mutable' },
  dims: { uniform: MoveDims },
});

export interface MaxPool2dHwc4Cfg {
  kernelSize: number;
  stride: number;
  padding: number;
}

export interface AvgPool2dHwc4Cfg {
  kernelSize: number;
  stride: number;
}

const makeMaxPoolHwc4Kernel = (cfg: MaxPool2dHwc4Cfg) => {
  const { kernelSize: k, stride, padding } = cfg;
  return tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const dims = moveLayout.$.dims;
    const hOut = d.u32((dims.h + 2 * padding - k) / stride) + 1;
    const wOut = d.u32((dims.w + 2 * padding - k) / stride) + 1;
    const idx = flatIndex(gid);
    if (idx >= dims.c4 * hOut * wOut) {
      return;
    }
    const ox = idx % wOut;
    const oy = d.u32(idx / wOut) % hOut;
    const cB = d.u32(idx / (wOut * hOut));

    let best = d.vec4h(-65504.0); // f16 lowest; every window has >= 1 valid tap
    for (let ky = d.u32(0); ky < k; ky++) {
      const iyP = oy * stride + ky;
      if (iyP >= padding && iyP < dims.h + padding) {
        const iy = iyP - padding;
        for (let kx = d.u32(0); kx < k; kx++) {
          const ixP = ox * stride + kx;
          if (ixP >= padding && ixP < dims.w + padding) {
            best = std.max(best, moveLayout.$.x[(iy * dims.w + (ixP - padding)) * dims.c4 + cB]!);
          }
        }
      }
    }
    moveLayout.$.out[(oy * wOut + ox) * dims.outC4 + dims.outOffB + cB] = d.vec4h(best);
  });
};

const makeUpsampleHwc4Kernel = (scale: number) =>
  tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const dims = moveLayout.$.dims;
    const hOut = dims.h * scale;
    const wOut = dims.w * scale;
    const idx = flatIndex(gid);
    if (idx >= dims.c4 * hOut * wOut) {
      return;
    }
    const ox = idx % wOut;
    const oy = d.u32(idx / wOut) % hOut;
    const cB = d.u32(idx / (wOut * hOut));
    moveLayout.$.out[(oy * wOut + ox) * dims.outC4 + dims.outOffB + cB] = d.vec4h(
      moveLayout.$.x[(d.u32(oy / scale) * dims.w + d.u32(ox / scale)) * dims.c4 + cB]!,
    );
  });

const makeAvgPoolHwc4Kernel = (cfg: AvgPool2dHwc4Cfg) => {
  const { kernelSize: k, stride } = cfg;
  const inv = 1 / (k * k);
  return tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const dims = moveLayout.$.dims;
    const hOut = d.u32((dims.h - k) / stride) + 1;
    const wOut = d.u32((dims.w - k) / stride) + 1;
    const idx = flatIndex(gid);
    if (idx >= dims.c4 * hOut * wOut) {
      return;
    }
    const ox = idx % wOut;
    const oy = d.u32(idx / wOut) % hOut;
    const cB = d.u32(idx / (wOut * hOut));

    // f32 accumulation: a k² sum of f16 values drifts too far at k up to 8.
    let acc = d.vec4f();
    for (let ky = d.u32(0); ky < k; ky++) {
      for (let kx = d.u32(0); kx < k; kx++) {
        acc += d.vec4f(
          moveLayout.$.x[((oy * stride + ky) * dims.w + (ox * stride + kx)) * dims.c4 + cB]!,
        );
      }
    }
    moveLayout.$.out[(oy * wOut + ox) * dims.c4 + cB] = d.vec4h(acc * inv);
  });
};

const makePad2dHwc4Kernel = (padH: number, padW: number) =>
  tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const dims = moveLayout.$.dims;
    const wOut = dims.w + 2 * padW;
    const hOut = dims.h + 2 * padH;
    const idx = flatIndex(gid);
    if (idx >= dims.c4 * hOut * wOut) {
      return;
    }
    const ox = idx % wOut;
    const oy = d.u32(idx / wOut) % hOut;
    const cB = d.u32(idx / (wOut * hOut));
    let v = d.vec4h();
    if (ox >= padW && ox < dims.w + padW && oy >= padH && oy < dims.h + padH) {
      v = d.vec4h(moveLayout.$.x[((oy - padH) * dims.w + (ox - padW)) * dims.c4 + cB]!);
    }
    moveLayout.$.out[(oy * wOut + ox) * dims.c4 + cB] = d.vec4h(v);
  });

const ResizeDims = d.struct({
  c4: d.u32,
  h: d.u32,
  w: d.u32,
  outH: d.u32,
  outW: d.u32,
  hScale: d.f32,
  wScale: d.f32,
});

const resizeLayout = tgpu.bindGroupLayout({
  x: { storage: d.arrayOf(d.vec4h), access: 'readonly' },
  out: { storage: d.arrayOf(d.vec4h), access: 'mutable' },
  dims: { uniform: ResizeDims },
});

const makeResizeBilinearHwc4Kernel = (alignCorners: boolean) =>
  tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const dims = resizeLayout.$.dims;
    const idx = flatIndex(gid);
    if (idx >= dims.c4 * dims.outH * dims.outW) {
      return;
    }
    const ox = idx % dims.outW;
    const oy = d.u32(idx / dims.outW) % dims.outH;
    const cB = d.u32(idx / (dims.outW * dims.outH));

    let sy = d.f32(0);
    let sx = d.f32(0);
    if (alignCorners) {
      sy = d.f32(oy) * dims.hScale;
      sx = d.f32(ox) * dims.wScale;
    }
    if (!alignCorners) {
      sy = std.max((d.f32(oy) + 0.5) * dims.hScale - 0.5, d.f32(0));
      sx = std.max((d.f32(ox) + 0.5) * dims.wScale - 0.5, d.f32(0));
    }
    const y0 = std.min(d.u32(sy), dims.h - 1);
    const x0 = std.min(d.u32(sx), dims.w - 1);
    const y1 = std.min(y0 + 1, dims.h - 1);
    const x1 = std.min(x0 + 1, dims.w - 1);
    const fy = sy - d.f32(y0);
    const fx = sx - d.f32(x0);

    const v00 = d.vec4f(resizeLayout.$.x[(y0 * dims.w + x0) * dims.c4 + cB]!);
    const v01 = d.vec4f(resizeLayout.$.x[(y0 * dims.w + x1) * dims.c4 + cB]!);
    const v10 = d.vec4f(resizeLayout.$.x[(y1 * dims.w + x0) * dims.c4 + cB]!);
    const v11 = d.vec4f(resizeLayout.$.x[(y1 * dims.w + x1) * dims.c4 + cB]!);
    const top = v00 * (1 - fx) + v01 * fx;
    const bot = v10 * (1 - fx) + v11 * fx;
    resizeLayout.$.out[(oy * dims.outW + ox) * dims.c4 + cB] = d.vec4h(top * (1 - fy) + bot * fy);
  });

const AffineDims = d.struct({ c: d.u32, h: d.u32, w: d.u32 });

const affineLayout = tgpu.bindGroupLayout({
  x: { storage: d.arrayOf(d.vec4h), access: 'readonly' },
  scale: { storage: d.arrayOf(d.f32), access: 'readonly' },
  shift: { storage: d.arrayOf(d.f32), access: 'readonly' }, // 1-elem dummy when hasShift=0
  out: { storage: d.arrayOf(d.vec4h), access: 'mutable' },
  dims: { uniform: AffineDims },
});

const makeChannelAffineHwc4Kernel = (hasShift: boolean) =>
  tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const dims = affineLayout.$.dims;
    const c4 = d.u32((dims.c + 3) / 4);
    const idx = flatIndex(gid);
    if (idx >= dims.h * dims.w * c4) {
      return;
    }
    const cB = idx % c4;
    const c0 = cB * 4;
    const last = dims.c - 1;
    const zero = d.f32(0);
    const sc = d.vec4f(
      std.select(zero, affineLayout.$.scale[std.min(c0, last)]!, c0 < dims.c),
      std.select(zero, affineLayout.$.scale[std.min(c0 + 1, last)]!, c0 + 1 < dims.c),
      std.select(zero, affineLayout.$.scale[std.min(c0 + 2, last)]!, c0 + 2 < dims.c),
      std.select(zero, affineLayout.$.scale[std.min(c0 + 3, last)]!, c0 + 3 < dims.c),
    );
    let r = d.vec4f(affineLayout.$.x[idx]!) * sc;
    if (hasShift) {
      r += d.vec4f(
        std.select(zero, affineLayout.$.shift[std.min(c0, last)]!, c0 < dims.c),
        std.select(zero, affineLayout.$.shift[std.min(c0 + 1, last)]!, c0 + 1 < dims.c),
        std.select(zero, affineLayout.$.shift[std.min(c0 + 2, last)]!, c0 + 2 < dims.c),
        std.select(zero, affineLayout.$.shift[std.min(c0 + 3, last)]!, c0 + 3 < dims.c),
      );
    }
    affineLayout.$.out[idx] = d.vec4h(r);
  });

export function createMaxPool2dHwc4Pipeline(root: TgpuRoot, cfg: MaxPool2dHwc4Cfg) {
  return root.createComputePipeline({ compute: makeMaxPoolHwc4Kernel(cfg) });
}

export function createAvgPool2dHwc4Pipeline(root: TgpuRoot, cfg: AvgPool2dHwc4Cfg) {
  return root.createComputePipeline({ compute: makeAvgPoolHwc4Kernel(cfg) });
}

export function createPad2dHwc4Pipeline(root: TgpuRoot, cfg: { padH: number; padW: number }) {
  return root.createComputePipeline({ compute: makePad2dHwc4Kernel(cfg.padH, cfg.padW) });
}

export function createResizeBilinearHwc4Pipeline(root: TgpuRoot, cfg: { alignCorners: number }) {
  return root.createComputePipeline({
    compute: makeResizeBilinearHwc4Kernel(cfg.alignCorners === 1),
  });
}

export function createChannelAffineHwc4Pipeline(root: TgpuRoot, cfg: { hasShift: number }) {
  return root.createComputePipeline({ compute: makeChannelAffineHwc4Kernel(cfg.hasShift === 1) });
}

export function resizeBilinearHwc4Handle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createResizeBilinearHwc4Pipeline>,
  args: { c: number; h: number; w: number; outH: number; outW: number; alignCorners: boolean },
  buffers: { x: F16Buffer; out: F16Buffer },
): KernelHandle {
  const { c, h, w, outH, outW, alignCorners } = args;
  const c4 = Math.ceil(c / 4);
  const hScale = alignCorners ? (outH > 1 ? (h - 1) / (outH - 1) : 0) : h / outH;
  const wScale = alignCorners ? (outW > 1 ? (w - 1) / (outW - 1) : 0) : w / outW;
  const dims = cachedUniform(root, ResizeDims, { c4, h, w, outH, outW, hScale, wScale });
  const bindGroup = cachedBindGroup(root, resizeLayout, { ...buffers, dims });
  return makeHandle(
    pipeline,
    'resizeBilinear_hwc4',
    bindGroup,
    Math.ceil((c4 * outH * outW) / WORKGROUP_SIZE),
  );
}

export function channelAffineHwc4Handle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createChannelAffineHwc4Pipeline>,
  args: { c: number; h: number; w: number },
  buffers: { x: F16Buffer; scale: F32Buffer; shift: F32Buffer; out: F16Buffer },
): KernelHandle {
  const { c, h, w } = args;
  const dims = cachedUniform(root, AffineDims, { c, h, w });
  const bindGroup = cachedBindGroup(root, affineLayout, { ...buffers, dims });
  const threads = h * w * Math.ceil(c / 4);
  return makeHandle(pipeline, 'channelAffine_hwc4', bindGroup, Math.ceil(threads / WORKGROUP_SIZE));
}

export function createUpsample2dHwc4Pipeline(root: TgpuRoot, cfg: { scale: number }) {
  return root.createComputePipeline({ compute: makeUpsampleHwc4Kernel(cfg.scale) });
}

export type MoveHwc4Pipeline =
  | ReturnType<typeof createMaxPool2dHwc4Pipeline>
  | ReturnType<typeof createAvgPool2dHwc4Pipeline>
  | ReturnType<typeof createPad2dHwc4Pipeline>
  | ReturnType<typeof createUpsample2dHwc4Pipeline>;

export function moveHwc4Handle(
  root: TgpuRoot,
  pipeline: MoveHwc4Pipeline,
  name: string,
  args: { c: number; h: number; w: number; outElems: number; outC4?: number; outOffB?: number },
  buffers: { x: F16Buffer; out: F16Buffer },
): KernelHandle {
  const c4 = Math.ceil(args.c / 4);
  const dims = cachedUniform(root, MoveDims, {
    c4,
    h: args.h,
    w: args.w,
    outC4: args.outC4 ?? c4,
    outOffB: args.outOffB ?? 0,
  });
  const bindGroup = cachedBindGroup(root, moveLayout, { ...buffers, dims });
  return makeHandle(
    pipeline,
    name,
    bindGroup,
    Math.ceil(args.outElems / 4 / WORKGROUP_SIZE),
    `${name} ${args.c}@${args.h}x${args.w}`,
  );
}

// ---------------------------------------------------------------- channel copy

const CopyDims = d.struct({
  pElems: d.u32,
  srcC4: d.u32,
  dstC4: d.u32,
  srcOffB: d.u32,
  dstOffB: d.u32,
  nB: d.u32,
});

const copyLayout = tgpu.bindGroupLayout({
  x: { storage: d.arrayOf(d.vec4h), access: 'readonly' },
  out: { storage: d.arrayOf(d.vec4h), access: 'mutable' },
  dims: { uniform: CopyDims },
});

const copyChHwc4Kernel = tgpu.computeFn({
  in: { gid: d.builtin.globalInvocationId },
  workgroupSize: [WORKGROUP_SIZE],
})(({ gid }) => {
  'use gpu';
  const dims = copyLayout.$.dims;
  const idx = flatIndex(gid);
  if (idx >= dims.pElems * dims.nB) {
    return;
  }
  const px = d.u32(idx / dims.nB);
  const b = idx % dims.nB;
  copyLayout.$.out[px * dims.dstC4 + dims.dstOffB + b] = d.vec4h(
    copyLayout.$.x[px * dims.srcC4 + dims.srcOffB + b]!,
  );
});

export function createCopyChHwc4Pipeline(root: TgpuRoot) {
  return root.createComputePipeline({ compute: copyChHwc4Kernel });
}

export function copyChHwc4Handle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createCopyChHwc4Pipeline>,
  args: {
    pElems: number;
    srcC4: number;
    dstC4: number;
    srcOffB: number;
    dstOffB: number;
    nB: number;
  },
  buffers: { x: F16Buffer; out: F16Buffer },
): KernelHandle {
  const dims = cachedUniform(root, CopyDims, args);
  const bindGroup = cachedBindGroup(root, copyLayout, { ...buffers, dims });
  return makeHandle(
    pipeline,
    'copyCh_hwc4',
    bindGroup,
    Math.ceil((args.pElems * args.nB) / WORKGROUP_SIZE),
  );
}

// ---------------------------------------------------------------- CHW ↔ HWC4

const ConvertDims = d.struct({ c: d.u32, h: d.u32, w: d.u32 });

const makeToHwc4Layout = (srcF16: boolean) =>
  tgpu.bindGroupLayout({
    x: { storage: d.arrayOf(srcF16 ? d.f16 : d.f32), access: 'readonly' },
    out: { storage: d.arrayOf(d.vec4h), access: 'mutable' },
    dims: { uniform: ConvertDims },
  });
const toHwc4Layouts = { f32: makeToHwc4Layout(false), f16: makeToHwc4Layout(true) };

const makeToHwc4Kernel = (srcKey: 'f32' | 'f16') => {
  const layout = toHwc4Layouts[srcKey];
  return tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const dims = layout.$.dims;
    const p = dims.h * dims.w;
    const c4 = d.u32((dims.c + 3) / 4);
    const idx = flatIndex(gid);
    if (idx >= p * c4) {
      return;
    }
    const px = d.u32(idx / c4);
    const cB = idx % c4;
    const c0 = cB * 4;
    const zero = d.f16(0);
    const last = dims.c - 1;
    layout.$.out[px * c4 + cB] = d.vec4h(
      std.select(zero, d.f16(layout.$.x[std.min(c0, last) * p + px]!), c0 < dims.c),
      std.select(zero, d.f16(layout.$.x[std.min(c0 + 1, last) * p + px]!), c0 + 1 < dims.c),
      std.select(zero, d.f16(layout.$.x[std.min(c0 + 2, last) * p + px]!), c0 + 2 < dims.c),
      std.select(zero, d.f16(layout.$.x[std.min(c0 + 3, last) * p + px]!), c0 + 3 < dims.c),
    );
  });
};
const toHwc4Kernels = { f32: makeToHwc4Kernel('f32'), f16: makeToHwc4Kernel('f16') };

const makeToChwLayout = (dstF16: boolean) =>
  tgpu.bindGroupLayout({
    x: { storage: d.arrayOf(d.vec4h), access: 'readonly' },
    out: { storage: d.arrayOf(dstF16 ? d.f16 : d.f32), access: 'mutable' },
    dims: { uniform: ConvertDims },
  });
const toChwLayouts = { f32: makeToChwLayout(false), f16: makeToChwLayout(true) };

const makeToChwKernel = (dstKey: 'f32' | 'f16') => {
  const layout = toChwLayouts[dstKey];
  const store = dstKey === 'f16' ? d.f16 : d.f32;
  return tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const dims = layout.$.dims;
    const p = dims.h * dims.w;
    const c4 = d.u32((dims.c + 3) / 4);
    const idx = flatIndex(gid);
    if (idx >= p * c4) {
      return;
    }
    const px = d.u32(idx / c4);
    const cB = idx % c4;
    const v = layout.$.x[px * c4 + cB]!;
    const c0 = cB * 4;
    for (const k of tgpu.unroll(std.range(4))) {
      if (c0 + k < dims.c) {
        layout.$.out[(c0 + k) * p + px] = store(d.f32([v.x, v.y, v.z, v.w][k]!));
      }
    }
  });
};
const toChwKernels = { f32: makeToChwKernel('f32'), f16: makeToChwKernel('f16') };

export function createToHwc4Pipeline(root: TgpuRoot, srcKey: 'f32' | 'f16') {
  return root.createComputePipeline({ compute: toHwc4Kernels[srcKey] });
}

export function createToChwPipeline(root: TgpuRoot, dstKey: 'f32' | 'f16') {
  return root.createComputePipeline({ compute: toChwKernels[dstKey] });
}

export function toHwc4Handle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createToHwc4Pipeline>,
  srcKey: 'f32' | 'f16',
  args: { c: number; h: number; w: number },
  buffers: { x: F32Buffer | F16Buffer; out: F16Buffer },
): KernelHandle {
  const { c, h, w } = args;
  const dims = cachedUniform(root, ConvertDims, { c, h, w });
  const bindGroup = cachedBindGroup(root, toHwc4Layouts[srcKey], { ...buffers, dims });
  const threads = h * w * Math.ceil(c / 4);
  return makeHandle(pipeline, 'toHwc4', bindGroup, Math.ceil(threads / WORKGROUP_SIZE));
}

export function toChwHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createToChwPipeline>,
  dstKey: 'f32' | 'f16',
  args: { c: number; h: number; w: number },
  buffers: { x: F16Buffer; out: F32Buffer | F16Buffer },
): KernelHandle {
  const { c, h, w } = args;
  const dims = cachedUniform(root, ConvertDims, { c, h, w });
  const bindGroup = cachedBindGroup(root, toChwLayouts[dstKey], { ...buffers, dims });
  const threads = h * w * Math.ceil(c / 4);
  return makeHandle(pipeline, 'toChw', bindGroup, Math.ceil(threads / WORKGROUP_SIZE));
}
