/** The HWC4 kernel family: conv, pool, upsample, channel-block copy and the
 *  CHW ↔ HWC4 boundary conversions.
 *
 *  Layout contract, shared with nn.Conv2d's weight repack and the hwc4 ops.
 *  Activations are `array<vec4<f16>>` at element (y·W + x)·C4 + cB, pixel-major
 *  with channels zero-padded to a multiple of 4. Conv weights are mat4 tiles of
 *  4 vec4<f16>, tile (oB, iB, tap) at ((oB·CI4 + iB)·kH·kW + tap), entry iL
 *  holding the output lanes for input lane iL; a stride-1 1×1 conv orders its
 *  tiles (iB, oB) instead, at iB·CO4 + oB. Depthwise weights are one vec4
 *  per (cB, tap). Padded weight lanes are zero, so padded activation lanes stay
 *  0 through the epilogue.
 *
 *  Kernels are factories over plain JS numbers, so geometry reaches WGSL as
 *  literals and branches prune. Input h and w stay a runtime uniform.
 *
 *  Per-element kernels take the channel block as the fastest index, so
 *  neighbouring threads touch neighbouring vec4s. */

import tgpu, { d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import {
  type F16Buffer,
  type F32Buffer,
  flatIndex,
  flatWorkgroupId,
  type KernelHandle,
  makeHandle,
  WORKGROUP_SIZE,
} from '../common.ts';
import { ACT_CODE, gelu } from '../activations.ts';
import { cachedBindGroup, cachedUniform } from '../../gpu/dispatchCache.ts';

export const WG_1X1 = 256;

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
  /** 1×1 (kind 1): input blocks per K-loop step. */
  kU?: number;
  ks?: number;
  /** Depthwise (kind 2): output columns per thread. */
  dwCols?: number;
  /** Direct (kind 0): output columns per thread. */
  cols?: number;
  /** Tiled (kind 4): blocks of 4 output channels per workgroup. */
  bmb?: number;
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

/** Output channel blocks up to which the direct conv orders threads block-first. */
const DIRECT_NARROW_BLOCKS = 8;

/** Fewest threads a direct conv keeps when it gives each several columns. */
const DIRECT_MIN_THREADS = 16384;

/** Output columns per direct-conv thread: 4, 2 or 1, the most that keeps
 *  DIRECT_MIN_THREADS threads. More columns share more input loads. */
export function directCols(co4: number, hOut: number, wOut: number): number {
  return [4, 2].find((c) => co4 * hOut * Math.ceil(wOut / c) >= DIRECT_MIN_THREADS) ?? 1;
}

/** Direct conv. A thread computes `cols` neighbouring output columns of one
 *  channel block, so each input row it loads serves every tap of every column. */
export const makeConvGenericKernel = (cfg: Conv2dHwc4Cfg) => {
  const { kH, kW, stride, padding } = cfg;
  const ci4 = Math.ceil(cfg.cIn / 4);
  const co4 = Math.ceil(cfg.cOut / 4);
  const kk = kH * kW;
  const cols = cfg.cols ?? 1;
  const span = kW + (cols - 1) * stride;
  const narrowOut = co4 <= DIRECT_NARROW_BLOCKS;
  const epilogue = makeEpilogue(cfg);
  return tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const dims = convLayout.$.dims;
    const hOut = std.intdiv(dims.h + 2 * padding - kH, stride) + 1;
    const wOut = std.intdiv(dims.w + 2 * padding - kW, stride) + 1;
    const wRuns = std.intdiv(wOut + cols - 1, cols);
    const idx = flatIndex(gid);
    if (idx >= co4 * hOut * wRuns) {
      return;
    }
    // Narrow outputs put the channel block fastest: lanes share input loads
    // and store whole pixels. Wide ones put the column run fastest, so lanes
    // share one weight row.
    const oB = narrowOut ? idx % co4 : std.intdiv(idx, wRuns * hOut);
    const ox0 = (narrowOut ? std.intdiv(idx, co4) % wRuns : idx % wRuns) * cols;
    const oy = narrowOut ? std.intdiv(idx, co4 * wRuns) : std.intdiv(idx, wRuns) % hOut;

    const accs = d.arrayOf(d.vec4h, cols)();
    for (const ky of tgpu.unroll(std.range(kH))) {
      const iyP = oy * stride + ky;
      if (iyP >= padding && iyP < dims.h + padding) {
        const rowBase = (iyP - padding) * dims.w;
        for (let iB = d.u32(0); iB < ci4; iB++) {
          const xs = d.arrayOf(d.vec4h, span)();
          for (const s of tgpu.unroll(std.range(span))) {
            const ixP = ox0 * stride + s;
            if (ixP >= padding && ixP < dims.w + padding) {
              xs[s] = d.vec4h(convLayout.$.x[(rowBase + ixP - padding) * ci4 + iB]!);
            }
          }
          for (const kx of tgpu.unroll(std.range(kW))) {
            // u32-wrapped: a bare JS int would drag the index math to i32.
            const tb = ((oB * ci4 + iB) * kk + d.u32(ky * kW + kx)) * 4;
            const w0 = convLayout.$.w[tb]!;
            const w1 = convLayout.$.w[tb + 1]!;
            const w2 = convLayout.$.w[tb + 2]!;
            const w3 = convLayout.$.w[tb + 3]!;
            for (const c of tgpu.unroll(std.range(cols))) {
              const xv = xs[kx + c * stride]!;
              accs[c] = accs[c]! + xv.x * w0 + xv.y * w1 + xv.z * w2 + xv.w * w3;
            }
          }
        }
      }
    }
    for (const c of tgpu.unroll(std.range(cols))) {
      const ox = ox0 + c;
      if (ox < wOut) {
        const pxOut = oy * wOut + ox;
        convLayout.$.out[pxOut * dims.outC4 + dims.outOffB + oB] = epilogue(accs[c]!, oB, pxOut);
      }
    }
  });
};

// ---------------------------------------------------------------- tiled conv
// Runs a full conv as a matrix multiply. A workgroup computes 64 output
// pixels × up to 64 output channels: it loads a block of weights and input
// pixels into workgroup memory once, and every thread reuses them. Sums in
// f16. convTiledBlocks picks the convs that use it.
//
// Based on webgpu-kernels/ai.onnx.Conv (conv-1x1-gemm-tiled-reg).

const TILED_BN = 64; // output pixels per workgroup
const TILED_WG_X = 16; // pixel lanes, 4 pixels each
const TILED_BKG = 8; // K groups (vec4 of input lanes) per K tile
/** Most channel blocks per workgroup (64 channels). */
const TILED_MAX_BLOCKS = 16;
/** Fewer workgroups than this leave the GPU idle. */
const TILED_MIN_WORKGROUPS = 24;
/** Outputs under TILED_WIDE_COUT channels need this many pixels for the tiled kernel. */
const TILED_NARROW_MIN_PIXELS = 6400;
const TILED_WIDE_COUT = 64;
const TILED_MIN_COUT = 32;
/** Down to TILED_DEEP_MIN_COUT channels, a conv uses the tiled kernel on maps of
 *  TILED_DEEP_MIN_PIXELS or more when each output reads TILED_DEEP_MIN_K inputs. */
const TILED_DEEP_MIN_COUT = 24;
const TILED_DEEP_MIN_PIXELS = 3136;
const TILED_DEEP_MIN_K = 64;

/** Channel blocks per workgroup: 16, or 8 or 4 for narrow outputs. */
function tiledBlocks(cOut: number): number {
  const co4 = Math.ceil(cOut / 4);
  return co4 >= TILED_MAX_BLOCKS ? TILED_MAX_BLOCKS : co4 > 4 ? 8 : 4;
}

/** Channel blocks per workgroup when this conv uses the tiled kernel, else 0.
 *  `k` is how many inputs each output reads: cIn·kH·kW. */
export function convTiledBlocks(cOut: number, outPixels: number, k: number): number {
  const bmb = tiledBlocks(cOut);
  const workgroups = Math.ceil(outPixels / TILED_BN) * Math.ceil(cOut / 4 / bmb);
  const wide = cOut >= TILED_WIDE_COUT;
  const narrow = cOut >= TILED_MIN_COUT && outPixels >= TILED_NARROW_MIN_PIXELS;
  const deep =
    cOut >= TILED_DEEP_MIN_COUT && outPixels >= TILED_DEEP_MIN_PIXELS && k >= TILED_DEEP_MIN_K;
  return workgroups >= TILED_MIN_WORKGROUPS && (wide || narrow || deep) ? bmb : 0;
}

const makeConvTiledKernel = (cfg: Conv2dHwc4Cfg) => {
  const { kH, kW, stride, padding } = cfg;
  const ci4 = Math.ceil(cfg.cIn / 4);
  const co4 = Math.ceil(cfg.cOut / 4);
  const kk = kH * kW;
  const kGroups = kk * ci4;
  const tiles = Math.ceil(kGroups / TILED_BKG);
  const TILED_BMB = cfg.bmb ?? TILED_MAX_BLOCKS;
  const TILED_THREADS = TILED_WG_X * (TILED_BMB / 2);
  const TILED_A_LOADS = (TILED_BKG * TILED_BMB * 4) / TILED_THREADS;
  const TILED_B_LOADS = (TILED_BKG * TILED_BN) / TILED_THREADS;
  const tiledA = tgpu.workgroupVar(d.arrayOf(d.vec4h, TILED_BKG * TILED_BMB * 4));
  const tiledB = tgpu.workgroupVar(d.arrayOf(d.vec4h, TILED_BKG * TILED_BN));
  const epilogue = makeEpilogue(cfg);
  return tgpu.computeFn({
    in: { lid: d.builtin.localInvocationIndex, wid: d.builtin.workgroupId },
    workgroupSize: [TILED_THREADS],
  })(({ lid, wid }) => {
    'use gpu';
    const dims = convLayout.$.dims;
    const hOut = std.intdiv(dims.h + 2 * padding - kH, stride) + 1;
    const wOut = std.intdiv(dims.w + 2 * padding - kW, stride) + 1;
    const nPix = hOut * wOut;
    const tx = lid % TILED_WG_X;
    const ty = std.intdiv(lid, TILED_WG_X);
    const nBase = wid.x * TILED_BN;
    const oBase = wid.y * TILED_BMB;

    const accs = d.arrayOf(d.vec4h, 8)();

    for (const t of std.range(tiles)) {
      const kg0 = t * TILED_BKG;
      // Weights for this K block. Zero past cOut or K.
      for (const i of tgpu.unroll(std.range(TILED_A_LOADS))) {
        const e = lid + i * TILED_THREADS;
        const iL = e % 4;
        const oBl = std.intdiv(e, 4) % TILED_BMB;
        const kgl = std.intdiv(e, 4 * TILED_BMB);
        const kg = kg0 + kgl;
        const oB = oBase + oBl;
        const tap = std.intdiv(kg, ci4);
        const iB = kg % ci4;
        let v = d.vec4h();
        if (kg < kGroups && oB < co4) {
          v = d.vec4h(convLayout.$.w[((oB * ci4 + iB) * kk + tap) * 4 + iL]!);
        }
        tiledA.$[e] = d.vec4h(v);
      }
      // Input pixels for this K block. Zero in the padding and past the last pixel.
      for (const i of tgpu.unroll(std.range(TILED_B_LOADS))) {
        const e = lid + i * TILED_THREADS;
        const kgl = e % TILED_BKG;
        const px = std.intdiv(e, TILED_BKG);
        const kg = kg0 + kgl;
        const n = nBase + px;
        const tap = std.intdiv(kg, ci4);
        const iB = kg % ci4;
        const ky = std.intdiv(tap, kW);
        const kx = tap % kW;
        const oy = std.intdiv(n, wOut);
        const ox = n % wOut;
        const iyP = oy * stride + ky;
        const ixP = ox * stride + kx;
        let v = d.vec4h();
        if (
          kg < kGroups &&
          n < nPix &&
          iyP >= padding &&
          iyP < dims.h + padding &&
          ixP >= padding &&
          ixP < dims.w + padding
        ) {
          v = d.vec4h(convLayout.$.x[((iyP - padding) * dims.w + (ixP - padding)) * ci4 + iB]!);
        }
        tiledB.$[kgl * TILED_BN + px] = d.vec4h(v);
      }
      std.workgroupBarrier();

      for (const kgl of std.range(TILED_BKG)) {
        const xs = d.arrayOf(d.vec4h, 4)();
        for (const j of tgpu.unroll(std.range(4))) {
          xs[j] = d.vec4h(tiledB.$[kgl * TILED_BN + tx * 4 + j]!);
        }
        for (const o of tgpu.unroll(std.range(2))) {
          const wb = (kgl * TILED_BMB + ty * 2 + o) * 4;
          const w0 = d.vec4h(tiledA.$[wb]!);
          const w1 = d.vec4h(tiledA.$[wb + 1]!);
          const w2 = d.vec4h(tiledA.$[wb + 2]!);
          const w3 = d.vec4h(tiledA.$[wb + 3]!);
          for (const j of tgpu.unroll(std.range(4))) {
            const xv = xs[j]!;
            accs[o * 4 + j] = accs[o * 4 + j]! + xv.x * w0 + xv.y * w1 + xv.z * w2 + xv.w * w3;
          }
        }
      }
      std.workgroupBarrier();
    }

    for (const o of tgpu.unroll(std.range(2))) {
      const oB = oBase + ty * 2 + o;
      for (const j of tgpu.unroll(std.range(4))) {
        const n = nBase + tx * 4 + j;
        if (oB < co4 && n < nPix) {
          convLayout.$.out[n * dims.outC4 + dims.outOffB + oB] = epilogue(
            d.vec4h(accs[o * 4 + j]!),
            oB,
            n,
          );
        }
      }
    }
  });
};

/** Input blocks the 1×1 conv sums in f16 before adding the partial into its
 *  f32 sum, longest first: whichever divides CI4. Short f16 runs keep the
 *  speed of f16 math and most of the precision of f32 sums. */
const CONV1X1_FLUSH_STEPS = [8, 4, 2, 1];

/** Below this many threads a 1×1 conv has too few SIMD groups to hide load
 *  latency, so each thread issues several input blocks' loads up front. */
const CONV1X1_UNROLL_MAX_THREADS = 8192;

/** Input blocks per step of the 1×1 kernel's K loop: 4 or 2 on small launches
 *  (whatever divides CI4), else 1, which keeps registers free for occupancy. */
export function conv1x1Unroll(ci4: number, threads: number): number {
  if (threads >= CONV1X1_UNROLL_MAX_THREADS) return 1;
  return ci4 % 4 === 0 ? 4 : ci4 % 2 === 0 ? 2 : 1;
}

export const makeConv1x1Kernel = (cfg: Conv2dHwc4Cfg) => {
  const ci4 = Math.ceil(cfg.cIn / 4);
  const co4 = Math.ceil(cfg.cOut / 4);
  const pxT = cfg.pxT === 8 ? 8 : 4;
  const kU = cfg.kU ?? 1;
  const flush = CONV1X1_FLUSH_STEPS.find((c) => ci4 % c === 0 && c % kU === 0)!;
  const epilogue = makeEpilogue(cfg);
  return tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const dims = convLayout.$.dims;
    const pT = std.intdiv(dims.h * dims.w, pxT);
    const idx = flatIndex(gid);
    if (idx >= pT * co4) {
      return;
    }
    const p0 = std.intdiv(idx, co4) * pxT;
    const oB = idx % co4;

    const accs = d.arrayOf(d.vec4f, pxT)();
    for (let c0 = d.u32(0); c0 < ci4; c0 += flush) {
      const part = d.arrayOf(d.vec4h, pxT)();
      for (let iB0 = c0; iB0 < c0 + flush; iB0 += kU) {
        const ws = d.arrayOf(d.vec4h, kU * 4)();
        const xs = d.arrayOf(d.vec4h, kU * pxT)();
        for (const u of tgpu.unroll(std.range(kU))) {
          const tb = ((iB0 + u) * co4 + oB) * 4;
          for (const l of tgpu.unroll(std.range(4))) {
            ws[u * 4 + l] = d.vec4h(convLayout.$.w[tb + l]!);
          }
          for (const k of tgpu.unroll(std.range(pxT))) {
            xs[u * pxT + k] = d.vec4h(convLayout.$.x[(p0 + k) * ci4 + iB0 + u]!);
          }
        }
        for (const u of tgpu.unroll(std.range(kU))) {
          for (const k of tgpu.unroll(std.range(pxT))) {
            const xv = xs[u * pxT + k]!;
            part[k] =
              part[k]! +
              xv.x * ws[u * 4]! +
              xv.y * ws[u * 4 + 1]! +
              xv.z * ws[u * 4 + 2]! +
              xv.w * ws[u * 4 + 3]!;
          }
        }
      }
      for (const k of tgpu.unroll(std.range(pxT))) {
        accs[k] = accs[k]! + d.vec4f(part[k]!);
      }
    }
    const ob0 = dims.outOffB + oB;
    for (const k of tgpu.unroll(std.range(pxT))) {
      convLayout.$.out[(p0 + k) * dims.outC4 + ob0] = epilogue(d.vec4h(accs[k]!), oB, p0 + k);
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
    const oLane = std.intdiv(lid, SMALL_1X1_KS);
    const oB = (wid.x % oGroups) * SMALL_1X1_OL + oLane;
    const p0 = std.intdiv(wid.x, oGroups) * SMALL_1X1_PXT;

    const accs = d.arrayOf(d.vec4h, SMALL_1X1_PXT)();
    if (oB < co4) {
      for (let iB = kLane; iB < ci4; iB += SMALL_1X1_KS) {
        const tb = (iB * co4 + oB) * 4;
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

/** Maps this narrow or narrower use one column per thread, to keep enough threads. */
const DW_NARROW_MAX_W = 14;
/** Kernels this wide use two columns, even on narrow maps. */
const DW_WIDE_KERNEL = 7;

/** Output columns per depthwise thread. Two columns share input loads; one
 *  gives more threads. */
export function depthwiseCols(kW: number, wOut: number): number {
  return kW === 1 || (wOut <= DW_NARROW_MAX_W && kW < DW_WIDE_KERNEL) ? 1 : 2;
}

/** Depthwise conv. Each thread computes 1 or 2 neighbouring output pixels of
 *  one channel block, and neighbouring threads take neighbouring channel
 *  blocks, so memory reads line up. Each input row is loaded once and reused
 *  by every tap. Sums in f32.
 *
 *  Based on webgpu-kernels/ai.onnx.Conv (conv2d-grouped-large-w4). */
const makeConvDwKernel = (cfg: Conv2dHwc4Cfg) => {
  const { kH, kW, stride, padding } = cfg;
  const cols = cfg.dwCols ?? 1;
  const c4n = Math.ceil(cfg.cOut / 4);
  const kk = kH * kW;
  const span = kW + (cols - 1) * stride;
  const epilogue = makeEpilogue(cfg);
  return tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const dims = convLayout.$.dims;
    const hOut = std.intdiv(dims.h + 2 * padding - kH, stride) + 1;
    const wOut = std.intdiv(dims.w + 2 * padding - kW, stride) + 1;
    const wRuns = std.intdiv(wOut + cols - 1, cols);
    const idx = flatIndex(gid);
    if (idx >= c4n * hOut * wRuns) {
      return;
    }
    const cB = idx % c4n;
    const run = std.intdiv(idx, c4n) % wRuns;
    const oy = std.intdiv(idx, c4n * wRuns);
    const ox0 = run * cols;

    const accs = d.arrayOf(d.vec4f, cols)();
    const xs = d.arrayOf(d.vec4f, span)();
    for (const ky of tgpu.unroll(std.range(kH))) {
      const iyP = oy * stride + ky;
      if (iyP >= padding && iyP < dims.h + padding) {
        const rowBase = (iyP - padding) * dims.w;
        for (const s of tgpu.unroll(std.range(span))) {
          const ixP = ox0 * stride + s;
          xs[s] = d.vec4f();
          if (ixP >= padding && ixP < dims.w + padding) {
            xs[s] = d.vec4f(convLayout.$.x[(rowBase + ixP - padding) * c4n + cB]!);
          }
        }
        for (const kx of tgpu.unroll(std.range(kW))) {
          const wv = d.vec4f(convLayout.$.w[cB * kk + d.u32(ky * kW + kx)]!);
          for (const c of tgpu.unroll(std.range(cols))) {
            accs[c] = accs[c]! + wv * xs[kx + c * stride]!;
          }
        }
      }
    }
    for (const c of tgpu.unroll(std.range(cols))) {
      const ox = ox0 + c;
      if (ox < wOut) {
        const pxOut = oy * wOut + ox;
        convLayout.$.out[pxOut * dims.outC4 + dims.outOffB + cB] = epilogue(
          d.vec4h(accs[c]!),
          cB,
          pxOut,
        );
      }
    }
  });
};

// ---------------------------------------------------------------- Winograd F(4×4, 3×3)
// A 3×3, stride 1, padding 1 conv in three dispatches. Bᵀ·d·B turns the 6×6
// input patch of each 4×4 output tile into 36 coefficient planes, each plane
// is a 1×1 conv against the matching plane of G·g·Gᵀ (packed at load), and
// Aᵀ·m·A turns the 36 products back into the tile. A quarter of the direct
// conv's multiplies. Planes are hwc4-shaped with tiles as pixels: plane k,
// tile t, block b at (k·tiles + t)·C4 + b.

/** Coefficient planes of F(4×4, 3×3). */
const WINOGRAD_PLANES = 36;
/** Tiles per thread in the plane products. */
const WINOGRAD_MUL_TILES = 4;

/** Bᵀ applied to one row or column of six. */
const winogradInRow = (x0: d.v4f, x1: d.v4f, x2: d.v4f, x3: d.v4f, x4: d.v4f, x5: d.v4f) => {
  'use gpu';
  const out = d.arrayOf(d.vec4f, 6)();
  out[0] = x0 * 4 - x2 * 5 + x4;
  out[1] = x3 + x4 - (x1 + x2) * 4;
  out[2] = (x1 - x2) * 4 - x3 + x4;
  out[3] = (x3 - x1) * 2 - x2 + x4;
  out[4] = (x1 - x3) * 2 - x2 + x4;
  out[5] = x1 * 4 - x3 * 5 + x5;
  return out;
};

/** Aᵀ applied to one row or column of six. */
const winogradOutRow = (x0: d.v4f, x1: d.v4f, x2: d.v4f, x3: d.v4f, x4: d.v4f, x5: d.v4f) => {
  'use gpu';
  const out = d.arrayOf(d.vec4f, 4)();
  out[0] = x0 + x1 + x2 + x3 + x4;
  out[1] = x1 - x2 + (x3 - x4) * 2;
  out[2] = x1 + x2 + (x3 + x4) * 4;
  out[3] = x1 - x2 + (x3 - x4) * 8 + x5;
  return out;
};

/** (tile, channel block) pairs per transform workgroup. Six lanes share a
 *  pair: lane r takes row r of the separable transform, then column r. */
const WINOGRAD_PAIRS = 16;
const WINOGRAD_LANES = 6;
const winogradRows = tgpu.workgroupVar(d.arrayOf(d.vec4f, WINOGRAD_PAIRS * 36));

const makeWinogradInKernel = (cfg: Conv2dHwc4Cfg) => {
  const ci4 = Math.ceil(cfg.cIn / 4);
  return tgpu.computeFn({
    in: { lid: d.builtin.localInvocationIndex, wid: d.builtin.workgroupId },
    workgroupSize: [WINOGRAD_PAIRS * WINOGRAD_LANES],
  })(({ lid, wid }) => {
    'use gpu';
    const dims = convLayout.$.dims;
    const tilesX = std.intdiv(dims.w + 3, 4);
    const tiles = tilesX * std.intdiv(dims.h + 3, 4);
    const p = lid % WINOGRAD_PAIRS;
    const lane = std.intdiv(lid, WINOGRAD_PAIRS);
    const pair = flatWorkgroupId(wid) * WINOGRAD_PAIRS + p;
    const live = pair < tiles * ci4;
    const iB = pair % ci4;
    const t = std.intdiv(pair, ci4);
    // Patch row `lane` in padded coordinates: input (y, x) is (yP - 1, xP - 1).
    const yP = std.intdiv(t, tilesX) * 4 + lane;
    const x0 = (t % tilesX) * 4;

    const xs = d.arrayOf(d.vec4f, 6)();
    if (live && yP >= 1 && yP <= dims.h) {
      for (const c of tgpu.unroll(std.range(6))) {
        const xP = x0 + c;
        if (xP >= 1 && xP <= dims.w) {
          xs[c] = d.vec4f(convLayout.$.x[((yP - 1) * dims.w + xP - 1) * ci4 + iB]!);
        }
      }
    }
    const row = winogradInRow(xs[0]!, xs[1]!, xs[2]!, xs[3]!, xs[4]!, xs[5]!);
    const sb = p * 36;
    for (const c of tgpu.unroll(std.range(6))) {
      winogradRows.$[sb + lane * 6 + c] = d.vec4f(row[c]!);
    }
    std.workgroupBarrier();
    if (live) {
      const col = winogradInRow(
        winogradRows.$[sb + lane]!,
        winogradRows.$[sb + 6 + lane]!,
        winogradRows.$[sb + 12 + lane]!,
        winogradRows.$[sb + 18 + lane]!,
        winogradRows.$[sb + 24 + lane]!,
        winogradRows.$[sb + 30 + lane]!,
      );
      for (const r of tgpu.unroll(std.range(6))) {
        convLayout.$.out[((r * 6 + lane) * tiles + t) * ci4 + iB] = d.vec4h(col[r]!);
      }
    }
  });
};

/** The 36 plane products, each a 1×1 conv with tiles as pixels. A thread
 *  takes WINOGRAD_MUL_TILES tiles of one output block. Sums in f32. */
const makeWinogradMulKernel = (cfg: Conv2dHwc4Cfg) => {
  const ci4 = Math.ceil(cfg.cIn / 4);
  const co4 = Math.ceil(cfg.cOut / 4);
  return tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const dims = convLayout.$.dims;
    const tiles = std.intdiv(dims.w + 3, 4) * std.intdiv(dims.h + 3, 4);
    const groups = std.intdiv(tiles + WINOGRAD_MUL_TILES - 1, WINOGRAD_MUL_TILES);
    const idx = flatIndex(gid);
    if (idx >= WINOGRAD_PLANES * groups * co4) {
      return;
    }
    const oB = idx % co4;
    const t0 = (std.intdiv(idx, co4) % groups) * WINOGRAD_MUL_TILES;
    const k = std.intdiv(idx, co4 * groups);
    const plane = k * tiles;

    const accs = d.arrayOf(d.vec4f, WINOGRAD_MUL_TILES)();
    for (let iB = d.u32(0); iB < ci4; iB++) {
      const wb = ((k * ci4 + iB) * co4 + oB) * 4;
      const w0 = convLayout.$.w[wb]!;
      const w1 = convLayout.$.w[wb + 1]!;
      const w2 = convLayout.$.w[wb + 2]!;
      const w3 = convLayout.$.w[wb + 3]!;
      for (const j of tgpu.unroll(std.range(WINOGRAD_MUL_TILES))) {
        // Clamped so a partial last group reads valid tiles; its extra sums are never stored.
        const xv = convLayout.$.x[(plane + std.min(t0 + j, tiles - 1)) * ci4 + iB]!;
        accs[j] = accs[j]! + d.vec4f(xv.x * w0 + xv.y * w1 + xv.z * w2 + xv.w * w3);
      }
    }
    for (const j of tgpu.unroll(std.range(WINOGRAD_MUL_TILES))) {
      if (t0 + j < tiles) {
        convLayout.$.out[(plane + t0 + j) * co4 + oB] = d.vec4h(accs[j]!);
      }
    }
  });
};

/** Aᵀ·m·A, then the conv epilogue on each of the tile's 16 pixels. */
const makeWinogradOutKernel = (cfg: Conv2dHwc4Cfg) => {
  const co4 = Math.ceil(cfg.cOut / 4);
  const epilogue = makeEpilogue(cfg);
  return tgpu.computeFn({
    in: { lid: d.builtin.localInvocationIndex, wid: d.builtin.workgroupId },
    workgroupSize: [WINOGRAD_PAIRS * WINOGRAD_LANES],
  })(({ lid, wid }) => {
    'use gpu';
    const dims = convLayout.$.dims;
    const tilesX = std.intdiv(dims.w + 3, 4);
    const tiles = tilesX * std.intdiv(dims.h + 3, 4);
    const p = lid % WINOGRAD_PAIRS;
    const lane = std.intdiv(lid, WINOGRAD_PAIRS);
    const pair = flatWorkgroupId(wid) * WINOGRAD_PAIRS + p;
    const live = pair < tiles * co4;
    const oB = pair % co4;
    const t = std.intdiv(pair, co4);

    const ms = d.arrayOf(d.vec4f, 6)();
    if (live) {
      for (const c of tgpu.unroll(std.range(6))) {
        ms[c] = d.vec4f(convLayout.$.x[((lane * 6 + c) * tiles + t) * co4 + oB]!);
      }
    }
    const row = winogradOutRow(ms[0]!, ms[1]!, ms[2]!, ms[3]!, ms[4]!, ms[5]!);
    const sb = p * 36;
    for (const c of tgpu.unroll(std.range(4))) {
      winogradRows.$[sb + lane * 4 + c] = d.vec4f(row[c]!);
    }
    std.workgroupBarrier();
    const x = (t % tilesX) * 4 + lane;
    if (live && lane < 4 && x < dims.w) {
      const col = winogradOutRow(
        winogradRows.$[sb + lane]!,
        winogradRows.$[sb + 4 + lane]!,
        winogradRows.$[sb + 8 + lane]!,
        winogradRows.$[sb + 12 + lane]!,
        winogradRows.$[sb + 16 + lane]!,
        winogradRows.$[sb + 20 + lane]!,
      );
      const y0 = std.intdiv(t, tilesX) * 4;
      for (const r of tgpu.unroll(std.range(4))) {
        const y = y0 + r;
        if (y < dims.h) {
          const px = y * dims.w + x;
          convLayout.$.out[px * dims.outC4 + dims.outOffB + oB] = epilogue(
            d.vec4h(col[r]!),
            oB,
            px,
          );
        }
      }
    }
  });
};

export function createWinogradPipelines(root: TgpuRoot, cfg: Conv2dHwc4Cfg) {
  checkHwc4Act(cfg);
  return {
    input: root.createComputePipeline({ compute: makeWinogradInKernel(cfg) }),
    mul: root.createComputePipeline({ compute: makeWinogradMulKernel(cfg) }),
    output: root.createComputePipeline({ compute: makeWinogradOutKernel(cfg) }),
  };
}

/** The three Winograd dispatches. `planesIn` and `planesOut` are scratch of
 *  at least winogradPlaneElems(h, w, cIn) and (…, cOut) f16 values. */
export function winogradHandles(
  root: TgpuRoot,
  pipelines: ReturnType<typeof createWinogradPipelines>,
  cfg: Conv2dHwc4Cfg,
  args: { h: number; w: number; outC4?: number; outOffB?: number },
  buffers: {
    x: F16Buffer;
    w: F16Buffer;
    bias: F16Buffer;
    res: F16Buffer;
    out: F16Buffer;
    planesIn: F16Buffer | F32Buffer;
    planesOut: F16Buffer | F32Buffer;
  },
): KernelHandle[] {
  const { h, w } = args;
  const ci4 = Math.ceil(cfg.cIn / 4);
  const co4 = Math.ceil(cfg.cOut / 4);
  const tiles = Math.ceil(h / 4) * Math.ceil(w / 4);
  const dims = cachedUniform(root, ConvDims, {
    h,
    w,
    outC4: args.outC4 ?? co4,
    outOffB: args.outOffB ?? 0,
  });
  const { x, w: weight, bias, res, out, planesIn, planesOut } = buffers;
  const shape = `${cfg.cIn}>${cfg.cOut}@${h}x${w}`;
  // Bindings a stage does not read get the conv's own buffers as filler.
  return [
    makeHandle(
      pipelines.input,
      'conv2d_hwc4_wgIn',
      cachedBindGroup(root, convLayout, { x, w: weight, bias, res, out: planesIn, dims }),
      Math.ceil((tiles * ci4) / WINOGRAD_PAIRS),
      `conv2d_hwc4_wgIn ${shape}`,
    ),
    makeHandle(
      pipelines.mul,
      'conv2d_hwc4_wgMul',
      cachedBindGroup(root, convLayout, {
        x: planesIn,
        w: weight,
        bias,
        res,
        out: planesOut,
        dims,
      }),
      Math.ceil((WINOGRAD_PLANES * Math.ceil(tiles / WINOGRAD_MUL_TILES) * co4) / WORKGROUP_SIZE),
      `conv2d_hwc4_wgMul ${shape}`,
    ),
    makeHandle(
      pipelines.output,
      'conv2d_hwc4_wgOut',
      cachedBindGroup(root, convLayout, { x: planesOut, w: weight, bias, res, out, dims }),
      Math.ceil((tiles * co4) / WINOGRAD_PAIRS),
      `conv2d_hwc4_wgOut ${shape}`,
    ),
  ];
}

/** f16 values in one set of coefficient planes for a c-channel h×w map. */
export function winogradPlaneElems(h: number, w: number, c: number): number {
  return WINOGRAD_PLANES * Math.ceil(h / 4) * Math.ceil(w / 4) * Math.ceil(c / 4) * 4;
}

const HWC4_ACTS: readonly number[] = [ACT_CODE.none, ACT_CODE.silu, ACT_CODE.gelu, ACT_CODE.relu];

function checkHwc4Act(cfg: Conv2dHwc4Cfg): void {
  if (!HWC4_ACTS.includes(cfg.hasAct)) {
    throw new Error(
      `conv2d hwc4: no f16 body for activation code ${cfg.hasAct} — this kernel fuses only none/silu/gelu/relu (ACT_CODE)`,
    );
  }
}

export function createConv2dHwc4Pipeline(root: TgpuRoot, cfg: Conv2dHwc4Cfg) {
  checkHwc4Act(cfg);
  // The split-K kernel's entrypoint takes workgroup builtins, the others a
  // global id; the two signatures do not unify in one expression.
  if (cfg.kind === 3) {
    return root.createComputePipeline({ compute: makeConv1x1SmallKernel(cfg) });
  }
  if (cfg.kind === 4) {
    return root.createComputePipeline({ compute: makeConvTiledKernel(cfg) });
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
      Math.ceil(threads / WORKGROUP_SIZE),
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
  if (cfg.kind === 4) {
    return makeHandle(
      pipeline,
      'conv2d_hwc4_tiled',
      bindGroup,
      [Math.ceil((hOut * wOut) / TILED_BN), Math.ceil(co4 / (cfg.bmb ?? TILED_MAX_BLOCKS))],
      `conv2d_hwc4_tiled ${shape}`,
    );
  }
  if (cfg.kind === 2) {
    const threads = co4 * hOut * Math.ceil(wOut / (cfg.dwCols ?? 1));
    return makeHandle(
      pipeline,
      'conv2d_hwc4_dw',
      bindGroup,
      Math.ceil(threads / WORKGROUP_SIZE),
      `conv2d_hwc4_dw ${shape}`,
    );
  }
  return makeHandle(
    pipeline,
    'conv2d_hwc4',
    bindGroup,
    Math.ceil((co4 * hOut * Math.ceil(wOut / (cfg.cols ?? 1))) / WORKGROUP_SIZE),
    `conv2d_hwc4 ${shape}`,
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
    const hOut = std.intdiv(dims.h + 2 * padding - k, stride) + 1;
    const wOut = std.intdiv(dims.w + 2 * padding - k, stride) + 1;
    const idx = flatIndex(gid);
    if (idx >= dims.c4 * hOut * wOut) {
      return;
    }
    const cB = idx % dims.c4;
    const ox = std.intdiv(idx, dims.c4) % wOut;
    const oy = std.intdiv(idx, dims.c4 * wOut);

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
    const cB = idx % dims.c4;
    const ox = std.intdiv(idx, dims.c4) % wOut;
    const oy = std.intdiv(idx, dims.c4 * wOut);
    moveLayout.$.out[(oy * wOut + ox) * dims.outC4 + dims.outOffB + cB] = d.vec4h(
      moveLayout.$.x[(std.intdiv(oy, scale) * dims.w + std.intdiv(ox, scale)) * dims.c4 + cB]!,
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
    const hOut = std.intdiv(dims.h - k, stride) + 1;
    const wOut = std.intdiv(dims.w - k, stride) + 1;
    const idx = flatIndex(gid);
    if (idx >= dims.c4 * hOut * wOut) {
      return;
    }
    const cB = idx % dims.c4;
    const ox = std.intdiv(idx, dims.c4) % wOut;
    const oy = std.intdiv(idx, dims.c4 * wOut);

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
    const cB = idx % dims.c4;
    const ox = std.intdiv(idx, dims.c4) % wOut;
    const oy = std.intdiv(idx, dims.c4 * wOut);
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
    const cB = idx % dims.c4;
    const ox = std.intdiv(idx, dims.c4) % dims.outW;
    const oy = std.intdiv(idx, dims.c4 * dims.outW);

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
    const c4 = std.intdiv(dims.c + 3, 4);
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
  const px = std.intdiv(idx, dims.nB);
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
    const c4 = std.intdiv(dims.c + 3, 4);
    const idx = flatIndex(gid);
    if (idx >= p * c4) {
      return;
    }
    const px = std.intdiv(idx, c4);
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
    const c4 = std.intdiv(dims.c + 3, 4);
    const idx = flatIndex(gid);
    if (idx >= p * c4) {
      return;
    }
    const px = std.intdiv(idx, c4);
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
