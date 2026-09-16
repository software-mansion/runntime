import tgpu, { d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import {
  type F32Buffer,
  GRID_Y_STRIDE,
  type KernelHandle,
  makeHandle,
  WORKGROUP_SIZE,
} from './common.ts';
import { cachedBindGroup, cachedUniform } from '../gpu/dispatchCache.ts';

/** Cross-scan directions: row and column, each forward and reverse. */
export const SSM_DIRS = 4;
export const SSM_STATE = 8;

const sourcePixel = (direction: number, position: number, width: number, height: number) => {
  'use gpu';
  let t = position;
  if (direction >= 2) {
    t = width * height - 1 - t;
  }
  if (direction === 1 || direction === 3) {
    return (t % height) * width + d.u32(t / height);
  }
  return t;
};

const Dims = d.struct({ c: d.u32, h: d.u32, w: d.u32, rank: d.u32 });

export const ssmScanProjectLayout = tgpu.bindGroupLayout({
  src: { storage: d.arrayOf(d.f32), access: 'readonly' }, // [C, H, W]
  xw: { storage: d.arrayOf(d.f32), access: 'readonly' }, // [4, rank+16, C]
  dw: { storage: d.arrayOf(d.f32), access: 'readonly' }, // [4, C, rank]
  out: { storage: d.arrayOf(d.f32), access: 'mutable' }, // [4C+64+4·rank, P]
  dims: { uniform: Dims },
});

export const ssmScanProjectXKernel = tgpu.computeFn({
  in: { gid: d.builtin.globalInvocationId },
  workgroupSize: [WORKGROUP_SIZE],
})(({ gid }) => {
  'use gpu';
  const D = ssmScanProjectLayout.$.dims;
  const P = D.h * D.w;
  const rows = D.rank + 2 * SSM_STATE;
  // Fold back the executor's gid.y split.
  const thread = gid.x + gid.y * GRID_Y_STRIDE;
  if (thread >= SSM_DIRS * rows * P) {
    return;
  }
  const pos = thread % P;
  const row = d.u32(thread / P) % rows;
  const dir = d.u32(thread / (P * rows));
  const pixel = sourcePixel(dir, pos, D.w, D.h);

  let acc = d.f32(0);
  const wBase = (dir * rows + row) * D.c;
  // Not comptime, so std.range cannot replace this loop.
  for (let ch = d.u32(0); ch < D.c; ch++) {
    acc += ssmScanProjectLayout.$.xw[wBase + ch]! * ssmScanProjectLayout.$.src[ch * P + pixel]!;
  }
  const bcBase = SSM_DIRS * D.c;
  if (row < D.rank) {
    ssmScanProjectLayout.$.out[(bcBase + 2 * SSM_DIRS * SSM_STATE + dir * D.rank + row) * P + pos] =
      acc;
  } else if (row < D.rank + SSM_STATE) {
    ssmScanProjectLayout.$.out[(bcBase + dir * SSM_STATE + (row - D.rank)) * P + pos] = acc;
  } else {
    ssmScanProjectLayout.$.out[
      (bcBase + SSM_DIRS * SSM_STATE + dir * SSM_STATE + (row - D.rank - SSM_STATE)) * P + pos
    ] = acc;
  }
});

export const ssmScanProjectDtKernel = tgpu.computeFn({
  in: { gid: d.builtin.globalInvocationId },
  workgroupSize: [WORKGROUP_SIZE],
})(({ gid }) => {
  'use gpu';
  const D = ssmScanProjectLayout.$.dims;
  const P = D.h * D.w;
  const thread = gid.x + gid.y * GRID_Y_STRIDE;
  if (thread >= SSM_DIRS * D.c * P) {
    return;
  }
  const pos = thread % P;
  const co = d.u32(thread / P) % D.c;
  const dir = d.u32(thread / (P * D.c));
  const dtBase = SSM_DIRS * D.c + 2 * SSM_DIRS * SSM_STATE + dir * D.rank;

  let acc = d.f32(0);
  const wBase = (dir * D.c + co) * D.rank;
  for (let r = d.u32(0); r < D.rank; r++) {
    acc +=
      ssmScanProjectLayout.$.dw[wBase + r]! * ssmScanProjectLayout.$.out[(dtBase + r) * P + pos]!;
  }
  ssmScanProjectLayout.$.out[(dir * D.c + co) * P + pos] = acc;
});

const ScanDims = d.struct({ c: d.u32, h: d.u32, w: d.u32 });

export const ssmSelectiveScanLayout = tgpu.bindGroupLayout({
  src: { storage: d.arrayOf(d.f32), access: 'readonly' }, // [C, H, W]
  proj: { storage: d.arrayOf(d.f32), access: 'readonly' }, // scan-project output; reads rows [0, 4C+64)
  a: { storage: d.arrayOf(d.f32), access: 'readonly' }, // [4C, 8]
  dSkip: { storage: d.arrayOf(d.f32), access: 'readonly' }, // [4C]
  deltaBias: { storage: d.arrayOf(d.f32), access: 'readonly' }, // [4C]
  out: { storage: d.arrayOf(d.f32), access: 'mutable' }, // [4C, P] traversal order
  dims: { uniform: ScanDims },
});

const scanState = tgpu.privateVar(d.arrayOf(d.f32, SSM_STATE));

const softplus = (value: number) => {
  'use gpu';
  if (value > 20) {
    return value;
  }
  if (value < -20) {
    return std.exp(value);
  }
  return std.log(1 + std.exp(value));
};

export const ssmSelectiveScanKernel = tgpu.computeFn({
  in: { gid: d.builtin.globalInvocationId },
  workgroupSize: [WORKGROUP_SIZE],
})(({ gid }) => {
  'use gpu';
  const D = ssmSelectiveScanLayout.$.dims;
  const P = D.h * D.w;
  const seq = gid.x + gid.y * GRID_Y_STRIDE; // = dir·C + channel
  if (seq >= SSM_DIRS * D.c) {
    return;
  }
  const dir = d.u32(seq / D.c);
  const ch = seq % D.c;
  const bBase = SSM_DIRS * D.c + dir * SSM_STATE;
  const cBase = bBase + SSM_DIRS * SSM_STATE;

  // Unrolled, so the recurrence indexes scanState with literals and the
  // compiler sees independent FMA chains.
  for (const st of tgpu.unroll(std.range(SSM_STATE))) {
    scanState.$[st] = d.f32(0);
  }

  for (let pos = d.u32(0); pos < P; pos++) {
    const pixel = sourcePixel(dir, pos, D.w, D.h);
    const u = ssmSelectiveScanLayout.$.src[ch * P + pixel]!;
    const delta = softplus(
      ssmSelectiveScanLayout.$.proj[seq * P + pos]! + ssmSelectiveScanLayout.$.deltaBias[seq]!,
    );
    let output = d.f32(0);
    for (const st of tgpu.unroll(std.range(SSM_STATE))) {
      const aExp = std.exp(delta * ssmSelectiveScanLayout.$.a[seq * SSM_STATE + st]!);
      const next = std.fma(
        aExp,
        scanState.$[st]!,
        delta * ssmSelectiveScanLayout.$.proj[(bBase + st) * P + pos]! * u,
      );
      scanState.$[st] = next;
      output = std.fma(ssmSelectiveScanLayout.$.proj[(cBase + st) * P + pos]!, next, output);
    }
    ssmSelectiveScanLayout.$.out[seq * P + pos] = std.fma(
      ssmSelectiveScanLayout.$.dSkip[seq]!,
      u,
      output,
    );
  }
});

export const ssmScanMergeLayout = tgpu.bindGroupLayout({
  directional: { storage: d.arrayOf(d.f32), access: 'readonly' }, // [4C, P]
  out: { storage: d.arrayOf(d.f32), access: 'mutable' }, // [C, H, W]
  dims: { uniform: ScanDims },
});

export const ssmScanMergeKernel = tgpu.computeFn({
  in: { gid: d.builtin.globalInvocationId },
  workgroupSize: [WORKGROUP_SIZE],
})(({ gid }) => {
  'use gpu';
  const D = ssmScanMergeLayout.$.dims;
  const P = D.h * D.w;
  const thread = gid.x + gid.y * GRID_Y_STRIDE;
  if (thread >= D.c * P) {
    return;
  }
  const ch = d.u32(thread / P);
  const pixel = thread % P;
  // Unrolled, so `dir` is comptime and each direction keeps only its own
  // address arithmetic.
  let sum = d.f32(0);
  for (const dir of tgpu.unroll(std.range(SSM_DIRS))) {
    let position = pixel;
    if (dir === 1 || dir === 3) {
      position = (pixel % D.w) * D.h + d.u32(pixel / D.w);
    }
    if (dir >= 2) {
      position = P - 1 - position;
    }
    sum += ssmScanMergeLayout.$.directional[(dir * D.c + ch) * P + position]!;
  }
  ssmScanMergeLayout.$.out[thread] = sum;
});

export function createSsmScanProjectXPipeline(root: TgpuRoot) {
  return root.createComputePipeline({ compute: ssmScanProjectXKernel });
}
export function createSsmScanProjectDtPipeline(root: TgpuRoot) {
  return root.createComputePipeline({ compute: ssmScanProjectDtKernel });
}
export function createSsmSelectiveScanPipeline(root: TgpuRoot) {
  return root.createComputePipeline({ compute: ssmSelectiveScanKernel });
}
export function createSsmScanMergePipeline(root: TgpuRoot) {
  return root.createComputePipeline({ compute: ssmScanMergeKernel });
}

export function ssmScanProjectHandles(
  root: TgpuRoot,
  xPipeline: ReturnType<typeof createSsmScanProjectXPipeline>,
  dtPipeline: ReturnType<typeof createSsmScanProjectDtPipeline>,
  args: { c: number; h: number; w: number; rank: number },
  buffers: { src: F32Buffer; xw: F32Buffer; dw: F32Buffer; out: F32Buffer },
): KernelHandle[] {
  const dims = cachedUniform(root, Dims, args);
  const bindGroup = cachedBindGroup(root, ssmScanProjectLayout, { ...buffers, dims });
  const p = args.h * args.w;
  const xThreads = SSM_DIRS * (args.rank + 2 * SSM_STATE) * p;
  const dtThreads = SSM_DIRS * args.c * p;
  return [
    makeHandle(xPipeline, 'ssmScanProjectX', bindGroup, Math.ceil(xThreads / WORKGROUP_SIZE)),
    makeHandle(dtPipeline, 'ssmScanProjectDt', bindGroup, Math.ceil(dtThreads / WORKGROUP_SIZE)),
  ];
}

export function ssmSelectiveScanHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createSsmSelectiveScanPipeline>,
  args: { c: number; h: number; w: number },
  buffers: {
    src: F32Buffer;
    proj: F32Buffer;
    a: F32Buffer;
    dSkip: F32Buffer;
    deltaBias: F32Buffer;
    out: F32Buffer;
  },
): KernelHandle {
  const dims = cachedUniform(root, ScanDims, args);
  const bindGroup = cachedBindGroup(root, ssmSelectiveScanLayout, { ...buffers, dims });
  const threads = SSM_DIRS * args.c;
  return makeHandle(pipeline, 'ssmSelectiveScan', bindGroup, Math.ceil(threads / WORKGROUP_SIZE));
}

export function ssmScanMergeHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createSsmScanMergePipeline>,
  args: { c: number; h: number; w: number },
  buffers: { directional: F32Buffer; out: F32Buffer },
): KernelHandle {
  const dims = cachedUniform(root, ScanDims, args);
  const bindGroup = cachedBindGroup(root, ssmScanMergeLayout, { ...buffers, dims });
  const threads = args.c * args.h * args.w;
  return makeHandle(pipeline, 'ssmScanMerge', bindGroup, Math.ceil(threads / WORKGROUP_SIZE));
}
