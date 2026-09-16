import tgpu, { d } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import {
  type F32Buffer,
  type FloatBuffer,
  type KernelHandle,
  flatIndex,
  makeHandle,
  type U32Buffer,
  WORKGROUP_SIZE,
} from '../common.ts';
import { cachedBindGroup, cachedUniform } from '../../gpu/dispatchCache.ts';
import { type Elem, F32_ELEM } from '../elem.ts';

/** sliceRows. Rows are contiguous, so this is a flat copy offset by start·cols
 *  on the read side and outBase on the write side. */
const SliceMeta = d.struct({ total: d.u32, cols: d.u32, start: d.u32, outBase: d.u32 });

function makeSliceRowsVariant(elem: Elem) {
  const layout = tgpu.bindGroupLayout({
    x: { storage: d.arrayOf(elem.scalar), access: 'readonly' },
    out: { storage: d.arrayOf(elem.scalar), access: 'mutable' },
    meta: { uniform: SliceMeta },
  });
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const meta = layout.$.meta;
    const i = flatIndex(gid);
    if (i >= meta.total) {
      return;
    }
    layout.$.out[meta.outBase + i] = layout.$.x[meta.start * meta.cols + i]!;
  });
  return { layout, kernel };
}

const sliceRowsVariants = new Map<string, ReturnType<typeof makeSliceRowsVariant>>();
export function sliceRowsVariant(elem: Elem): ReturnType<typeof makeSliceRowsVariant> {
  let v = sliceRowsVariants.get(elem.key);
  if (!v) {
    v = makeSliceRowsVariant(elem);
    sliceRowsVariants.set(elem.key, v);
  }
  return v;
}

export const sliceRowsLayout = sliceRowsVariant(F32_ELEM).layout;

const ConcatMeta = d.struct({ total: d.u32, aTotal: d.u32 });

function makeConcatRowsVariant(elem: Elem) {
  const layout = tgpu.bindGroupLayout({
    a: { storage: d.arrayOf(elem.scalar), access: 'readonly' },
    b: { storage: d.arrayOf(elem.scalar), access: 'readonly' },
    out: { storage: d.arrayOf(elem.scalar), access: 'mutable' },
    meta: { uniform: ConcatMeta },
  });
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const meta = layout.$.meta;
    const i = flatIndex(gid);
    if (i >= meta.total) {
      return;
    }
    if (i < meta.aTotal) {
      layout.$.out[i] = layout.$.a[i]!;
    } else {
      layout.$.out[i] = layout.$.b[i - meta.aTotal]!;
    }
  });
  return { layout, kernel };
}

const concatRowsVariants = new Map<string, ReturnType<typeof makeConcatRowsVariant>>();
export function concatRowsVariant(elem: Elem): ReturnType<typeof makeConcatRowsVariant> {
  let v = concatRowsVariants.get(elem.key);
  if (!v) {
    v = makeConcatRowsVariant(elem);
    concatRowsVariants.set(elem.key, v);
  }
  return v;
}

const GatherFromMeta = d.struct({ total: d.u32, cols: d.u32 });

function makeGatherFromVariant(elem: Elem) {
  const layout = tgpu.bindGroupLayout({
    x: { storage: d.arrayOf(elem.scalar), access: 'readonly' },
    idx: { storage: d.arrayOf(d.f32), access: 'readonly' },
    out: { storage: d.arrayOf(elem.scalar), access: 'mutable' },
    meta: { uniform: GatherFromMeta },
  });
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const meta = layout.$.meta;
    const i = flatIndex(gid);
    if (i >= meta.total) {
      return;
    }
    const r = d.u32(i / meta.cols); // explicit integer division (codebase convention)
    const c = i % meta.cols;
    const srcRow = d.u32(layout.$.idx[r]!);
    layout.$.out[i] = layout.$.x[srcRow * meta.cols + c]!;
  });
  return { layout, kernel };
}

const gatherFromVariants = new Map<string, ReturnType<typeof makeGatherFromVariant>>();
export function gatherRowsFromVariant(elem: Elem): ReturnType<typeof makeGatherFromVariant> {
  let v = gatherFromVariants.get(elem.key);
  if (!v) {
    v = makeGatherFromVariant(elem);
    gatherFromVariants.set(elem.key, v);
  }
  return v;
}

const WriteMeta = d.struct({ total: d.u32, offset: d.u32 });

function makeWriteRowsVariant(elem: Elem) {
  const layout = tgpu.bindGroupLayout({
    src: { storage: d.arrayOf(elem.scalar), access: 'readonly' },
    dst: { storage: d.arrayOf(elem.scalar), access: 'mutable' },
    meta: { uniform: WriteMeta },
  });
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const meta = layout.$.meta;
    const i = flatIndex(gid);
    if (i >= meta.total) {
      return;
    }
    layout.$.dst[meta.offset + i] = layout.$.src[i]!;
  });
  return { layout, kernel };
}

const writeRowsVariants = new Map<string, ReturnType<typeof makeWriteRowsVariant>>();
export function writeRowsVariant(elem: Elem): ReturnType<typeof makeWriteRowsVariant> {
  let v = writeRowsVariants.get(elem.key);
  if (!v) {
    v = makeWriteRowsVariant(elem);
    writeRowsVariants.set(elem.key, v);
  }
  return v;
}

const GatherMeta = d.struct({ total: d.u32, cols: d.u32 });

function makeGatherVariant(elem: Elem) {
  const layout = tgpu.bindGroupLayout({
    x: { storage: d.arrayOf(elem.scalar), access: 'readonly' },
    idx: { storage: d.arrayOf(d.u32), access: 'readonly' },
    out: { storage: d.arrayOf(elem.scalar), access: 'mutable' },
    meta: { uniform: GatherMeta },
  });
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const meta = layout.$.meta;
    const i = flatIndex(gid);
    if (i >= meta.total) {
      return;
    }
    const r = d.u32(i / meta.cols); // explicit integer division (codebase convention)
    const c = i % meta.cols;
    layout.$.out[i] = layout.$.x[layout.$.idx[r]! * meta.cols + c]!;
  });
  return { layout, kernel };
}

const gatherVariants = new Map<string, ReturnType<typeof makeGatherVariant>>();
export function gatherRowsVariant(elem: Elem): ReturnType<typeof makeGatherVariant> {
  let v = gatherVariants.get(elem.key);
  if (!v) {
    v = makeGatherVariant(elem);
    gatherVariants.set(elem.key, v);
  }
  return v;
}

export const gatherRowsLayout = gatherRowsVariant(F32_ELEM).layout;

export function createSliceRowsPipeline(root: TgpuRoot, elem: Elem = F32_ELEM) {
  return root.createComputePipeline({ compute: sliceRowsVariant(elem).kernel });
}
export function createConcatRowsPipeline(root: TgpuRoot, elem: Elem = F32_ELEM) {
  return root.createComputePipeline({ compute: concatRowsVariant(elem).kernel });
}
export function createGatherRowsPipeline(root: TgpuRoot, elem: Elem = F32_ELEM) {
  return root.createComputePipeline({ compute: gatherRowsVariant(elem).kernel });
}
export function createWriteRowsPipeline(root: TgpuRoot, elem: Elem = F32_ELEM) {
  return root.createComputePipeline({ compute: writeRowsVariant(elem).kernel });
}
export function createGatherRowsFromPipeline(root: TgpuRoot, elem: Elem = F32_ELEM) {
  return root.createComputePipeline({ compute: gatherRowsFromVariant(elem).kernel });
}
export function sliceRowsHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createSliceRowsPipeline>,
  args: { total: number; cols: number; start: number; outBase?: number },
  buffers: { x: FloatBuffer; out: FloatBuffer },
  elem: Elem = F32_ELEM,
): KernelHandle {
  const meta = cachedUniform(root, SliceMeta, { outBase: 0, ...args });
  const bindGroup = cachedBindGroup(root, sliceRowsVariant(elem).layout, { ...buffers, meta });
  return makeHandle(pipeline, 'sliceRows', bindGroup, Math.ceil(args.total / WORKGROUP_SIZE));
}

export function concatRowsHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createConcatRowsPipeline>,
  args: { total: number; aTotal: number },
  buffers: { a: FloatBuffer; b: FloatBuffer; out: FloatBuffer },
  elem: Elem = F32_ELEM,
): KernelHandle {
  const meta = cachedUniform(root, ConcatMeta, args);
  const bindGroup = cachedBindGroup(root, concatRowsVariant(elem).layout, { ...buffers, meta });
  return makeHandle(pipeline, 'concatRows', bindGroup, Math.ceil(args.total / WORKGROUP_SIZE));
}

export function gatherRowsFromHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createGatherRowsFromPipeline>,
  args: { total: number; cols: number },
  buffers: { x: FloatBuffer; idx: F32Buffer; out: FloatBuffer },
  elem: Elem = F32_ELEM,
): KernelHandle {
  const meta = root.createBuffer(GatherFromMeta, args).$usage('uniform');
  const bindGroup = root.createBindGroup(gatherRowsFromVariant(elem).layout, { ...buffers, meta });
  return makeHandle(pipeline, 'gatherRowsFrom', bindGroup, Math.ceil(args.total / WORKGROUP_SIZE));
}

export function writeRowsHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createWriteRowsPipeline>,
  args: { total: number; offset: number },
  buffers: { src: FloatBuffer; dst: FloatBuffer },
  elem: Elem = F32_ELEM,
): KernelHandle {
  const meta = root.createBuffer(WriteMeta, args).$usage('uniform');
  const bindGroup = root.createBindGroup(writeRowsVariant(elem).layout, { ...buffers, meta });
  return makeHandle(pipeline, 'writeRows', bindGroup, Math.ceil(args.total / WORKGROUP_SIZE));
}

export function gatherRowsHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createGatherRowsPipeline>,
  args: { total: number; cols: number },
  idx: U32Buffer,
  buffers: { x: FloatBuffer; out: FloatBuffer },
  elem: Elem = F32_ELEM,
): KernelHandle {
  const meta = cachedUniform(root, GatherMeta, args);
  // `idx` is a fresh upload each eval, so an identity-keyed cache would never
  // hit and would grow one dead entry per call.
  const bindGroup = root.createBindGroup(gatherRowsVariant(elem).layout, {
    ...buffers,
    idx,
    meta: meta as never,
  });
  return makeHandle(pipeline, 'gatherRows', bindGroup, Math.ceil(args.total / WORKGROUP_SIZE));
}
