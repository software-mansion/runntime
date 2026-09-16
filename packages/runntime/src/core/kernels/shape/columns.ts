import tgpu, { d } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import {
  type FloatBuffer,
  type KernelHandle,
  flatIndex,
  makeHandle,
  WORKGROUP_SIZE,
} from '../common.ts';
import { cachedBindGroup, cachedUniform } from '../../gpu/dispatchCache.ts';
import { type Elem, F32_ELEM } from '../elem.ts';

const SliceMeta = d.struct({ total: d.u32, srcCols: d.u32, start: d.u32, outCols: d.u32 });

const ConcatMeta = d.struct({ total: d.u32, aCols: d.u32, bCols: d.u32 });

function makeVariant(elem: Elem) {
  const sliceLayout = tgpu.bindGroupLayout({
    x: { storage: d.arrayOf(elem.scalar), access: 'readonly' },
    out: { storage: d.arrayOf(elem.scalar), access: 'mutable' },
    meta: { uniform: SliceMeta },
  });

  const sliceKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const meta = sliceLayout.$.meta;
    const i = flatIndex(gid);
    if (i >= meta.total) {
      return;
    }
    const r = d.u32(i / meta.outCols); // explicit integer division (codebase convention)
    const c = i % meta.outCols;
    sliceLayout.$.out[i] = sliceLayout.$.x[r * meta.srcCols + meta.start + c]!;
  });

  const concatLayout = tgpu.bindGroupLayout({
    a: { storage: d.arrayOf(elem.scalar), access: 'readonly' },
    b: { storage: d.arrayOf(elem.scalar), access: 'readonly' },
    out: { storage: d.arrayOf(elem.scalar), access: 'mutable' },
    meta: { uniform: ConcatMeta },
  });

  const concatKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const meta = concatLayout.$.meta;
    const i = flatIndex(gid);
    if (i >= meta.total) {
      return;
    }
    const outCols = meta.aCols + meta.bCols;
    const r = d.u32(i / outCols); // explicit integer division (codebase convention)
    const c = i % outCols;
    if (c < meta.aCols) {
      concatLayout.$.out[i] = concatLayout.$.a[r * meta.aCols + c]!;
    } else {
      concatLayout.$.out[i] = concatLayout.$.b[r * meta.bCols + (c - meta.aCols)]!;
    }
  });

  return { sliceLayout, sliceKernel, concatLayout, concatKernel };
}

const variants = new Map<string, ReturnType<typeof makeVariant>>();
export function columnsVariant(elem: Elem): ReturnType<typeof makeVariant> {
  let v = variants.get(elem.key);
  if (!v) {
    v = makeVariant(elem);
    variants.set(elem.key, v);
  }
  return v;
}

export const sliceColsLayout = columnsVariant(F32_ELEM).sliceLayout;
export const concatColsLayout = columnsVariant(F32_ELEM).concatLayout;

export function createSliceColsPipeline(root: TgpuRoot, elem: Elem = F32_ELEM) {
  return root.createComputePipeline({ compute: columnsVariant(elem).sliceKernel });
}
export function createConcatColsPipeline(root: TgpuRoot, elem: Elem = F32_ELEM) {
  return root.createComputePipeline({ compute: columnsVariant(elem).concatKernel });
}

export function sliceColsHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createSliceColsPipeline>,
  args: { total: number; srcCols: number; start: number; outCols: number },
  buffers: { x: FloatBuffer; out: FloatBuffer },
  elem: Elem = F32_ELEM,
): KernelHandle {
  const meta = cachedUniform(root, SliceMeta, args);
  const bindGroup = cachedBindGroup(root, columnsVariant(elem).sliceLayout, { ...buffers, meta });
  return makeHandle(pipeline, 'sliceCols', bindGroup, Math.ceil(args.total / WORKGROUP_SIZE));
}

export function concatColsHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createConcatColsPipeline>,
  args: { total: number; aCols: number; bCols: number },
  buffers: { a: FloatBuffer; b: FloatBuffer; out: FloatBuffer },
  elem: Elem = F32_ELEM,
): KernelHandle {
  const meta = cachedUniform(root, ConcatMeta, args);
  const bindGroup = cachedBindGroup(root, columnsVariant(elem).concatLayout, { ...buffers, meta });
  return makeHandle(pipeline, 'concatCols', bindGroup, Math.ceil(args.total / WORKGROUP_SIZE));
}
