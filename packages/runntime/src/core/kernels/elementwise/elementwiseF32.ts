import tgpu, { d, std } from 'typegpu';
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

const SWIGLU_LIMIT = 7.0;
const SWIGLU_ALPHA = 1.702;

const Meta = d.struct({ total: d.u32, cols: d.u32, mode: d.u32 });

function makeVariant(elem: Elem) {
  const f16 = elem.key === 'f16';
  const layout = tgpu.bindGroupLayout({
    a: { storage: d.arrayOf(elem.scalar), access: 'readonly' },
    b: { storage: d.arrayOf(elem.scalar), access: 'readonly' },
    out: { storage: d.arrayOf(elem.scalar), access: 'mutable' },
    meta: { uniform: Meta },
  });

  const bIndex = (i: number): number => {
    'use gpu';
    const m = layout.$.meta.mode;
    const cols = layout.$.meta.cols;
    if (m === 1) return i % cols; // row broadcast
    if (m === 2) return d.u32(i / cols); // col broadcast
    if (m === 3) return 0; // scalar broadcast (GPU-computed [1,1])
    return i; // elementwise
  };

  const addKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const i = flatIndex(gid);
    if (i >= layout.$.meta.total) return;
    if (f16) {
      layout.$.out[i] = d.f16(d.f32(layout.$.a[i]!) + d.f32(layout.$.b[bIndex(i)]!));
    } else {
      layout.$.out[i] = layout.$.a[i]! + layout.$.b[bIndex(i)]!;
    }
  });

  const subKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const i = flatIndex(gid);
    if (i >= layout.$.meta.total) return;
    if (f16) {
      layout.$.out[i] = d.f16(d.f32(layout.$.a[i]!) - d.f32(layout.$.b[bIndex(i)]!));
    } else {
      layout.$.out[i] = layout.$.a[i]! - layout.$.b[bIndex(i)]!;
    }
  });

  const mulKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const i = flatIndex(gid);
    if (i >= layout.$.meta.total) return;
    if (f16) {
      layout.$.out[i] = d.f16(d.f32(layout.$.a[i]!) * d.f32(layout.$.b[bIndex(i)]!));
    } else {
      layout.$.out[i] = layout.$.a[i]! * layout.$.b[bIndex(i)]!;
    }
  });

  const swigluKernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const i = flatIndex(gid);
    if (i >= layout.$.meta.total) return;
    if (f16) {
      const g = std.min(d.f32(layout.$.a[i]!), SWIGLU_LIMIT);
      const l = std.clamp(d.f32(layout.$.b[i]!), -SWIGLU_LIMIT, SWIGLU_LIMIT);
      layout.$.out[i] = d.f16(g * (1 / (1 + std.exp(-SWIGLU_ALPHA * g))) * (l + 1));
    } else {
      const g = std.min(layout.$.a[i]!, SWIGLU_LIMIT);
      const l = std.clamp(layout.$.b[i]!, -SWIGLU_LIMIT, SWIGLU_LIMIT);
      layout.$.out[i] = g * (1 / (1 + std.exp(-SWIGLU_ALPHA * g))) * (l + 1);
    }
  });

  return { layout, addKernel, subKernel, mulKernel, swigluKernel };
}

const variants = new Map<string, ReturnType<typeof makeVariant>>();
export function elementwiseVariant(elem: Elem): ReturnType<typeof makeVariant> {
  let v = variants.get(elem.key);
  if (!v) {
    v = makeVariant(elem);
    variants.set(elem.key, v);
  }
  return v;
}

export const elementwiseF32Layout = elementwiseVariant(F32_ELEM).layout;

export function createAddF32Pipeline(root: TgpuRoot, elem: Elem = F32_ELEM) {
  return root.createComputePipeline({ compute: elementwiseVariant(elem).addKernel });
}
export function createSubF32Pipeline(root: TgpuRoot, elem: Elem = F32_ELEM) {
  return root.createComputePipeline({ compute: elementwiseVariant(elem).subKernel });
}
export function createMulF32Pipeline(root: TgpuRoot, elem: Elem = F32_ELEM) {
  return root.createComputePipeline({ compute: elementwiseVariant(elem).mulKernel });
}
export function createSwigluF32Pipeline(root: TgpuRoot, elem: Elem = F32_ELEM) {
  return root.createComputePipeline({ compute: elementwiseVariant(elem).swigluKernel });
}

export function elementwiseF32Handle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createAddF32Pipeline>,
  name: string,
  total: number,
  cols: number,
  mode: number,
  buffers: { a: FloatBuffer; b: FloatBuffer; out: FloatBuffer },
  elem: Elem = F32_ELEM,
): KernelHandle {
  const meta = cachedUniform(root, Meta, { total, cols, mode });
  const bindGroup = cachedBindGroup(root, elementwiseVariant(elem).layout, { ...buffers, meta });
  return makeHandle(pipeline, name, bindGroup, Math.ceil(total / WORKGROUP_SIZE));
}
