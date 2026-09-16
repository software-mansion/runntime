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
import { gelu } from '../activations.ts';
import { type Elem, F32_ELEM } from '../elem.ts';

/** `a` is the scalar operand, or clamp's lo bound; `b` is clamp's hi bound. */
const Meta = d.struct({ total: d.u32, a: d.f32, b: d.f32 });

function makeUnaryVariant(elem: Elem) {
  const layout = tgpu.bindGroupLayout({
    x: { storage: d.arrayOf(elem.scalar), access: 'readonly' },
    out: { storage: d.arrayOf(elem.scalar), access: 'mutable' },
    meta: { uniform: Meta },
  });
  const store = elem.scalar;
  const shell = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  });

  const rsqrt = shell(({ gid }) => {
    'use gpu';
    const i = flatIndex(gid);
    if (i >= layout.$.meta.total) return;
    layout.$.out[i] = store(std.inverseSqrt(d.f32(layout.$.x[i]!)));
  });

  const addScalar = shell(({ gid }) => {
    'use gpu';
    const i = flatIndex(gid);
    if (i >= layout.$.meta.total) return;
    layout.$.out[i] = store(d.f32(layout.$.x[i]!) + layout.$.meta.a);
  });

  const sigmoid = shell(({ gid }) => {
    'use gpu';
    const i = flatIndex(gid);
    if (i >= layout.$.meta.total) return;
    layout.$.out[i] = store(1 / (1 + std.exp(-d.f32(layout.$.x[i]!))));
  });

  const mulScalar = shell(({ gid }) => {
    'use gpu';
    const i = flatIndex(gid);
    if (i >= layout.$.meta.total) return;
    layout.$.out[i] = store(d.f32(layout.$.x[i]!) * layout.$.meta.a);
  });

  const clampScalar = shell(({ gid }) => {
    'use gpu';
    const i = flatIndex(gid);
    if (i >= layout.$.meta.total) return;
    layout.$.out[i] = store(std.clamp(d.f32(layout.$.x[i]!), layout.$.meta.a, layout.$.meta.b));
  });

  const tanh = shell(({ gid }) => {
    'use gpu';
    const i = flatIndex(gid);
    if (i >= layout.$.meta.total) return;
    layout.$.out[i] = store(std.tanh(d.f32(layout.$.x[i]!)));
  });

  const geluFn = shell(({ gid }) => {
    'use gpu';
    const i = flatIndex(gid);
    if (i >= layout.$.meta.total) return;
    layout.$.out[i] = store(gelu(d.f32(layout.$.x[i]!)));
  });

  const asinh = shell(({ gid }) => {
    'use gpu';
    const i = flatIndex(gid);
    if (i >= layout.$.meta.total) return;
    const x = d.f32(layout.$.x[i]!);
    const ax = std.abs(x);
    layout.$.out[i] = store(std.sign(x) * std.log(ax + std.sqrt(ax * ax + 1)));
  });

  const silu = shell(({ gid }) => {
    'use gpu';
    const i = flatIndex(gid);
    if (i >= layout.$.meta.total) return;
    const x = d.f32(layout.$.x[i]!);
    layout.$.out[i] = store(x / (1 + std.exp(-x)));
  });

  return {
    layout,
    rsqrt,
    addScalar,
    sigmoid,
    mulScalar,
    clampScalar,
    tanh,
    gelu: geluFn,
    asinh,
    silu,
  };
}

const variants = new Map<string, ReturnType<typeof makeUnaryVariant>>();
export function unaryVariant(elem: Elem): ReturnType<typeof makeUnaryVariant> {
  let v = variants.get(elem.key);
  if (!v) {
    v = makeUnaryVariant(elem);
    variants.set(elem.key, v);
  }
  return v;
}

export function createRsqrtPipeline(root: TgpuRoot, elem: Elem = F32_ELEM) {
  return root.createComputePipeline({ compute: unaryVariant(elem).rsqrt });
}
export function createAddScalarPipeline(root: TgpuRoot, elem: Elem = F32_ELEM) {
  return root.createComputePipeline({ compute: unaryVariant(elem).addScalar });
}
export function createSigmoidPipeline(root: TgpuRoot, elem: Elem = F32_ELEM) {
  return root.createComputePipeline({ compute: unaryVariant(elem).sigmoid });
}
export function createMulScalarPipeline(root: TgpuRoot, elem: Elem = F32_ELEM) {
  return root.createComputePipeline({ compute: unaryVariant(elem).mulScalar });
}
export function createClampScalarPipeline(root: TgpuRoot, elem: Elem = F32_ELEM) {
  return root.createComputePipeline({ compute: unaryVariant(elem).clampScalar });
}
export function createTanhPipeline(root: TgpuRoot, elem: Elem = F32_ELEM) {
  return root.createComputePipeline({ compute: unaryVariant(elem).tanh });
}
export function createGeluPipeline(root: TgpuRoot, elem: Elem = F32_ELEM) {
  return root.createComputePipeline({ compute: unaryVariant(elem).gelu });
}
export function createSiluPipeline(root: TgpuRoot, elem: Elem = F32_ELEM) {
  return root.createComputePipeline({ compute: unaryVariant(elem).silu });
}
export function createAsinhPipeline(root: TgpuRoot, elem: Elem = F32_ELEM) {
  return root.createComputePipeline({ compute: unaryVariant(elem).asinh });
}

export function unaryHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createRsqrtPipeline>,
  name: string,
  total: number,
  a: number,
  b: number,
  buffers: { x: FloatBuffer; out: FloatBuffer },
  elem: Elem = F32_ELEM,
): KernelHandle {
  const meta = cachedUniform(root, Meta, { total, a, b });
  const bindGroup = cachedBindGroup(root, unaryVariant(elem).layout, { ...buffers, meta });
  return makeHandle(pipeline, name, bindGroup, Math.ceil(total / WORKGROUP_SIZE));
}
