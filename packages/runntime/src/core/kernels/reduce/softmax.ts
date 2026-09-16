import tgpu, { d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import {
  type FloatBuffer,
  flatIndex,
  type KernelHandle,
  makeHandle,
  WORKGROUP_SIZE,
} from '../common.ts';
import { cachedBindGroup, cachedUniform } from '../../gpu/dispatchCache.ts';
import { type Elem, F32_ELEM } from '../elem.ts';

const Dims = d.struct({ rows: d.u32, cols: d.u32 });

function makeSoftmaxVariant(elem: Elem) {
  const layout = tgpu.bindGroupLayout({
    x: { storage: d.arrayOf(elem.scalar), access: 'readonly' },
    out: { storage: d.arrayOf(elem.scalar), access: 'mutable' },
    dims: { uniform: Dims },
  });
  const storeScalar = elem.scalar;
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const rows = layout.$.dims.rows;
    const cols = layout.$.dims.cols;
    const row = flatIndex(gid);
    if (row >= rows) return;
    const base = row * cols;
    let m = d.f32(layout.$.x[base]!);
    for (let j = d.u32(1); j < cols; j++) {
      m = std.max(m, d.f32(layout.$.x[base + j]!));
    }
    let sum = d.f32(0);
    for (let j = d.u32(0); j < cols; j++) {
      sum += std.exp(d.f32(layout.$.x[base + j]!) - m);
    }
    for (let j = d.u32(0); j < cols; j++) {
      layout.$.out[base + j] = storeScalar(std.exp(d.f32(layout.$.x[base + j]!) - m) / sum);
    }
  });
  return { layout, kernel };
}

const variants = new Map<string, ReturnType<typeof makeSoftmaxVariant>>();
export function softmaxVariant(elem: Elem): ReturnType<typeof makeSoftmaxVariant> {
  let v = variants.get(elem.key);
  if (!v) {
    v = makeSoftmaxVariant(elem);
    variants.set(elem.key, v);
  }
  return v;
}

export function createSoftmaxPipeline(root: TgpuRoot, elem: Elem = F32_ELEM) {
  return root.createComputePipeline({ compute: softmaxVariant(elem).kernel });
}

export function softmaxHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createSoftmaxPipeline>,
  rows: number,
  cols: number,
  buffers: { x: FloatBuffer; out: FloatBuffer },
  elem: Elem = F32_ELEM,
): KernelHandle {
  const dims = cachedUniform(root, Dims, { rows, cols });
  const bindGroup = cachedBindGroup(root, softmaxVariant(elem).layout, { ...buffers, dims });
  return makeHandle(pipeline, 'softmax', bindGroup, Math.ceil(rows / WORKGROUP_SIZE));
}
