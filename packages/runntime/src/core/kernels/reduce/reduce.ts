import tgpu, { d } from 'typegpu';
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

function makeMeanVariant(elem: Elem) {
  const layout = tgpu.bindGroupLayout({
    x: { storage: d.arrayOf(elem.scalar), access: 'readonly' },
    out: { storage: d.arrayOf(elem.scalar), access: 'mutable' },
    dims: { uniform: Dims },
  });
  const storeScalar = elem.scalar;
  const shell = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  });
  const kernel = shell(({ gid }) => {
    'use gpu';
    const rows = layout.$.dims.rows;
    const cols = layout.$.dims.cols;
    const row = flatIndex(gid);
    if (row >= rows) return;
    const base = row * cols;
    let sum = d.f32(0);
    for (let j = d.u32(0); j < cols; j++) sum += d.f32(layout.$.x[base + j]!);
    layout.$.out[row] = storeScalar(sum / d.f32(cols));
  });
  return { layout, kernel };
}

function makeMeanSquareVariant(elem: Elem) {
  const layout = tgpu.bindGroupLayout({
    x: { storage: d.arrayOf(elem.scalar), access: 'readonly' },
    out: { storage: d.arrayOf(d.f32), access: 'mutable' },
    dims: { uniform: Dims },
  });
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
    let sum = d.f32(0);
    for (let j = d.u32(0); j < cols; j++) {
      const v = d.f32(layout.$.x[base + j]!);
      sum += v * v;
    }
    layout.$.out[row] = sum / d.f32(cols);
  });
  return { layout, kernel };
}

const squareVariants = new Map<string, ReturnType<typeof makeMeanSquareVariant>>();
export function meanSquareVariant(elem: Elem): ReturnType<typeof makeMeanSquareVariant> {
  let v = squareVariants.get(elem.key);
  if (!v) {
    v = makeMeanSquareVariant(elem);
    squareVariants.set(elem.key, v);
  }
  return v;
}

const variants = new Map<string, ReturnType<typeof makeMeanVariant>>();
export function meanVariant(elem: Elem): ReturnType<typeof makeMeanVariant> {
  let v = variants.get(elem.key);
  if (!v) {
    v = makeMeanVariant(elem);
    variants.set(elem.key, v);
  }
  return v;
}

export function createMeanPipeline(root: TgpuRoot, elem: Elem = F32_ELEM, square = false) {
  return root.createComputePipeline({
    compute: square ? meanSquareVariant(elem).kernel : meanVariant(elem).kernel,
  });
}

export function meanHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createMeanPipeline>,
  rows: number,
  cols: number,
  buffers: { x: FloatBuffer; out: FloatBuffer },
  elem: Elem = F32_ELEM,
  square = false,
): KernelHandle {
  const dims = cachedUniform(root, Dims, { rows, cols });
  const layout = square ? meanSquareVariant(elem).layout : meanVariant(elem).layout;
  const bindGroup = cachedBindGroup(root, layout, { ...buffers, dims });
  return makeHandle(pipeline, 'mean', bindGroup, Math.ceil(rows / WORKGROUP_SIZE));
}
