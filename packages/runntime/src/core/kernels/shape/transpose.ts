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

/** INPUT dims; out is [cols, rows]. */
const Dims = d.struct({ rows: d.u32, cols: d.u32 });

function makeTransposeVariant(elem: Elem) {
  const layout = tgpu.bindGroupLayout({
    x: { storage: d.arrayOf(elem.scalar), access: 'readonly' },
    out: { storage: d.arrayOf(elem.scalar), access: 'mutable' },
    dims: { uniform: Dims },
  });
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const rows = layout.$.dims.rows;
    const cols = layout.$.dims.cols;
    const i = flatIndex(gid);
    if (i >= rows * cols) {
      return;
    }
    const r = d.u32(i / cols); // explicit integer division (codebase convention)
    const c = i % cols;
    layout.$.out[c * rows + r] = layout.$.x[i]!;
  });
  return { layout, kernel };
}

const variants = new Map<string, ReturnType<typeof makeTransposeVariant>>();
export function transposeVariant(elem: Elem): ReturnType<typeof makeTransposeVariant> {
  let v = variants.get(elem.key);
  if (!v) {
    v = makeTransposeVariant(elem);
    variants.set(elem.key, v);
  }
  return v;
}

export function createTransposePipeline(root: TgpuRoot, elem: Elem = F32_ELEM) {
  return root.createComputePipeline({ compute: transposeVariant(elem).kernel });
}

export function transposeHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createTransposePipeline>,
  rows: number,
  cols: number,
  buffers: { x: FloatBuffer; out: FloatBuffer },
  elem: Elem = F32_ELEM,
): KernelHandle {
  const dims = cachedUniform(root, Dims, { rows, cols });
  const bindGroup = cachedBindGroup(root, transposeVariant(elem).layout, { ...buffers, dims });
  return makeHandle(pipeline, 'transpose', bindGroup, Math.ceil((rows * cols) / WORKGROUP_SIZE));
}
