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

/** Storage-width conversion: out[i] = to(x[i]). The one kernel whose input
 *  and output element types differ, so it takes two Elems instead of one.
 *
 *  It exists because a few producers are f32 in every variant — topk packs
 *  expert ids next to their weights, argmaxDot returns a row index — and a
 *  halved model still has to multiply those f32 results into f16
 *  activations. Every other op keeps one element type across its bindings. */

const Meta = d.struct({ total: d.u32 });

function makeCastVariant(from: Elem, to: Elem) {
  const layout = tgpu.bindGroupLayout({
    x: { storage: d.arrayOf(from.scalar), access: 'readonly' },
    out: { storage: d.arrayOf(to.scalar), access: 'mutable' },
    meta: { uniform: Meta },
  });
  const store = to.scalar;
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const i = flatIndex(gid);
    if (i >= layout.$.meta.total) return;
    layout.$.out[i] = store(d.f32(layout.$.x[i]!));
  });
  return { layout, kernel };
}

const variants = new Map<string, ReturnType<typeof makeCastVariant>>();
export function castVariant(from: Elem, to: Elem): ReturnType<typeof makeCastVariant> {
  const key = `${from.key}->${to.key}`;
  let v = variants.get(key);
  if (!v) {
    v = makeCastVariant(from, to);
    variants.set(key, v);
  }
  return v;
}

export function createCastPipeline(root: TgpuRoot, from: Elem = F32_ELEM, to: Elem = F32_ELEM) {
  return root.createComputePipeline({ compute: castVariant(from, to).kernel });
}

export function castHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createCastPipeline>,
  total: number,
  buffers: { x: FloatBuffer; out: FloatBuffer },
  from: Elem = F32_ELEM,
  to: Elem = F32_ELEM,
): KernelHandle {
  const meta = cachedUniform(root, Meta, { total });
  const bindGroup = cachedBindGroup(root, castVariant(from, to).layout, { ...buffers, meta });
  return makeHandle(pipeline, 'astype', bindGroup, Math.ceil(total / WORKGROUP_SIZE));
}
