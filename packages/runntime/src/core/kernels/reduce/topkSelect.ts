import tgpu, { d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import {
  flatIndex,
  type FloatBuffer,
  type KernelHandle,
  makeHandle,
  WORKGROUP_SIZE,
} from '../common.ts';
import { cachedBindGroup, cachedUniform } from '../../gpu/dispatchCache.ts';
import { type Elem, F32_ELEM } from '../elem.ts';

/** One thread per token: descending top-4 of `experts` logits (strict `>` on
 *  entry and bubble → lowest index wins ties, matching torch.topk), then
 *  softmax over the four selected logits. Packed output row: [id0..id3 (f32),
 *  w0..w3]. Plain scalar locals for the 8 slots with an unrolled bubble —
 *  dynamically-indexed local arrays crash some mobile Vulkan shader
 *  compilers. */
const Meta = d.struct({ tokens: d.u32, experts: d.u32 });

function makeTopkVariant(elem: Elem) {
  const layout = tgpu.bindGroupLayout({
    logits: { storage: d.arrayOf(elem.scalar), access: 'readonly' },
    out: { storage: d.arrayOf(d.f32), access: 'mutable' },
    meta: { uniform: Meta },
  });
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const meta = layout.$.meta;
    const t = flatIndex(gid);
    if (t >= meta.tokens) {
      return;
    }
    const NEG = d.f32(-3.4e38);
    let v0 = NEG;
    let v1 = NEG;
    let v2 = NEG;
    let v3 = NEG;
    let i0 = d.u32(0);
    let i1 = d.u32(0);
    let i2 = d.u32(0);
    let i3 = d.u32(0);
    for (let e = d.u32(0); e < meta.experts; e++) {
      const v = d.f32(layout.$.logits[t * meta.experts + e]!);
      if (v > v3) {
        v3 = v;
        i3 = e;
        if (v3 > v2) {
          const tv = v2;
          const ti = i2;
          v2 = v3;
          i2 = i3;
          v3 = tv;
          i3 = ti;
        }
        if (v2 > v1) {
          const tv = v1;
          const ti = i1;
          v1 = v2;
          i1 = i2;
          v2 = tv;
          i2 = ti;
        }
        if (v1 > v0) {
          const tv = v0;
          const ti = i0;
          v0 = v1;
          i0 = i1;
          v1 = tv;
          i1 = ti;
        }
      }
    }
    const m = v0; // descending — v0 is the max
    const e0 = std.exp(v0 - m);
    const e1 = std.exp(v1 - m);
    const e2 = std.exp(v2 - m);
    const e3 = std.exp(v3 - m);
    const z = e0 + e1 + e2 + e3;
    const base = t * 8;
    layout.$.out[base] = d.f32(i0);
    layout.$.out[base + 1] = d.f32(i1);
    layout.$.out[base + 2] = d.f32(i2);
    layout.$.out[base + 3] = d.f32(i3);
    layout.$.out[base + 4] = e0 / z;
    layout.$.out[base + 5] = e1 / z;
    layout.$.out[base + 6] = e2 / z;
    layout.$.out[base + 7] = e3 / z;
  });
  return { layout, kernel };
}

const variants = new Map<string, ReturnType<typeof makeTopkVariant>>();
export function topkSelectVariant(elem: Elem): ReturnType<typeof makeTopkVariant> {
  let v = variants.get(elem.key);
  if (!v) {
    v = makeTopkVariant(elem);
    variants.set(elem.key, v);
  }
  return v;
}

export function createTopkSelectPipeline(root: TgpuRoot, elem: Elem = F32_ELEM) {
  return root.createComputePipeline({ compute: topkSelectVariant(elem).kernel });
}

export function topkSelectHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createTopkSelectPipeline>,
  args: { tokens: number; experts: number },
  buffers: { logits: FloatBuffer; out: FloatBuffer },
  elem: Elem = F32_ELEM,
): KernelHandle {
  const meta = cachedUniform(root, Meta, args);
  const bindGroup = cachedBindGroup(root, topkSelectVariant(elem).layout, { ...buffers, meta });
  return makeHandle(pipeline, 'topk', bindGroup, Math.ceil(args.tokens / WORKGROUP_SIZE));
}
