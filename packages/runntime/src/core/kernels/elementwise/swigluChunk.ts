import tgpu, { d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import {
  type FloatBuffer,
  flatIndex,
  type KernelHandle,
  makeHandle,
  WORKGROUP_SIZE,
} from '../common.ts';
import { type Elem, F32_ELEM } from '../elem.ts';

/** Fused SwiGLU chunk: x[t, 2F] → out[t, F] with
 *      out[r, c] = x[r, c] · silu(x[r, F + c])
 *  — HF fc1(x).chunk(2) order (hidden = first F columns, gate = second),
 *  out = silu(gate) · hidden. Collapses the sliceCols ×2 → silu → mul chain
 *  (4 dispatches) into one. Unclamped — the gpt-oss variant with clamps is
 *  the separate `swiglu` kernel. One thread per output element; `half` = F
 *  rides the uniform, so one pipeline serves every width. */

const Dims = d.struct({ total: d.u32, half: d.u32 });

function makeSwigluChunkVariant(elem: Elem) {
  const layout = tgpu.bindGroupLayout({
    x: { storage: d.arrayOf(elem.scalar), access: 'readonly' }, // [t, 2·half]
    out: { storage: d.arrayOf(elem.scalar), access: 'mutable' }, // [t, half]
    dims: { uniform: Dims },
  });
  const storeScalar = elem.scalar;
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const total = layout.$.dims.total;
    const half = layout.$.dims.half;
    const i = flatIndex(gid);
    if (i >= total) return;
    const row = d.u32(i / half);
    const c = i % half;
    const base = row * 2 * half;
    const hidden = d.f32(layout.$.x[base + c]!);
    const gate = d.f32(layout.$.x[base + half + c]!);
    layout.$.out[i] = storeScalar(hidden * (gate * (1 / (1 + std.exp(-gate)))));
  });
  return { layout, kernel };
}

const variants = new Map<string, ReturnType<typeof makeSwigluChunkVariant>>();
export function swigluChunkVariant(elem: Elem): ReturnType<typeof makeSwigluChunkVariant> {
  let v = variants.get(elem.key);
  if (!v) {
    v = makeSwigluChunkVariant(elem);
    variants.set(elem.key, v);
  }
  return v;
}

export function createSwigluChunkPipeline(root: TgpuRoot, elem: Elem = F32_ELEM) {
  return root.createComputePipeline({ compute: swigluChunkVariant(elem).kernel });
}

export function swigluChunkHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createSwigluChunkPipeline>,
  args: { total: number; half: number },
  buffers: { x: FloatBuffer; out: FloatBuffer },
  elem: Elem = F32_ELEM,
): KernelHandle {
  const dims = root.createBuffer(Dims, { total: args.total, half: args.half }).$usage('uniform');
  const bindGroup = root.createBindGroup(swigluChunkVariant(elem).layout, { ...buffers, dims });
  return makeHandle(pipeline, 'swigluChunk', bindGroup, Math.ceil(args.total / WORKGROUP_SIZE));
}
