/** GLU then a causal depthwise 1-D convolution, time-major, in one dispatch:
 *
 *      g[t, c]   = clamp(x[t, c]) · σ(clamp(x[t, C + c]))
 *      out[t, c] = Σ_k w[k, c] · g[t − K + 1 + k, c]
 *
 *  x is [T, 2C], w tap-major [K, C]. Frames before the start count as 0.
 *  One thread per (frame, 4 channels); each tap recomputes its g. */

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

const Dims = d.struct({ rows: d.u32, words: d.u32, kernel: d.u32, lo: d.f32, hi: d.f32 });

const makeLayout = (elem: Elem) =>
  tgpu.bindGroupLayout({
    x: { storage: d.arrayOf(elem.vec4), access: 'readonly' },
    w: { storage: d.arrayOf(elem.vec4), access: 'readonly' },
    out: { storage: d.arrayOf(elem.vec4), access: 'mutable' },
    dims: { uniform: Dims },
  });

function makeVariant(elem: Elem) {
  const layout = makeLayout(elem);
  const store = elem.vec4;
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const D = layout.$.dims;
    const idx = flatIndex(gid);
    if (idx >= D.rows * D.words) return;
    const t = std.intdiv(idx, D.words);
    const c = idx % D.words;
    const lo = d.vec4f(D.lo);
    const hi = d.vec4f(D.hi);
    let acc = d.vec4f();
    for (let k = d.u32(0); k < D.kernel; k++) {
      // The guard comes first so u32 never underflows.
      if (t + k + 1 >= D.kernel) {
        const src = (t + k + 1 - D.kernel) * 2 * D.words + c;
        const a = std.clamp(d.vec4f(layout.$.x[src]!), lo, hi);
        const b = std.clamp(d.vec4f(layout.$.x[src + D.words]!), lo, hi);
        const g = a / (d.vec4f(1) + std.exp(-b));
        acc += d.vec4f(layout.$.w[k * D.words + c]!) * g;
      }
    }
    layout.$.out[idx] = store(acc);
  });
  return { layout, kernel };
}

const variants = new Map<string, ReturnType<typeof makeVariant>>();
function variantFor(elem: Elem): ReturnType<typeof makeVariant> {
  let v = variants.get(elem.key);
  if (!v) {
    v = makeVariant(elem);
    variants.set(elem.key, v);
  }
  return v;
}

export function createGluConv1dPipeline(root: TgpuRoot, elem: Elem = F32_ELEM) {
  return root.createComputePipeline({ compute: variantFor(elem).kernel });
}

export function gluConv1dHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createGluConv1dPipeline>,
  args: { rows: number; cols: number; kernel: number; lo: number; hi: number },
  buffers: { x: FloatBuffer; w: FloatBuffer; out: FloatBuffer },
  elem: Elem = F32_ELEM,
): KernelHandle {
  const { rows, kernel, lo, hi } = args;
  const words = args.cols / 4;
  const dims = cachedUniform(root, Dims, { rows, words, kernel, lo, hi });
  const bindGroup = cachedBindGroup(root, variantFor(elem).layout, { ...buffers, dims });
  return makeHandle(pipeline, 'gluConv1d', bindGroup, Math.ceil((rows * words) / WORKGROUP_SIZE));
}
