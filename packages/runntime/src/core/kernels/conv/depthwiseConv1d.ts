/** A causal depthwise 1-D convolution, time-major, in one dispatch, with an
 *  optional GLU in front of it:
 *
 *      g[t, c]   = x[t, c]                                  plain
 *      g[t, c]   = clamp(x[t, c]) · σ(clamp(x[t, C + c]))   with the GLU
 *      out[t, c] = Σ_k w[k, c] · g[t − K + 1 + k, c]
 *
 *  x is [T, C], or [T, 2C] with the GLU, and w tap-major [K, C]. Frames
 *  before the start count as 0. One thread per (frame, 4 channels); with the
 *  GLU each tap recomputes its g. Sums in f32. */

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
import type { Elem } from '../elem.ts';

const Dims = d.struct({ rows: d.u32, words: d.u32, kernel: d.u32, lo: d.f32, hi: d.f32 });

const makeLayout = (elem: Elem) =>
  tgpu.bindGroupLayout({
    x: { storage: d.arrayOf(elem.vec4), access: 'readonly' },
    w: { storage: d.arrayOf(elem.vec4), access: 'readonly' },
    out: { storage: d.arrayOf(elem.vec4), access: 'mutable' },
    dims: { uniform: Dims },
  });

function makeVariant(elem: Elem, glu: boolean) {
  const layout = makeLayout(elem);
  const store = elem.vec4;
  // With the GLU a frame of x holds its values, then their gates.
  const frameWords = glu ? 2 : 1;
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
        const src = (t + k + 1 - D.kernel) * frameWords * D.words + c;
        let g = d.vec4f(layout.$.x[src]!);
        if (glu) {
          const b = std.clamp(d.vec4f(layout.$.x[src + D.words]!), lo, hi);
          g = std.clamp(g, lo, hi) / (d.vec4f(1) + std.exp(-b));
        }
        acc += d.vec4f(layout.$.w[k * D.words + c]!) * g;
      }
    }
    layout.$.out[idx] = store(acc);
  });
  return { layout, kernel };
}

const variants = new Map<string, ReturnType<typeof makeVariant>>();
function variantFor(elem: Elem, glu: boolean): ReturnType<typeof makeVariant> {
  const key = `${elem.key}:${glu}`;
  let v = variants.get(key);
  if (!v) {
    v = makeVariant(elem, glu);
    variants.set(key, v);
  }
  return v;
}

/** The conv's pipeline over `elem` storage, with the GLU in front when `glu`. */
export function createDepthwiseConv1dPipeline(root: TgpuRoot, elem: Elem, glu: boolean) {
  return root.createComputePipeline({ compute: variantFor(elem, glu).kernel });
}

/** One dispatch over `rows` frames of `channels` channels, a multiple of 4.
 *  `glu` holds the GLU's clamp bounds, for a pipeline made with the GLU. */
export function depthwiseConv1dHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createDepthwiseConv1dPipeline>,
  args: { rows: number; channels: number; kernel: number; glu?: { lo: number; hi: number } },
  buffers: { x: FloatBuffer; w: FloatBuffer; out: FloatBuffer },
  elem: Elem,
): KernelHandle {
  const { rows, kernel, glu } = args;
  const words = args.channels / 4;
  const dims = cachedUniform(root, Dims, {
    rows,
    words,
    kernel,
    lo: glu?.lo ?? 0,
    hi: glu?.hi ?? 0,
  });
  const layout = variantFor(elem, glu !== undefined).layout;
  const bindGroup = cachedBindGroup(root, layout, { ...buffers, dims });
  return makeHandle(
    pipeline,
    glu ? 'gluConv1d' : 'depthwiseConv1d',
    bindGroup,
    Math.ceil((rows * words) / WORKGROUP_SIZE),
  );
}
