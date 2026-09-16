import tgpu, { d } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import {
  type F32Buffer,
  type FloatBuffer,
  flatIndex,
  type KernelHandle,
  makeHandle,
  WORKGROUP_SIZE,
} from '../common.ts';
import { cachedBindGroup, cachedUniform } from '../../gpu/dispatchCache.ts';
import { type Elem, F32_ELEM } from '../elem.ts';

/** Rotary position embedding over every head section of a [T, W] slab in one
 *  dispatch, one thread per pair:
 *
 *      out[2p]   = x[2p]·cos[t,d] − x[2p+1]·sin[t,d]
 *      out[2p+1] = x[2p+1]·cos[t,d] + x[2p]·sin[t,d]
 *
 *  cos and sin are pair-duplicated [T, headDim] tables with any scaling
 *  pre-folded by the caller.
 *
 *  srcStart and srcCols optionally read a column window of a wider slab, so
 *  roping a fused qkv slice is one dispatch rather than a slice then a rope. */

const Dims = d.struct({ rows: d.u32 }); // T — the only runtime-dynamic dim

const Config = d.struct({
  cols: d.u32, // output W = heads·headDim (weight-determined)
  headDim: d.u32,
  srcStart: d.u32, // first input column of the window
  srcCols: d.u32, // input row stride (== cols when not slicing)
});
const config = tgpu.accessor(Config, { cols: 0, headDim: 2, srcStart: 0, srcCols: 0 });

function makeRopeVariant(elem: Elem) {
  const layout = tgpu.bindGroupLayout({
    x: { storage: d.arrayOf(elem.scalar), access: 'readonly' },
    cos: { storage: d.arrayOf(d.f32), access: 'readonly' }, // [T, headDim]
    sin: { storage: d.arrayOf(d.f32), access: 'readonly' }, // [T, headDim]
    out: { storage: d.arrayOf(elem.scalar), access: 'mutable' },
    dims: { uniform: Dims },
  });
  const storeScalar = elem.scalar;
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const D = layout.$.dims;
    const C = config.$;
    const pairs = d.u32(C.cols / 2);
    const idx = flatIndex(gid);
    if (idx >= D.rows * pairs) {
      return;
    }
    const row = d.u32(idx / pairs);
    const col0 = (idx % pairs) * 2;
    const dcol = col0 % C.headDim; // table column (pair-duplicated: d == d+1)
    const c = layout.$.cos[row * C.headDim + dcol]!;
    const s = layout.$.sin[row * C.headDim + dcol]!;
    const srcBase = row * C.srcCols + C.srcStart + col0;
    const x0 = d.f32(layout.$.x[srcBase]!);
    const x1 = d.f32(layout.$.x[srcBase + 1]!);
    const oBase = row * C.cols + col0;
    layout.$.out[oBase] = storeScalar(x0 * c - x1 * s);
    layout.$.out[oBase + 1] = storeScalar(x1 * c + x0 * s);
  });
  return { layout, kernel };
}

const variants = new Map<string, ReturnType<typeof makeRopeVariant>>();
export function ropeVariant(elem: Elem): ReturnType<typeof makeRopeVariant> {
  let v = variants.get(elem.key);
  if (!v) {
    v = makeRopeVariant(elem);
    variants.set(elem.key, v);
  }
  return v;
}

export function createRopePipeline(
  root: TgpuRoot,
  cfg: { cols: number; headDim: number; srcStart?: number; srcCols?: number },
  elem: Elem = F32_ELEM,
) {
  return root
    .with(config, {
      cols: cfg.cols,
      headDim: cfg.headDim,
      srcStart: cfg.srcStart ?? 0,
      srcCols: cfg.srcCols ?? cfg.cols,
    })
    .createComputePipeline({ compute: ropeVariant(elem).kernel });
}

export function ropeHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createRopePipeline>,
  args: { rows: number; cols: number },
  buffers: { x: FloatBuffer; cos: F32Buffer; sin: F32Buffer; out: FloatBuffer },
  elem: Elem = F32_ELEM,
): KernelHandle {
  const dims = cachedUniform(root, Dims, { rows: args.rows });
  const bindGroup = cachedBindGroup(root, ropeVariant(elem).layout, { ...buffers, dims });
  return makeHandle(
    pipeline,
    'rope',
    bindGroup,
    Math.ceil((args.rows * (args.cols / 2)) / WORKGROUP_SIZE),
  );
}
