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
 *  dispatch, one thread per pair of columns (a, b):
 *
 *      out[a] = x[a]·cos[t,a] − x[b]·sin[t,a]
 *      out[b] = x[b]·cos[t,a] + x[a]·sin[t,a]
 *
 *  The layout picks the pairs of a head: 'interleaved' is (2p, 2p + 1),
 *  'half' is (p, p + headDim/2), HF's rotate_half.
 *
 *  cos and sin are [T, headDim] tables with one entry per column of a head,
 *  both columns of a pair holding the pair's value; the kernel reads the
 *  first column's. A scale that is part of the rope (YaRN's concentration) is
 *  in the tables; the score scale is sdpa's.
 *
 *  srcStart and srcCols optionally read a column window of a wider slab, so
 *  roping a fused qkv slice is one dispatch rather than a slice then a rope. */

/** The pair layouts, in the order a rope node's layout attr indexes. */
export const ROPE_LAYOUTS = ['interleaved', 'half'] as const;

/** Which columns of a head make a rotation pair. */
export type RopeLayout = (typeof ROPE_LAYOUTS)[number];

const Dims = d.struct({ rows: d.u32 }); // T — the only runtime-dynamic dim

const Config = d.struct({
  cols: d.u32, // output W = heads·headDim (weight-determined)
  headDim: d.u32,
  srcStart: d.u32, // first input column of the window
  srcCols: d.u32, // input row stride (== cols when not slicing)
  pairStep: d.u32, // columns between the first columns of two pairs of a head
  pairGap: d.u32, // columns from a pair's first column to its second
});
const config = tgpu.accessor(Config, {
  cols: 0,
  headDim: 2,
  srcStart: 0,
  srcCols: 0,
  pairStep: 2,
  pairGap: 1,
});

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
    const pair = idx % pairs;
    const headPairs = d.u32(C.headDim / 2);
    // The pair's first column within its head, which is also its table column.
    const dcol = (pair % headPairs) * C.pairStep;
    const col0 = d.u32(pair / headPairs) * C.headDim + dcol;
    const c = layout.$.cos[row * C.headDim + dcol]!;
    const s = layout.$.sin[row * C.headDim + dcol]!;
    const srcBase = row * C.srcCols + C.srcStart + col0;
    const x0 = d.f32(layout.$.x[srcBase]!);
    const x1 = d.f32(layout.$.x[srcBase + C.pairGap]!);
    const oBase = row * C.cols + col0;
    layout.$.out[oBase] = storeScalar(x0 * c - x1 * s);
    layout.$.out[oBase + C.pairGap] = storeScalar(x1 * c + x0 * s);
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
  cfg: { cols: number; headDim: number; layout: RopeLayout; srcStart?: number; srcCols?: number },
  elem: Elem = F32_ELEM,
) {
  const half = cfg.layout === 'half';
  return root
    .with(config, {
      cols: cfg.cols,
      headDim: cfg.headDim,
      srcStart: cfg.srcStart ?? 0,
      srcCols: cfg.srcCols ?? cfg.cols,
      pairStep: half ? 1 : 2,
      pairGap: half ? cfg.headDim / 2 : 1,
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
