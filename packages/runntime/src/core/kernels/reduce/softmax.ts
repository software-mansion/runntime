/** Softmax over the last dim. One workgroup per row.
 *
 *  Each thread keeps a running max and sum over its share of the row, in vec4
 *  when cols % 4 == 0, so one read of the row gives both. With subgroups the
 *  workgroup is one 32-lane subgroup and subgroupMax / subgroupAdd combine the
 *  threads; without, a 64-thread tree in workgroup memory does. A second read
 *  writes exp(x − max) / sum. Sums in f32.
 *
 *  Based on webgpu-kernels/ai.onnx.Softmax (softmax-online). */

import tgpu, { d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import {
  type FloatBuffer,
  flatWorkgroupId,
  type KernelHandle,
  makeHandle,
  WORKGROUP_SIZE,
} from '../common.ts';
import { cachedBindGroup, cachedUniform } from '../../gpu/dispatchCache.ts';
import { type Elem, F32_ELEM } from '../elem.ts';

const Dims = d.struct({ rows: d.u32, cols: d.u32 });

const SUBGROUP_WIDTH = 32;
/** Start of the running max. A thread with no columns adds (NEG_FLT_MAX, 0),
 *  which changes nothing. */
const NEG_FLT_MAX = -3.4028234663852886e38;

const TREE_STRIDES = Array.from(
  { length: Math.log2(WORKGROUP_SIZE) },
  (_, level) => WORKGROUP_SIZE >> (level + 1),
);

const partialMax = tgpu.workgroupVar(d.arrayOf(d.f32, WORKGROUP_SIZE));
const partialSum = tgpu.workgroupVar(d.arrayOf(d.f32, WORKGROUP_SIZE));

/** Combines every lane's (max, sum) into the row's (max, sum). */
const subgroupMaxSum = (m: number, s: number): d.v2f => {
  'use gpu';
  const mx = std.subgroupMax(m);
  return d.vec2f(mx, std.subgroupAdd(s * std.exp(m - mx)));
};

/** The same through a fixed pairwise tree in workgroup memory. Each step
 *  rescales both sums to the larger max. */
const treeMaxSum = (m: number, s: number, tid: number): d.v2f => {
  'use gpu';
  partialMax.$[tid] = m;
  partialSum.$[tid] = s;
  std.workgroupBarrier();
  for (const stride of tgpu.unroll(TREE_STRIDES)) {
    if (tid < stride) {
      const ma: number = partialMax.$[tid]!;
      const mb: number = partialMax.$[tid + stride]!;
      const mm: number = std.max(ma, mb);
      partialSum.$[tid] =
        partialSum.$[tid]! * std.exp(ma - mm) + partialSum.$[tid + stride]! * std.exp(mb - mm);
      partialMax.$[tid] = mm;
    }
    std.workgroupBarrier();
  }
  return d.vec2f(partialMax.$[0]!, partialSum.$[0]!);
};

const makeScalarLayout = (elem: Elem) =>
  tgpu.bindGroupLayout({
    x: { storage: d.arrayOf(elem.scalar), access: 'readonly' },
    out: { storage: d.arrayOf(elem.scalar), access: 'mutable' },
    dims: { uniform: Dims },
  });

// The same buffers read as vec4. Rows start on a vec4 boundary because
// cols % 4 == 0.
const makeVec4Layout = (elem: Elem) =>
  tgpu.bindGroupLayout({
    x: { storage: d.arrayOf(elem.vec4), access: 'readonly' },
    out: { storage: d.arrayOf(elem.vec4), access: 'mutable' },
    dims: { uniform: Dims },
  });

interface Layouts {
  scalar: ReturnType<typeof makeScalarLayout>;
  vec4: ReturnType<typeof makeVec4Layout>;
}
const layoutsPerElem = new Map<string, Layouts>();
function layoutsFor(elem: Elem): Layouts {
  let l = layoutsPerElem.get(elem.key);
  if (!l) {
    l = { scalar: makeScalarLayout(elem), vec4: makeVec4Layout(elem) };
    layoutsPerElem.set(elem.key, l);
  }
  return l;
}

function makeSoftmaxVariant(elem: Elem, vec4: boolean, subgroups: boolean) {
  const { scalar: S, vec4: V } = layoutsFor(elem);
  const storeScalar = elem.scalar;
  const storeVec4 = elem.vec4;
  const wg = subgroups ? SUBGROUP_WIDTH : WORKGROUP_SIZE;
  const kernel = tgpu.computeFn({
    in: { wid: d.builtin.workgroupId, lid: d.builtin.localInvocationId },
    workgroupSize: [wg],
  })(({ wid, lid }) => {
    'use gpu';
    // `vec4` is fixed per pipeline, so the shader uses only one layout.
    const D = vec4 ? V.$.dims : S.$.dims;
    const row = flatWorkgroupId(wid);
    // Past 65535 rows the grid is 2-D, and its last row runs past the end.
    if (row >= D.rows) return;
    const tid = lid.x;
    const words = D.cols >>> 2;

    // This thread's running (max, sum). A new max rescales the sum so far.
    let m = d.f32(NEG_FLT_MAX);
    let s = d.f32(0);
    if (vec4) {
      const base = row * words;
      for (let i = tid; i < words; i += wg) {
        const v = d.vec4f(V.$.x[base + i]!);
        const mNew = std.max(m, std.max(std.max(v.x, v.y), std.max(v.z, v.w)));
        const e = std.exp(v - d.vec4f(mNew));
        s = s * std.exp(m - mNew) + (e.x + e.y + e.z + e.w);
        m = mNew;
      }
    } else {
      const base = row * D.cols;
      for (let j = tid; j < D.cols; j += wg) {
        const v = d.f32(S.$.x[base + j]!);
        const mNew = std.max(m, v);
        s = s * std.exp(m - mNew) + std.exp(v - mNew);
        m = mNew;
      }
    }

    // Row (max, sum). The early return above is the same for the whole
    // workgroup, so the barriers and subgroup ops here are safe.
    const stats = subgroups ? subgroupMaxSum(m, s) : treeMaxSum(m, s, tid);
    const rowMax = stats.x;
    const rowSum = stats.y;

    if (vec4) {
      const base = row * words;
      for (let i = tid; i < words; i += wg) {
        const v = d.vec4f(V.$.x[base + i]!);
        V.$.out[base + i] = storeVec4(std.exp(v - d.vec4f(rowMax)) / rowSum);
      }
    } else {
      const base = row * D.cols;
      for (let j = tid; j < D.cols; j += wg) {
        const v = d.f32(S.$.x[base + j]!);
        S.$.out[base + j] = storeScalar(std.exp(v - rowMax) / rowSum);
      }
    }
  });
  return { layout: vec4 ? V : S, kernel };
}

const variants = new Map<string, ReturnType<typeof makeSoftmaxVariant>>();
export function softmaxVariant(
  elem: Elem,
  vec4: boolean,
  subgroups: boolean,
): ReturnType<typeof makeSoftmaxVariant> {
  const key = `${elem.key}:${vec4}:${subgroups}`;
  let v = variants.get(key);
  if (!v) {
    v = makeSoftmaxVariant(elem, vec4, subgroups);
    variants.set(key, v);
  }
  return v;
}

export function createSoftmaxPipeline(
  root: TgpuRoot,
  elem: Elem = F32_ELEM,
  vec4 = false,
  subgroups = false,
) {
  return root.createComputePipeline({ compute: softmaxVariant(elem, vec4, subgroups).kernel });
}

export function softmaxHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createSoftmaxPipeline>,
  args: { rows: number; cols: number; vec4: boolean; subgroups: boolean },
  buffers: { x: FloatBuffer; out: FloatBuffer },
  elem: Elem = F32_ELEM,
): KernelHandle {
  const dims = cachedUniform(root, Dims, { rows: args.rows, cols: args.cols });
  const { layout } = softmaxVariant(elem, args.vec4, args.subgroups);
  const bindGroup = cachedBindGroup(root, layout, { ...buffers, dims });
  // One workgroup per row.
  return makeHandle(pipeline, 'softmax', bindGroup, args.rows);
}
