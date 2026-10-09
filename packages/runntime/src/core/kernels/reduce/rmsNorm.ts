/** RMSNorm in one dispatch: x · rsqrt(mean(x²) + eps) · weight. One
 *  workgroup per row.
 *
 *  Each thread sums x² over its share of the row, in vec4 when cols % 4 == 0.
 *  With subgroups the workgroup is one 32-lane subgroup and subgroupAdd adds
 *  up the row; without, a 64-thread tree in workgroup memory does. Sums in
 *  f32.
 *
 *  Based on webgpu-kernels/ai.onnx.RMSNormalization (norm-row-stats). */

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

const Dims = d.struct({ rows: d.u32, cols: d.u32, eps: d.f32 });

const SUBGROUP_WIDTH = 32;

const TREE_STRIDES = Array.from(
  { length: Math.log2(WORKGROUP_SIZE) },
  (_, level) => WORKGROUP_SIZE >> (level + 1),
);

const partial = tgpu.workgroupVar(d.arrayOf(d.f32, WORKGROUP_SIZE));

/** Workgroup sum through a fixed pairwise tree, so every run gives the same
 *  result. Every thread gets the total. */
const treeSum = (v: number, tid: number): number => {
  'use gpu';
  partial.$[tid] = v;
  std.workgroupBarrier();
  for (const stride of tgpu.unroll(TREE_STRIDES)) {
    if (tid < stride) {
      partial.$[tid] = partial.$[tid]! + partial.$[tid + stride]!;
    }
    std.workgroupBarrier();
  }
  return partial.$[0]!;
};

const makeScalarLayout = (elem: Elem) =>
  tgpu.bindGroupLayout({
    x: { storage: d.arrayOf(elem.scalar), access: 'readonly' },
    weight: { storage: d.arrayOf(elem.scalar), access: 'readonly' },
    out: { storage: d.arrayOf(elem.scalar), access: 'mutable' },
    dims: { uniform: Dims },
  });

// The same buffers read as vec4. Rows start on a vec4 boundary because
// cols % 4 == 0.
const makeVec4Layout = (elem: Elem) =>
  tgpu.bindGroupLayout({
    x: { storage: d.arrayOf(elem.vec4), access: 'readonly' },
    weight: { storage: d.arrayOf(elem.vec4), access: 'readonly' },
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

function makeVariant(elem: Elem, vec4: boolean, subgroups: boolean) {
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

    let sumSq = d.f32(0);
    if (vec4) {
      const base = row * words;
      for (let i = tid; i < words; i += wg) {
        const v = d.vec4f(V.$.x[base + i]!);
        sumSq += std.dot(v, v);
      }
    } else {
      const base = row * D.cols;
      for (let j = tid; j < D.cols; j += wg) {
        const v = d.f32(S.$.x[base + j]!);
        sumSq += v * v;
      }
    }

    // Row total. The early return above is the same for the whole workgroup,
    // so the barriers and subgroup ops here are safe.
    const total = subgroups ? std.subgroupAdd(sumSq) : treeSum(sumSq, tid);
    const inv = std.inverseSqrt(total / d.f32(D.cols) + D.eps);

    if (vec4) {
      const base = row * words;
      for (let i = tid; i < words; i += wg) {
        const v = d.vec4f(V.$.x[base + i]!) * inv;
        V.$.out[base + i] = storeVec4(v * d.vec4f(V.$.weight[i]!));
      }
    } else {
      const base = row * D.cols;
      for (let j = tid; j < D.cols; j += wg) {
        const v = d.f32(S.$.x[base + j]!) * inv;
        S.$.out[base + j] = storeScalar(v * d.f32(S.$.weight[j]!));
      }
    }
  });
  return { layout: vec4 ? V : S, kernel };
}

const variants = new Map<string, ReturnType<typeof makeVariant>>();
export function rmsNormVariant(
  elem: Elem,
  vec4: boolean,
  subgroups: boolean,
): ReturnType<typeof makeVariant> {
  const key = `${elem.key}:${vec4}:${subgroups}`;
  let v = variants.get(key);
  if (!v) {
    v = makeVariant(elem, vec4, subgroups);
    variants.set(key, v);
  }
  return v;
}

export function createRmsNormPipeline(
  root: TgpuRoot,
  elem: Elem = F32_ELEM,
  vec4 = false,
  subgroups = false,
) {
  return root.createComputePipeline({ compute: rmsNormVariant(elem, vec4, subgroups).kernel });
}

export function rmsNormHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createRmsNormPipeline>,
  args: { rows: number; cols: number; eps: number; vec4: boolean; subgroups: boolean },
  buffers: { x: FloatBuffer; weight: FloatBuffer; out: FloatBuffer },
  elem: Elem = F32_ELEM,
): KernelHandle {
  const dims = cachedUniform(root, Dims, { rows: args.rows, cols: args.cols, eps: args.eps });
  const { layout } = rmsNormVariant(elem, args.vec4, args.subgroups);
  const bindGroup = cachedBindGroup(root, layout, { ...buffers, dims });
  // One workgroup per row.
  return makeHandle(pipeline, 'rmsNorm', bindGroup, args.rows);
}
