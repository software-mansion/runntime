import tgpu, { d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import {
  type FloatBuffer,
  flatWorkgroupId,
  type KernelHandle,
  makeHandle,
  WORKGROUP_SIZE,
} from '../common.ts';
import { type Elem, F32_ELEM } from '../elem.ts';

/** Fused LayerNorm in one dispatch, with torch's biased 1/N variance.
 *
 *  One workgroup per row: threads stride the columns and merge through two tree
 *  reductions — mean, then variance around it, avoiding the sumsq − mean²
 *  cancellation. The tree pairing is fixed, so results are deterministic per
 *  (cols, workgroup size).
 *
 *  Everything rides the uniform, so one pipeline serves every width, eps and
 *  bias form. */

const Dims = d.struct({ rows: d.u32, cols: d.u32, eps: d.f32, hasBias: d.u32 });

const makeLayout = (elem: Elem) =>
  tgpu.bindGroupLayout({
    x: { storage: d.arrayOf(elem.scalar), access: 'readonly' },
    weight: { storage: d.arrayOf(elem.scalar), access: 'readonly' },
    bias: { storage: d.arrayOf(elem.scalar), access: 'readonly' }, // dead when hasBias=0
    out: { storage: d.arrayOf(elem.scalar), access: 'mutable' },
    dims: { uniform: Dims },
  });

const TREE_STRIDES = Array.from(
  { length: Math.log2(WORKGROUP_SIZE) },
  (_, level) => WORKGROUP_SIZE >> (level + 1),
);

const partial = tgpu.workgroupVar(d.arrayOf(d.f32, WORKGROUP_SIZE));

function makeVariant(elem: Elem) {
  const f16 = elem.key === 'f16';
  const L = makeLayout(elem);
  const kernel = tgpu.computeFn({
    in: { wid: d.builtin.workgroupId, lid: d.builtin.localInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ wid, lid }) => {
    'use gpu';
    const D = L.$.dims;
    const row = flatWorkgroupId(wid); // one workgroup per row — dispatch count is rows
    // Past 65535 rows the grid is 2-D and its last column runs past the end;
    // those workgroups must not write (clamped OOB stores land on the last row).
    if (row >= D.rows) return;
    const tid = lid.x;
    const base = row * D.cols;

    // Row sum: threads stride the columns, partials merge through a fixed
    // pairwise tree (barriers sit in uniform control flow — the stride loop
    // bound is workgroup-uniform).
    let sum = d.f32(0);
    for (let j = tid; j < D.cols; j += WORKGROUP_SIZE) {
      if (f16) {
        sum += d.f32(L.$.x[base + j]!);
      } else {
        sum += L.$.x[base + j]!;
      }
    }
    partial.$[tid] = sum;
    std.workgroupBarrier();
    for (const stride of tgpu.unroll(TREE_STRIDES)) {
      if (tid < stride) {
        partial.$[tid] = partial.$[tid]! + partial.$[tid + stride]!;
      }
      std.workgroupBarrier();
    }
    const m = partial.$[0]! / d.f32(D.cols);
    // Every thread has read partial[0]; barrier before the next round's writes.
    std.workgroupBarrier();

    // Biased variance around the mean, same tree.
    let vs = d.f32(0);
    for (let j = tid; j < D.cols; j += WORKGROUP_SIZE) {
      if (f16) {
        const c = d.f32(L.$.x[base + j]!) - m;
        vs += c * c;
      } else {
        const c = L.$.x[base + j]! - m;
        vs += c * c;
      }
    }
    partial.$[tid] = vs;
    std.workgroupBarrier();
    for (const stride of tgpu.unroll(TREE_STRIDES)) {
      if (tid < stride) {
        partial.$[tid] = partial.$[tid]! + partial.$[tid + stride]!;
      }
      std.workgroupBarrier();
    }
    const inv = std.inverseSqrt(partial.$[0]! / d.f32(D.cols) + D.eps);

    // Normalize: threads stride the columns again (x is L1-hot by now).
    for (let j = tid; j < D.cols; j += WORKGROUP_SIZE) {
      if (f16) {
        let v = (d.f32(L.$.x[base + j]!) - m) * inv * d.f32(L.$.weight[j]!);
        if (D.hasBias > 0) {
          v += d.f32(L.$.bias[j]!);
        }
        L.$.out[base + j] = d.f16(v);
      } else {
        let v = (L.$.x[base + j]! - m) * inv * L.$.weight[j]!;
        if (D.hasBias > 0) {
          v += L.$.bias[j]!;
        }
        L.$.out[base + j] = v;
      }
    }
  });
  return { layout: L, kernel };
}

const variants = new Map<string, ReturnType<typeof makeVariant>>();
export function layerNormVariant(elem: Elem): ReturnType<typeof makeVariant> {
  let v = variants.get(elem.key);
  if (!v) {
    v = makeVariant(elem);
    variants.set(elem.key, v);
  }
  return v;
}

export const layerNormLayout = layerNormVariant(F32_ELEM).layout;

export function createLayerNormPipeline(root: TgpuRoot, elem: Elem = F32_ELEM) {
  return root.createComputePipeline({ compute: layerNormVariant(elem).kernel });
}

export function layerNormHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createLayerNormPipeline>,
  args: { rows: number; cols: number; eps: number; hasBias: number },
  buffers: { x: FloatBuffer; weight: FloatBuffer; bias: FloatBuffer; out: FloatBuffer },
  elem: Elem = F32_ELEM,
): KernelHandle {
  const dims = root
    .createBuffer(Dims, { rows: args.rows, cols: args.cols, eps: args.eps, hasBias: args.hasBias })
    .$usage('uniform');
  const bindGroup = root.createBindGroup(layerNormVariant(elem).layout, { ...buffers, dims });
  // One workgroup per row — not one thread.
  return makeHandle(pipeline, 'layerNorm', bindGroup, args.rows);
}
