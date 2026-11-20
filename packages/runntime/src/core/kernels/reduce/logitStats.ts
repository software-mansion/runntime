/** Argmax and log-probability per block of logit rows, so a sampler reads
 *  back two or three numbers per block instead of the logits.
 *
 *  Row r of x holds `blocks` blocks of `block` logits from column 0. For
 *  block b of row r the kernel writes, at out[(b·rows + r)·width]:
 *  - the argmax over the block's first argmaxCols columns, of x or of
 *    (1 + w)·x − w·guide; ties take the lowest column;
 *  - that column's x minus the block's logsumexp;
 *  - with `held` (f32 column indices), the log-probability of column
 *    held[b·rows + r] the same way.
 *  width is 3 with `held`, else 2. One workgroup per block of a row, merged
 *  as a tree in workgroup memory. Sums in f32. */

import tgpu, { d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import {
  type F32Buffer,
  type FloatBuffer,
  flatWorkgroupId,
  type KernelHandle,
  makeHandle,
  WORKGROUP_SIZE,
} from '../common.ts';
import { cachedBindGroup, cachedUniform } from '../../gpu/dispatchCache.ts';
import { type Elem, F32_ELEM } from '../elem.ts';
import { NEG_FLT_MAX, TREE_STRIDES, treeMaxSum } from './softmax.ts';

const Dims = d.struct({ rows: d.u32, rowStride: d.u32, weight: d.f32 });

const partialBest = tgpu.workgroupVar(d.arrayOf(d.f32, WORKGROUP_SIZE));
const partialArg = tgpu.workgroupVar(d.arrayOf(d.u32, WORKGROUP_SIZE));

const makeLayout = (elem: Elem) =>
  tgpu.bindGroupLayout({
    x: { storage: d.arrayOf(elem.scalar), access: 'readonly' },
    guide: { storage: d.arrayOf(elem.scalar), access: 'readonly' }, // like x; dead when guided=0
    held: { storage: d.arrayOf(d.f32), access: 'readonly' }, // [blocks·rows]; dead when held=0
    out: { storage: d.arrayOf(d.f32), access: 'mutable' }, // [blocks·rows, width]
    dims: { uniform: Dims },
  });

const layouts = new Map<string, ReturnType<typeof makeLayout>>();
function layoutFor(elem: Elem): ReturnType<typeof makeLayout> {
  let l = layouts.get(elem.key);
  if (!l) {
    l = makeLayout(elem);
    layouts.set(elem.key, l);
  }
  return l;
}

export interface LogitStatsCfg {
  blocks: number;
  block: number;
  argmaxCols: number;
  guided: number;
  held: number;
}

function makeLogitStatsKernel(cfg: LogitStatsCfg, elem: Elem) {
  const { blocks, block, argmaxCols, guided, held } = cfg;
  const width = held > 0 ? 3 : 2;
  const L = layoutFor(elem);
  return tgpu.computeFn({
    in: { wid: d.builtin.workgroupId, lid: d.builtin.localInvocationIndex },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ wid, lid }) => {
    'use gpu';
    const D = L.$.dims;
    const slot = flatWorkgroupId(wid);
    // Uniform per workgroup, so the barriers below stay uniform.
    if (slot >= blocks * D.rows) return;
    const b = std.intdiv(slot, D.rows);
    const base = (slot % D.rows) * D.rowStride + b * block;

    let m = d.f32(NEG_FLT_MAX);
    let s = d.f32(0);
    let best = d.f32(NEG_FLT_MAX);
    let arg = d.u32(0);
    for (let c = lid; c < block; c += WORKGROUP_SIZE) {
      const x = d.f32(L.$.x[base + c]!);
      const mm = std.max(m, x);
      s = s * std.exp(m - mm) + std.exp(x - mm);
      m = mm;
      if (c < argmaxCols) {
        let v = x;
        if (guided > 0) {
          v = (1 + D.weight) * x - D.weight * d.f32(L.$.guide[base + c]!);
        }
        if (v > best) {
          best = v;
          arg = c;
        }
      }
    }
    const maxSum = treeMaxSum(m, s, lid);
    // The same tree for the argmax; on a tie the lower column wins.
    partialBest.$[lid] = best;
    partialArg.$[lid] = arg;
    std.workgroupBarrier();
    for (const stride of tgpu.unroll(TREE_STRIDES)) {
      if (lid < stride) {
        const va: number = partialBest.$[lid]!;
        const vb: number = partialBest.$[lid + stride]!;
        const ab: number = partialArg.$[lid + stride]!;
        if (vb > va || (vb === va && ab < partialArg.$[lid]!)) {
          partialBest.$[lid] = vb;
          partialArg.$[lid] = ab;
        }
      }
      std.workgroupBarrier();
    }
    if (lid === 0) {
      const top = partialArg.$[0]!;
      const logSum = maxSum.x + std.log(maxSum.y);
      L.$.out[slot * width] = d.f32(top);
      L.$.out[slot * width + 1] = d.f32(L.$.x[base + top]!) - logSum;
      if (held > 0) {
        const column = d.u32(L.$.held[slot]!);
        L.$.out[slot * width + 2] = d.f32(L.$.x[base + column]!) - logSum;
      }
    }
  });
}

/** One pipeline per dtype, block geometry and inputs. */
export function createLogitStatsPipeline(
  root: TgpuRoot,
  cfg: LogitStatsCfg,
  elem: Elem = F32_ELEM,
) {
  return root.createComputePipeline({ compute: makeLogitStatsKernel(cfg, elem) });
}

/** One workgroup per (block, row). */
export function logitStatsHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createLogitStatsPipeline>,
  args: { rows: number; rowStride: number; blocks: number; weight: number },
  buffers: { x: FloatBuffer; guide?: FloatBuffer; held: F32Buffer; out: F32Buffer },
  elem: Elem = F32_ELEM,
): KernelHandle {
  const dims = cachedUniform(root, Dims, {
    rows: args.rows,
    rowStride: args.rowStride,
    weight: args.weight,
  });
  // An absent guide aliases x; unguided pipelines never read it.
  const bindGroup = cachedBindGroup(root, layoutFor(elem), {
    ...buffers,
    guide: buffers.guide ?? buffers.x,
    dims,
  });
  return makeHandle(pipeline, 'logitStats', bindGroup, args.blocks * args.rows);
}
