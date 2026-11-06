/** Softmax over a logit vector and its k best entries, best first, in one
 *  dispatch: the same list decodeTopK in zoo/mobilenetv4/pipeline.ts gives. */

import tgpu, { d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import { WORKGROUP_SIZE, type F32Buffer } from '../common.ts';
import { cachedBindGroup, cachedUniform } from '../../gpu/dispatchCache.ts';

/** One entry of the list: the index and its softmax probability. */
export const classScore = d.struct({ classId: d.u32, confidence: d.f32 });

/** `k` entries, best first. Both fields are 4 bytes, so nothing is padded
 *  and the kernel writes it as words. */
export const classScoreList = (k: number) => d.arrayOf(classScore, k);

const Params = d.struct({ n: d.u32, k: d.u32 });

const layout = tgpu.bindGroupLayout({
  logits: { storage: d.arrayOf(d.f32), access: 'readonly' },
  out: { storage: d.arrayOf(d.u32), access: 'mutable' },
  params: { uniform: Params },
});

const TREE_STRIDES = Array.from(
  { length: Math.log2(WORKGROUP_SIZE) },
  (_, level) => WORKGROUP_SIZE >> (level + 1),
);

const partial = tgpu.workgroupVar(d.arrayOf(d.f32, WORKGROUP_SIZE));

/** Largest finite f32, the seed of the max fold. */
const F32_MAX = 3.4028234663852886e38;

/** One thread per entry. Every workgroup folds the max and the sum of
 *  exponents itself, so no second dispatch waits on a shared one. An
 *  entry's rank is how many entries beat it, by logit and then by index,
 *  the order of a stable sort. */
const kernel = tgpu.computeFn({
  in: { gid: d.builtin.globalInvocationId, lid: d.builtin.localInvocationId },
  workgroupSize: [WORKGROUP_SIZE],
})(({ gid, lid }) => {
  'use gpu';
  const P = layout.$.params;
  const logits = layout.$.logits;
  const tid = lid.x;
  let hi = d.f32(-F32_MAX);
  for (let i = tid; i < P.n; i += WORKGROUP_SIZE) {
    hi = std.max(hi, logits[i]!);
  }
  partial.$[tid] = hi;
  std.workgroupBarrier();
  for (const stride of tgpu.unroll(TREE_STRIDES)) {
    if (tid < stride) {
      partial.$[tid] = std.max(partial.$[tid]!, partial.$[tid + stride]!);
    }
    std.workgroupBarrier();
  }
  const max = partial.$[0]!;
  std.workgroupBarrier();
  let sum = d.f32(0);
  for (let i = tid; i < P.n; i += WORKGROUP_SIZE) {
    sum += std.exp(logits[i]! - max);
  }
  partial.$[tid] = sum;
  std.workgroupBarrier();
  for (const stride of tgpu.unroll(TREE_STRIDES)) {
    if (tid < stride) {
      partial.$[tid] = partial.$[tid]! + partial.$[tid + stride]!;
    }
    std.workgroupBarrier();
  }
  const total = partial.$[0]!;

  const c = gid.x;
  if (c >= P.n) {
    return;
  }
  const mine = logits[c]!;
  let rank = d.u32(0);
  for (let j = d.u32(0); j < P.n; j++) {
    const v = logits[j]!;
    if (v > mine || (v === mine && j < c)) {
      rank += d.u32(1);
    }
  }
  if (rank >= P.k) {
    return;
  }
  layout.$.out[2 * rank] = c;
  layout.$.out[2 * rank + 1] = std.bitcastF32toU32(std.exp(mine - max) / total);
});

/** Records logits-to-list dispatches. Create once per model. */
export function createSoftmaxTopK(root: TgpuRoot) {
  const pipeline = root.createComputePipeline({ compute: kernel });
  return {
    /** Writes the `k` best of the `n` logits into `out`, a classScoreList. */
    encode(
      encoder: GPUCommandEncoder,
      logits: F32Buffer,
      out: GPUBuffer,
      args: { readonly n: number; readonly k: number },
    ): void {
      const params = cachedUniform(root, Params, args);
      const pass = encoder.beginComputePass();
      pipeline
        .with(pass)
        .with(cachedBindGroup(root, layout, { logits, out, params }))
        .dispatchWorkgroups(Math.ceil(args.n / WORKGROUP_SIZE));
      pass.end();
    },
  };
}
