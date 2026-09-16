import tgpu, { d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import { type F32Buffer, type FloatBuffer, type KernelHandle, makeHandle } from '../common.ts';
import { type Elem, F32_ELEM } from '../elem.ts';

/** Fused gemv and argmax for greedy decoding, taking the lowest index on ties.
 *  The logits vector never materializes, so the per-token readback is 4 bytes.
 *
 *  Two dispatches in the same pass, where in-pass ordering makes the partials
 *  visible to the reducer. The product is bandwidth-bound, so the layout chases
 *  throughput twice: ARGMAX_SPLIT workgroups stride the rows, and within a
 *  workgroup KL adjacent threads share one row so their loads coalesce. */

const WG = 256;
const KL = 8;
const RL = 32;
export const ARGMAX_SPLIT = 128;
const NEG_SEED = -3.4e38; // ~ -FLT_MAX; every real dot beats it

const Dims = d.struct({ rows: d.u32 });

const Config = d.struct({ cols: d.u32 });
const config = tgpu.accessor(Config, { cols: 1 });

const laneDots = tgpu.workgroupVar(d.arrayOf(d.f32, WG));
const bestVal = tgpu.workgroupVar(d.arrayOf(d.f32, RL));
const bestIdx = tgpu.workgroupVar(d.arrayOf(d.u32, RL));

function makeArgmaxDotVariant(elem: Elem) {
  const layout = tgpu.bindGroupLayout({
    w: { storage: d.arrayOf(elem.scalar), access: 'readonly' }, // [rows, cols]
    x: { storage: d.arrayOf(elem.scalar), access: 'readonly' }, // [1, cols]
    partials: { storage: d.arrayOf(d.f32), access: 'mutable' }, // [SPLIT, 2] (val, idx)
    dims: { uniform: Dims },
  });
  const kernel = tgpu.computeFn({
    in: {
      lid: d.builtin.localInvocationId,
      wid: d.builtin.workgroupId,
    },
    workgroupSize: [WG],
  })(({ lid, wid }) => {
    'use gpu';
    const rows = layout.$.dims.rows;
    const cols = config.$.cols;
    const tid = lid.x;
    const kl = tid % KL;
    const rl = d.u32(tid / KL);

    // Rows scan in increasing order, so a strict > keeps the lowest winning
    // index. The step condition is workgroup-uniform, so every thread reaches
    // the barriers the same number of times.
    let v = d.f32(NEG_SEED);
    let idx = d.u32(0);
    for (let stepBase = wid.x * RL; stepBase < rows; stepBase += RL * ARGMAX_SPLIT) {
      const row = stepBase + rl;
      let partial = d.f32(0);
      if (row < rows) {
        const base = row * cols;
        for (let j = kl; j < cols; j += KL) {
          partial += d.f32(layout.$.w[base + j]!) * d.f32(layout.$.x[j]!);
        }
      }
      laneDots.$[tid] = partial;
      std.workgroupBarrier();
      if (kl === 0 && row < rows) {
        let dot = d.f32(0);
        for (const l of tgpu.unroll(std.range(KL))) {
          dot += laneDots.$[rl * KL + l]!;
        }
        if (dot > v) {
          v = dot;
          idx = row;
        }
      }
      std.workgroupBarrier();
    }

    if (kl === 0) {
      bestVal.$[rl] = v;
      bestIdx.$[rl] = idx;
    }
    std.workgroupBarrier();

    // Cluster order is not row order across steps, so equal values must
    // compare indices explicitly.
    if (tid === 0) {
      let m = bestVal.$[0]!;
      let mi = bestIdx.$[0]!;
      for (const i of std.range(1, RL)) {
        const cv = bestVal.$[i]!;
        const ci = bestIdx.$[i]!;
        if (cv > m) {
          m = cv;
          mi = ci;
        } else if (cv === m) {
          if (ci < mi) {
            mi = ci;
          }
        }
      }
      layout.$.partials[wid.x * 2] = m;
      layout.$.partials[wid.x * 2 + 1] = d.f32(mi);
    }
  });
  return { layout, kernel };
}

const variants = new Map<string, ReturnType<typeof makeArgmaxDotVariant>>();
export function argmaxDotVariant(elem: Elem): ReturnType<typeof makeArgmaxDotVariant> {
  let v = variants.get(elem.key);
  if (!v) {
    v = makeArgmaxDotVariant(elem);
    variants.set(elem.key, v);
  }
  return v;
}

export const argmaxReduceLayout = tgpu.bindGroupLayout({
  partials: { storage: d.arrayOf(d.f32), access: 'readonly' }, // [SPLIT, 2]
  out: { storage: d.arrayOf(d.f32), access: 'mutable' }, // [1] — f32(bestRow)
});

export const argmaxDotReduceKernel = tgpu.computeFn({
  in: { gid: d.builtin.globalInvocationId },
  workgroupSize: [1],
})(({ gid }) => {
  'use gpu';
  if (gid.x > 0) return;
  let m = argmaxReduceLayout.$.partials[0]!;
  let mi = argmaxReduceLayout.$.partials[1]!;
  for (const i of std.range(1, ARGMAX_SPLIT)) {
    const cv = argmaxReduceLayout.$.partials[i * 2]!;
    const ci = argmaxReduceLayout.$.partials[i * 2 + 1]!;
    if (cv > m) {
      m = cv;
      mi = ci;
    } else if (cv === m) {
      if (ci < mi) {
        mi = ci;
      }
    }
  }
  argmaxReduceLayout.$.out[0] = mi;
});

export function createArgmaxDotPipeline(
  root: TgpuRoot,
  cfg: { cols: number },
  elem: Elem = F32_ELEM,
) {
  return root.with(config, cfg).createComputePipeline({ compute: argmaxDotVariant(elem).kernel });
}

export function createArgmaxDotReducePipeline(root: TgpuRoot) {
  return root.createComputePipeline({ compute: argmaxDotReduceKernel });
}

export function argmaxDotPartialHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createArgmaxDotPipeline>,
  rows: number,
  buffers: { w: FloatBuffer; x: FloatBuffer; partials: F32Buffer },
  elem: Elem = F32_ELEM,
): KernelHandle {
  const dims = root.createBuffer(Dims, { rows }).$usage('uniform');
  const bindGroup = root.createBindGroup(argmaxDotVariant(elem).layout, { ...buffers, dims });
  return makeHandle(pipeline, 'argmaxDot', bindGroup, ARGMAX_SPLIT);
}

export function argmaxDotReduceHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createArgmaxDotReducePipeline>,
  buffers: { partials: F32Buffer; out: F32Buffer },
): KernelHandle {
  const bindGroup = root.createBindGroup(argmaxReduceLayout, buffers);
  return makeHandle(pipeline, 'argmaxDotReduce', bindGroup, 1);
}
