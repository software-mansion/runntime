import tgpu, { d } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import {
  type F32Buffer,
  type FloatBuffer,
  type KernelHandle,
  flatIndex,
  makeHandle,
  WORKGROUP_SIZE,
} from '../common.ts';
import { cachedBindGroup, cachedUniform } from '../../gpu/dispatchCache.ts';
import { type Elem, F32_ELEM } from '../elem.ts';

/** Indexed matmul plus bias, with the expert index read per row from a GPU
 *  buffer, so routing stays on the device with no readback.
 *
 *  `expertIdx` stays f32 in every variant: it holds row numbers, and f16 is
 *  exact only to 2048. One thread per (token, 4-column unit), so N must divide
 *  by 4. Only M rides a uniform. */

const Dims = d.struct({ m: d.u32 });

const Config = d.struct({
  k: d.u32,
  n: d.u32,
  n4: d.u32, // n/4 — col units per row
});
const config = tgpu.accessor(Config, { k: 0, n: 0, n4: 0 });

function makeGatherVariant(elem: Elem) {
  const layout = tgpu.bindGroupLayout({
    a: { storage: d.arrayOf(elem.scalar), access: 'readonly' },
    w: { storage: d.arrayOf(elem.scalar), access: 'readonly' }, // [E·K, N]
    bias: { storage: d.arrayOf(elem.scalar), access: 'readonly' }, // [E, N]
    expertIdx: { storage: d.arrayOf(d.f32), access: 'readonly' }, // [m], from topk
    out: { storage: d.arrayOf(elem.scalar), access: 'mutable' },
    dims: { uniform: Dims },
  });
  const storeScalar = elem.scalar;
  const f16 = elem.key === 'f16';
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const C = config.$;
    const idx = flatIndex(gid);
    if (idx >= layout.$.dims.m * C.n4) {
      return;
    }
    const row = d.u32(idx / C.n4);
    const col0 = (idx % C.n4) * 4;

    const expert = d.u32(layout.$.expertIdx[row]!);
    const wRowBase = expert * C.k;

    let acc = d.vec4f();
    for (let j = d.u32(0); j < C.k; j++) {
      const wBase = (wRowBase + j) * C.n + col0;
      // One vec4h -> vec4f conversion per weight quad, not four scalar ones.
      const wv = f16
        ? d.vec4f(
            d.vec4h(
              layout.$.w[wBase]!,
              layout.$.w[wBase + 1]!,
              layout.$.w[wBase + 2]!,
              layout.$.w[wBase + 3]!,
            ),
          )
        : d.vec4f(
            layout.$.w[wBase]!,
            layout.$.w[wBase + 1]!,
            layout.$.w[wBase + 2]!,
            layout.$.w[wBase + 3]!,
          );
      acc += d.f32(layout.$.a[row * C.k + j]!) * wv;
    }

    // Fold in per-expert bias.
    const bBase = expert * C.n + col0;
    acc += f16
      ? d.vec4f(
          d.vec4h(
            layout.$.bias[bBase]!,
            layout.$.bias[bBase + 1]!,
            layout.$.bias[bBase + 2]!,
            layout.$.bias[bBase + 3]!,
          ),
        )
      : d.vec4f(
          layout.$.bias[bBase]!,
          layout.$.bias[bBase + 1]!,
          layout.$.bias[bBase + 2]!,
          layout.$.bias[bBase + 3]!,
        );

    const oBase = row * C.n + col0;
    layout.$.out[oBase] = storeScalar(acc.x);
    layout.$.out[oBase + 1] = storeScalar(acc.y);
    layout.$.out[oBase + 2] = storeScalar(acc.z);
    layout.$.out[oBase + 3] = storeScalar(acc.w);
  });
  return { layout, kernel };
}

const variants = new Map<string, ReturnType<typeof makeGatherVariant>>();
export function matmulGatherVariant(elem: Elem): ReturnType<typeof makeGatherVariant> {
  let v = variants.get(elem.key);
  if (!v) {
    v = makeGatherVariant(elem);
    variants.set(elem.key, v);
  }
  return v;
}

export function createMatmulGatherPipeline(
  root: TgpuRoot,
  cfg: { k: number; n: number },
  elem: Elem = F32_ELEM,
) {
  return root
    .with(config, { k: cfg.k, n: cfg.n, n4: cfg.n / 4 })
    .createComputePipeline({ compute: matmulGatherVariant(elem).kernel });
}

export function matmulGatherHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createMatmulGatherPipeline>,
  args: { m: number; n: number },
  buffers: {
    a: FloatBuffer;
    w: FloatBuffer;
    bias: FloatBuffer;
    expertIdx: F32Buffer;
    out: FloatBuffer;
  },
  elem: Elem = F32_ELEM,
): KernelHandle {
  const dims = cachedUniform(root, Dims, { m: args.m });
  const bindGroup = cachedBindGroup(root, matmulGatherVariant(elem).layout, { ...buffers, dims });
  return makeHandle(
    pipeline,
    'matmulGather',
    bindGroup,
    Math.ceil((args.m * (args.n / 4)) / WORKGROUP_SIZE),
  );
}
