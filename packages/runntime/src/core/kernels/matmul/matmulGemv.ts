import tgpu, { d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import { flatWorkgroupId, type FloatBuffer, type KernelHandle, makeHandle } from '../common.ts';
import { activationFor, actSlot } from '../activations.ts';
import { type Elem, F32_ELEM } from '../elem.ts';

/** Single-row matmul, the m == 1 route the executor picks by shape. The batch
 *  kernel would leave one row's dispatch with N/4 threads and a K-deep serial
 *  loop.
 *
 *  A workgroup serves TN output columns with K split across KS lanes; partials
 *  merge through workgroup memory in fixed lane order, so the result is
 *  deterministic per shape. Epilogue order matches matmul.ts. */

const WG = 256;
const KS = 8;
const TN = WG / KS;

const Config = d.struct({
  k: d.u32,
  n: d.u32,
  hasBias: d.u32, // comptime 0|1 — epilogue adds bias[col]
  hasAdd: d.u32, // comptime 0|1 — epilogue adds addend[col] (residual fold)
});
const config = tgpu.accessor(Config, { k: 0, n: 0, hasBias: 0, hasAdd: 0 });

const partial = tgpu.workgroupVar(d.arrayOf(d.f32, WG));

function makeGemvVariant(elem: Elem) {
  const layout = tgpu.bindGroupLayout({
    a: { storage: d.arrayOf(elem.scalar), access: 'readonly' }, // [1, K]
    b: { storage: d.arrayOf(elem.scalar), access: 'readonly' }, // [K, N]
    bias: { storage: d.arrayOf(elem.scalar), access: 'readonly' }, // [N]; dead when hasBias=0
    addend: { storage: d.arrayOf(elem.scalar), access: 'readonly' }, // [1, N]; dead when hasAdd=0
    out: { storage: d.arrayOf(elem.scalar), access: 'mutable' }, // [1, N]
  });
  const storeScalar = elem.scalar;
  const kernel = tgpu.computeFn({
    in: { wid: d.builtin.workgroupId, lid: d.builtin.localInvocationId },
    workgroupSize: [WG],
  })(({ wid, lid }) => {
    'use gpu';
    const k = config.$.k;
    const n = config.$.n;
    const tid = lid.x;
    const colLane = tid % TN;
    const kLane = d.u32(tid / TN); // explicit integer division (codebase convention)
    const col = flatWorkgroupId(wid) * TN + colLane;

    let acc = d.f32(0);
    if (col < n) {
      for (let j = kLane; j < k; j += KS) {
        acc += d.f32(layout.$.a[j]!) * d.f32(layout.$.b[j * n + col]!);
      }
    }
    partial.$[tid] = acc;
    std.workgroupBarrier();

    if (kLane === 0) {
      if (col < n) {
        let sum = d.f32(0);
        for (const s of tgpu.unroll(std.range(KS))) {
          sum += partial.$[s * TN + colLane]!;
        }
        if (config.$.hasBias > 0) {
          sum += d.f32(layout.$.bias[col]!);
        }
        if (config.$.hasAdd > 0) {
          sum += d.f32(layout.$.addend[col]!);
        }
        layout.$.out[col] = storeScalar(sum);
      }
    }
  });
  return { layout, kernel };
}

const gemvVariants = new Map<string, ReturnType<typeof makeGemvVariant>>();
export function matmulGemvVariant(elem: Elem): ReturnType<typeof makeGemvVariant> {
  let v = gemvVariants.get(elem.key);
  if (!v) {
    v = makeGemvVariant(elem);
    gemvVariants.set(elem.key, v);
  }
  return v;
}

export function createMatmulGemvPipeline(
  root: TgpuRoot,
  cfg: { k: number; n: number; hasBias?: number; hasAdd?: number },
  elem: Elem = F32_ELEM,
) {
  return root
    .with(config, {
      k: cfg.k,
      n: cfg.n,
      hasBias: cfg.hasBias ?? 0,
      hasAdd: cfg.hasAdd ?? 0,
    })
    .createComputePipeline({ compute: matmulGemvVariant(elem).kernel });
}

export function matmulGemvHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createMatmulGemvPipeline>,
  n: number,
  buffers: {
    a: FloatBuffer;
    b: FloatBuffer;
    out: FloatBuffer;
    bias?: FloatBuffer;
    addend?: FloatBuffer;
  },
  elem: Elem = F32_ELEM,
): KernelHandle {
  // Absent epilogue operands alias `a`; those pipelines never read them.
  const bindGroup = root.createBindGroup(matmulGemvVariant(elem).layout, {
    ...buffers,
    bias: buffers.bias ?? buffers.a,
    addend: buffers.addend ?? buffers.a,
  });
  // One workgroup per TN output columns.
  return makeHandle(pipeline, 'matmulGemv', bindGroup, Math.ceil(n / TN));
}

// ── Subgroup small-m variant (4 ≤ m ≤ MATMUL_SMALL_M_MAX_ROWS, m % 4 == 0) ─

export const MATMUL_SMALL_M_MAX_ROWS = 32;

const SUBGROUP_WIDTH = 32;

const BLOCK_ROWS = 4;

interface SmallMCfg {
  k: number;
  n: number; // n % 4 == 0 (the executor keeps other widths off this route)
  hasBias: number; // 0|1 — epilogue adds bias[col]
  hasAdd: number; // 0|1 — epilogue adds addend[row, col] (residual)
}

function makeMatmulSmallMKernel(cfg: SmallMCfg, elem: Elem) {
  const { k, n, hasBias, hasAdd } = cfg;
  const fullSteps = Math.floor(k / SUBGROUP_WIDTH);
  const tailLanes = k % SUBGROUP_WIDTH; // lanes with one extra K element
  const layout = matmulGemvVariant(elem).layout;
  const storeScalar = elem.scalar;
  const f16 = elem.key === 'f16';
  return tgpu.computeFn({
    in: { wid: d.builtin.workgroupId, lid: d.builtin.localInvocationId },
    workgroupSize: [SUBGROUP_WIDTH],
  })(({ wid, lid }) => {
    'use gpu';
    const lane = lid.x;
    const col0 = wid.x * 4;
    const row0 = wid.y * BLOCK_ROWS;

    // Indexed with a comptime `r`, so they stay in registers.
    const accs = d.arrayOf(d.vec4f, BLOCK_ROWS)();
    for (const step of std.range(fullSteps)) {
      const j = lane + d.u32(step) * SUBGROUP_WIDTH;
      const bBase = j * n + col0;
      const bv = f16
        ? d.vec4f(
            d.vec4h(
              layout.$.b[bBase]!,
              layout.$.b[bBase + 1]!,
              layout.$.b[bBase + 2]!,
              layout.$.b[bBase + 3]!,
            ),
          )
        : d.vec4f(
            layout.$.b[bBase]!,
            layout.$.b[bBase + 1]!,
            layout.$.b[bBase + 2]!,
            layout.$.b[bBase + 3]!,
          );
      const aBase = row0 * k + j;
      for (const r of tgpu.unroll(std.range(BLOCK_ROWS))) {
        accs[r] = accs[r]! + d.f32(layout.$.a[aBase + r * k]!) * bv;
      }
    }
    // k % 32 tail. Absent from the shader when k divides evenly.
    if (tailLanes > 0) {
      if (lane < tailLanes) {
        const j = lane + d.u32(fullSteps) * SUBGROUP_WIDTH;
        const bBase = j * n + col0;
        const bv = f16
          ? d.vec4f(
              d.vec4h(
                layout.$.b[bBase]!,
                layout.$.b[bBase + 1]!,
                layout.$.b[bBase + 2]!,
                layout.$.b[bBase + 3]!,
              ),
            )
          : d.vec4f(
              layout.$.b[bBase]!,
              layout.$.b[bBase + 1]!,
              layout.$.b[bBase + 2]!,
              layout.$.b[bBase + 3]!,
            );
        const aBase = row0 * k + j;
        for (const r of tgpu.unroll(std.range(BLOCK_ROWS))) {
          accs[r] = accs[r]! + d.f32(layout.$.a[aBase + r * k]!) * bv;
        }
      }
    }
    for (const r of tgpu.unroll(std.range(BLOCK_ROWS))) {
      accs[r] = std.subgroupAdd(accs[r]!);
    }

    if (lane === 0) {
      if (hasBias > 0) {
        const bias = f16
          ? d.vec4f(
              d.vec4h(
                layout.$.bias[col0]!,
                layout.$.bias[col0 + 1]!,
                layout.$.bias[col0 + 2]!,
                layout.$.bias[col0 + 3]!,
              ),
            )
          : d.vec4f(
              layout.$.bias[col0]!,
              layout.$.bias[col0 + 1]!,
              layout.$.bias[col0 + 2]!,
              layout.$.bias[col0 + 3]!,
            );
        for (const r of tgpu.unroll(std.range(BLOCK_ROWS))) {
          accs[r] = accs[r]! + bias;
        }
      }
      for (const r of tgpu.unroll(std.range(BLOCK_ROWS))) {
        accs[r] = actSlot.$(accs[r]!);
      }
      for (const r of tgpu.unroll(std.range(BLOCK_ROWS))) {
        const oBase = (row0 + r) * n + col0;
        if (hasAdd > 0) {
          accs[r] =
            accs[r]! +
            (f16
              ? d.vec4f(
                  d.vec4h(
                    layout.$.addend[oBase]!,
                    layout.$.addend[oBase + 1]!,
                    layout.$.addend[oBase + 2]!,
                    layout.$.addend[oBase + 3]!,
                  ),
                )
              : d.vec4f(
                  layout.$.addend[oBase]!,
                  layout.$.addend[oBase + 1]!,
                  layout.$.addend[oBase + 2]!,
                  layout.$.addend[oBase + 3]!,
                ));
        }
        layout.$.out[oBase] = storeScalar(accs[r]!.x);
        layout.$.out[oBase + 1] = storeScalar(accs[r]!.y);
        layout.$.out[oBase + 2] = storeScalar(accs[r]!.z);
        layout.$.out[oBase + 3] = storeScalar(accs[r]!.w);
      }
    }
  });
}

export function createMatmulSmallMPipeline(
  root: TgpuRoot,
  cfg: { k: number; n: number; hasBias?: number; hasAdd?: number; act?: number },
  elem: Elem = F32_ELEM,
) {
  return root.with(actSlot, activationFor(cfg.act)).createComputePipeline({
    compute: makeMatmulSmallMKernel(
      {
        k: cfg.k,
        n: cfg.n,
        hasBias: cfg.hasBias ?? 0,
        hasAdd: cfg.hasAdd ?? 0,
      },
      elem,
    ),
  });
}

export function matmulSmallMHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createMatmulSmallMPipeline>,
  m: number,
  n: number,
  buffers: {
    a: FloatBuffer;
    b: FloatBuffer;
    out: FloatBuffer;
    bias?: FloatBuffer;
    addend?: FloatBuffer;
  },
  elem: Elem = F32_ELEM,
): KernelHandle {
  // Absent epilogue operands alias `a`; those pipelines never read them.
  const bindGroup = root.createBindGroup(matmulGemvVariant(elem).layout, {
    ...buffers,
    bias: buffers.bias ?? buffers.a,
    addend: buffers.addend ?? buffers.a,
  });
  // One 32-lane workgroup per (4-column unit, BLOCK_ROWS-row block).
  return makeHandle(pipeline, 'matmulSmallM', bindGroup, [n / 4, m / BLOCK_ROWS]);
}
