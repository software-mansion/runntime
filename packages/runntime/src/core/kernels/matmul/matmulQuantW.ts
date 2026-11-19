/** Matmul with a quantW weight for a few rows: out = act(a·W + bias) + addend.
 *  W holds int8 or int4 values in the blocked-unit layout, times f32 group
 *  scales. a, bias, addend and out are f16 or f32; sums are f32.
 *
 *  A workgroup serves ROWS rows × UNITS 4-column units, with K split across
 *  SLICES lanes per unit; partials add up in a fixed tree in workgroup memory,
 *  so every run gives the same result. W can be a view into one expert of a
 *  fused [E·K, N] weight: baseRow must be a K-aligned expert boundary, which
 *  the op enforces, and scales are indexed [scaleBase + kGroup·N + col], with
 *  scaleBase a multiple of 4 since scales are read as vec4. */

import tgpu, { d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import { flatWorkgroupId, type KernelHandle, makeHandle } from '../common.ts';
import { cachedBindGroup, cachedUniform } from '../../gpu/dispatchCache.ts';
import { activationFor, actSlot } from '../activations.ts';
import { type Elem, F32_ELEM } from '../elem.ts';
import {
  columnWord,
  deqVec4,
  type MatmulQuantBuffers,
  type MatmulQuantCfg,
} from '../quantCommon.ts';

const Dims = d.struct({
  m: d.u32,
  baseRow: d.u32, // per-expert view offset, varies per call
  scaleBase: d.u32,
});

function makeLayout(elem: Elem) {
  return tgpu.bindGroupLayout({
    a: { storage: d.arrayOf(elem.vec4), access: 'readonly' }, // [M, K/4]
    w: { storage: d.arrayOf(d.u32), access: 'readonly' },
    scales: { storage: d.arrayOf(d.vec4f), access: 'readonly' }, // [.., N/4]
    bias: { storage: d.arrayOf(elem.vec4), access: 'readonly' }, // [N/4]; dead when hasBias=0
    addend: { storage: d.arrayOf(elem.vec4), access: 'readonly' }, // [M, N/4]; dead when hasAdd=0
    out: { storage: d.arrayOf(elem.vec4), access: 'mutable' },
    dims: { uniform: Dims },
  });
}

const layouts = new Map<string, ReturnType<typeof makeLayout>>();
function layoutFor(elem: Elem): ReturnType<typeof makeLayout> {
  let l = layouts.get(elem.key);
  if (!l) {
    l = makeLayout(elem);
    layouts.set(elem.key, l);
  }
  return l;
}

/** Rows of a per workgroup. Each lane keeps one f32 vec4 accumulator per row. */
const ROWS = 4;
/** Units per workgroup. A unit computes 4 output columns. */
const UNITS = 8;
/** Lanes that split one unit's K between them. */
const SLICES = 16;
/** Threads per workgroup. */
const LANES = UNITS * SLICES;

/** Steps of the tree that adds the slices: SLICES/2 down to 1. */
const SLICE_STRIDES = Array.from(
  { length: Math.log2(SLICES) },
  (_, level) => SLICES >> (level + 1),
);

const partial = tgpu.workgroupVar(d.arrayOf(d.vec4f, ROWS * LANES));

function makeMatmulQuantWKernel(cfg: MatmulQuantCfg, elem: Elem) {
  const { k, n, bits, groupSize, hasBias, hasAdd } = cfg;
  const layout = layoutFor(elem);
  const storeVec4 = elem.vec4;
  const k4 = k / 4;
  const n4 = n / 4;
  const unitGroups = Math.ceil(n4 / UNITS);
  const tileWords = bits / 2;
  const cw1 = columnWord(1, bits);
  const cw2 = columnWord(2, bits);
  const cw3 = columnWord(3, bits);
  return tgpu.computeFn({
    in: { wid: d.builtin.workgroupId, lid: d.builtin.localInvocationIndex },
    workgroupSize: [LANES],
  })(({ wid, lid }) => {
    'use gpu';
    const D = layout.$.dims;
    const group = flatWorkgroupId(wid);
    const row0 = std.intdiv(group, unitGroups) * ROWS;
    if (row0 >= D.m) {
      return;
    }
    const lane = lid % UNITS;
    const slice = std.intdiv(lid, UNITS);
    const unit = (group % unitGroups) * UNITS + lane;
    const expert = std.intdiv(D.baseRow, k); // blocked tiles are expert-granular
    const wBase = (expert * n4 + unit) * k4 * tileWords;
    const sBase = std.intdiv(D.scaleBase, 4) + unit;

    // Indexed with a comptime `r`, so they stay in registers.
    const accs = d.arrayOf(d.vec4f, ROWS)();
    if (unit < n4) {
      for (let i = slice; i < k4; i += SLICES) {
        const tb = wBase + i * tileWords;
        const c0 = deqVec4(layout.$.w[tb]!, 0, bits);
        const c1 = deqVec4(layout.$.w[tb + cw1]!, (4 * bits) % 32, bits);
        const c2 = deqVec4(layout.$.w[tb + cw2]!, (8 * bits) % 32, bits);
        const c3 = deqVec4(layout.$.w[tb + cw3]!, (12 * bits) % 32, bits);
        const s = layout.$.scales[sBase + std.intdiv(i * 4, groupSize) * n4]!;
        for (const r of tgpu.unroll(std.range(ROWS))) {
          if (row0 + r < D.m) {
            const x = d.vec4f(layout.$.a[(row0 + r) * k4 + i]!);
            accs[r] =
              accs[r]! +
              d.vec4f(std.dot(x, c0), std.dot(x, c1), std.dot(x, c2), std.dot(x, c3)) * s;
          }
        }
      }
    }
    for (const r of tgpu.unroll(std.range(ROWS))) {
      partial.$[r * LANES + lid] = d.vec4f(accs[r]!);
    }
    std.workgroupBarrier();
    for (const stride of tgpu.unroll(SLICE_STRIDES)) {
      if (slice < stride) {
        for (const r of tgpu.unroll(std.range(ROWS))) {
          partial.$[r * LANES + lid] =
            partial.$[r * LANES + lid]! + partial.$[r * LANES + lid + stride * UNITS]!;
        }
      }
      std.workgroupBarrier();
    }

    if (slice === 0 && unit < n4) {
      let bias = d.vec4f();
      if (hasBias > 0) {
        bias = d.vec4f(layout.$.bias[unit]!);
      }
      for (const r of tgpu.unroll(std.range(ROWS))) {
        const row = row0 + r;
        if (row < D.m) {
          let v = actSlot.$(partial.$[r * LANES + lane]! + bias);
          if (hasAdd > 0) {
            v += d.vec4f(layout.$.addend[row * n4 + unit]!);
          }
          layout.$.out[row * n4 + unit] = storeVec4(v);
        }
      }
    }
  });
}

/** One pipeline per dtype, K, N, bits, group size, epilogue and activation. */
export function createMatmulQuantWPipeline(
  root: TgpuRoot,
  cfg: MatmulQuantCfg,
  elem: Elem = F32_ELEM,
) {
  return root
    .with(actSlot, activationFor(cfg.act))
    .createComputePipeline({ compute: makeMatmulQuantWKernel(cfg, elem) });
}

/** One workgroup per ROWS rows × UNITS 4-column units. */
export function matmulQuantWHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createMatmulQuantWPipeline>,
  args: { m: number; n: number; baseRow: number; scaleBase: number },
  buffers: MatmulQuantBuffers,
  elem: Elem = F32_ELEM,
): KernelHandle {
  const dims = cachedUniform(root, Dims, {
    m: args.m,
    baseRow: args.baseRow,
    scaleBase: args.scaleBase,
  });
  // Absent epilogue operands alias `a`; those pipelines never read them.
  const bindGroup = cachedBindGroup(root, layoutFor(elem), {
    ...buffers,
    bias: buffers.bias ?? buffers.a,
    addend: buffers.addend ?? buffers.a,
    dims,
  });
  return makeHandle(
    pipeline,
    'matmulQuantW',
    bindGroup,
    Math.ceil(args.n / 4 / UNITS) * Math.ceil(args.m / ROWS),
  );
}
