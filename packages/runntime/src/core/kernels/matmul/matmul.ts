import tgpu, { d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import {
  flatIndex,
  type FloatBuffer,
  type KernelHandle,
  makeHandle,
  WORKGROUP_SIZE,
} from '../common.ts';
import { cachedBindGroup, cachedUniform } from '../../gpu/dispatchCache.ts';
import { activationFor, actSlot } from '../activations.ts';
import { type Elem, F32_ELEM } from '../elem.ts';

/** Runtime row count, plus the MoE row view. `baseRow` stays a uniform because
 *  every expert differs only there; baking it would compile one pipeline per
 *  expert. */
const Dims = d.struct({ m: d.u32, baseRow: d.u32 });

export interface MatmulCfg {
  k: number;
  n: number;
  hasBias: number; // 0 | 1 — epilogue adds bias[col]
  hasAdd: number; // 0 | 1 — epilogue adds addend[row, col] (residual fold)
  accum: number;
  hasBaseRow: number;
}

export const ACC_F32 = 0;
export const ACC_F16 = 1;

let matmulAccum = ACC_F32;
export function setMatmulAccum(mode: number): void {
  matmulAccum = mode;
}
export function getMatmulAccum(): number {
  return matmulAccum;
}

const cfgOf = (c: {
  k: number;
  n: number;
  hasBias?: number;
  hasAdd?: number;
  accum?: number;
  hasBaseRow?: number;
}): MatmulCfg => ({
  k: c.k,
  n: c.n,
  hasBias: c.hasBias ?? 0,
  hasAdd: c.hasAdd ?? 0,
  accum: c.accum ?? ACC_F32,
  hasBaseRow: c.hasBaseRow ?? 0,
});

const makeLayout = (elem: Elem) =>
  tgpu.bindGroupLayout({
    a: { storage: d.arrayOf(elem.scalar), access: 'readonly' },
    b: { storage: d.arrayOf(elem.scalar), access: 'readonly' },
    bias: { storage: d.arrayOf(elem.scalar), access: 'readonly' }, // [N]; dead when hasBias=0
    addend: { storage: d.arrayOf(elem.scalar), access: 'readonly' }, // [M,N]; dead when hasAdd=0
    out: { storage: d.arrayOf(elem.scalar), access: 'mutable' },
    dims: { uniform: Dims },
  });

const layouts = new Map<string, ReturnType<typeof makeLayout>>();
export function matmulLayoutFor(elem: Elem): ReturnType<typeof makeLayout> {
  let l = layouts.get(elem.key);
  if (!l) {
    l = makeLayout(elem);
    layouts.set(elem.key, l);
  }
  return l;
}

export const matmulLayout = matmulLayoutFor(F32_ELEM);

export function makeMatmulKernel(cfg: MatmulCfg, elem: Elem = F32_ELEM) {
  const { k, n, hasBias, hasAdd, hasBaseRow } = cfg;
  const L = matmulLayoutFor(elem);
  // Only f16 storage has anything to gain from an f16 accumulator.
  const accF16 = elem.key === 'f16' && cfg.accum === ACC_F16;
  const f16 = elem.key === 'f16';
  const n4c = Math.ceil(n / 4); // col units per row (tail unit may be partial)
  const hasTail = n % 4 !== 0;
  return tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const m = L.$.dims.m;
    const mQuads = d.u32((m + 3) / 4); // explicit integer division (codebase convention)
    const idx = flatIndex(gid);
    if (idx >= mQuads * n4c) {
      return;
    }
    const r0 = d.u32(idx / n4c) * 4;
    const col0 = (idx % n4c) * 4;
    // Clamped so loads stay in bounds; the writeback discards duplicates.
    const r1 = std.min(r0 + 1, m - 1);
    const r2 = std.min(r0 + 2, m - 1);
    const r3 = std.min(r0 + 3, m - 1);

    if (!hasTail || col0 + 4 <= n) {
      let acc0 = d.vec4f();
      let acc1 = d.vec4f();
      let acc2 = d.vec4f();
      let acc3 = d.vec4f();
      if (accF16) {
        // Half-width FMAs, widened once at the end.
        let h0 = d.vec4h();
        let h1 = d.vec4h();
        let h2 = d.vec4h();
        let h3 = d.vec4h();
        for (let j = d.u32(0); j < k; j++) {
          const bBase = (hasBaseRow > 0 ? L.$.dims.baseRow + j : j) * n + col0;
          const bv = d.vec4h(
            L.$.b[bBase]!,
            L.$.b[bBase + 1]!,
            L.$.b[bBase + 2]!,
            L.$.b[bBase + 3]!,
          );
          h0 += L.$.a[r0 * k + j]! * bv;
          h1 += L.$.a[r1 * k + j]! * bv;
          h2 += L.$.a[r2 * k + j]! * bv;
          h3 += L.$.a[r3 * k + j]! * bv;
        }
        acc0 = d.vec4f(h0);
        acc1 = d.vec4f(h1);
        acc2 = d.vec4f(h2);
        acc3 = d.vec4f(h3);
      }
      if (!accF16) {
        for (let j = d.u32(0); j < k; j++) {
          const bBase = (hasBaseRow > 0 ? L.$.dims.baseRow + j : j) * n + col0;
          if (f16) {
            // One conversion per weight quad, not four scalar ones.
            const bv = d.vec4f(
              d.vec4h(L.$.b[bBase]!, L.$.b[bBase + 1]!, L.$.b[bBase + 2]!, L.$.b[bBase + 3]!),
            );
            acc0 += d.f32(L.$.a[r0 * k + j]!) * bv;
            acc1 += d.f32(L.$.a[r1 * k + j]!) * bv;
            acc2 += d.f32(L.$.a[r2 * k + j]!) * bv;
            acc3 += d.f32(L.$.a[r3 * k + j]!) * bv;
          } else {
            const bv = d.vec4f(
              L.$.b[bBase]!,
              L.$.b[bBase + 1]!,
              L.$.b[bBase + 2]!,
              L.$.b[bBase + 3]!,
            );
            acc0 += L.$.a[r0 * k + j]! * bv;
            acc1 += L.$.a[r1 * k + j]! * bv;
            acc2 += L.$.a[r2 * k + j]! * bv;
            acc3 += L.$.a[r3 * k + j]! * bv;
          }
        }
      }
      if (hasBias > 0) {
        const bias = f16
          ? d.vec4f(
              d.vec4h(
                L.$.bias[col0]!,
                L.$.bias[col0 + 1]!,
                L.$.bias[col0 + 2]!,
                L.$.bias[col0 + 3]!,
              ),
            )
          : d.vec4f(L.$.bias[col0]!, L.$.bias[col0 + 1]!, L.$.bias[col0 + 2]!, L.$.bias[col0 + 3]!);
        acc0 += bias;
        acc1 += bias;
        acc2 += bias;
        acc3 += bias;
      }
      acc0 = actSlot.$(acc0);
      acc1 = actSlot.$(acc1);
      acc2 = actSlot.$(acc2);
      acc3 = actSlot.$(acc3);
      // Comptime `r`, so only the live-row test survives into WGSL.
      for (const r of tgpu.unroll(std.range(4))) {
        const row = r0 + r;
        if (row < m) {
          const oBase = row * n + col0;
          let v = d.vec4f([acc0, acc1, acc2, acc3][r]!);
          if (hasAdd > 0) {
            v += f16
              ? d.vec4f(
                  d.vec4h(
                    L.$.addend[oBase]!,
                    L.$.addend[oBase + 1]!,
                    L.$.addend[oBase + 2]!,
                    L.$.addend[oBase + 3]!,
                  ),
                )
              : d.vec4f(
                  L.$.addend[oBase]!,
                  L.$.addend[oBase + 1]!,
                  L.$.addend[oBase + 2]!,
                  L.$.addend[oBase + 3]!,
                );
          }
          if (f16) {
            const h = d.vec4h(v);
            L.$.out[oBase] = h.x;
            L.$.out[oBase + 1] = h.y;
            L.$.out[oBase + 2] = h.z;
            L.$.out[oBase + 3] = h.w;
          } else {
            L.$.out[oBase] = v.x;
            L.$.out[oBase + 1] = v.y;
            L.$.out[oBase + 2] = v.z;
            L.$.out[oBase + 3] = v.w;
          }
        }
      }
      return;
    }
    // Tail unit: one scalar per remaining column. Absent when N divides by 4.
    if (hasTail) {
      for (const r of tgpu.unroll(std.range(4))) {
        const row = r0 + r;
        if (row >= m) {
          return;
        }
        for (let col = col0; col < n; col++) {
          let acc = d.f32(0);
          if (f16) {
            for (const j of std.range(k)) {
              const bj = hasBaseRow > 0 ? L.$.dims.baseRow + j : j;
              acc += d.f32(L.$.a[row * k + j]!) * d.f32(L.$.b[bj * n + col]!);
            }
          } else {
            for (const j of std.range(k)) {
              const bj = hasBaseRow > 0 ? L.$.dims.baseRow + j : j;
              acc += L.$.a[row * k + j]! * L.$.b[bj * n + col]!;
            }
          }
          if (hasBias > 0) {
            acc += d.f32(L.$.bias[col]!);
          }
          acc = actSlot.$(acc);
          if (hasAdd > 0) {
            acc += d.f32(L.$.addend[row * n + col]!);
          }
          if (f16) {
            L.$.out[row * n + col] = d.f16(acc);
          } else {
            L.$.out[row * n + col] = acc;
          }
        }
      }
    }
  });
}

export function createMatmulPipeline(
  root: TgpuRoot,
  cfg: { k: number; n: number; hasBias?: number; hasAdd?: number; act?: number; accum?: number },
  elem: Elem = F32_ELEM,
) {
  return root
    .with(actSlot, activationFor(cfg.act))
    .createComputePipeline({ compute: makeMatmulKernel(cfgOf(cfg), elem) });
}

export function matmulHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createMatmulPipeline>,
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
  baseRow = 0,
): KernelHandle {
  const dims = cachedUniform(root, Dims, { m, baseRow });
  // Absent epilogue operands alias `a`; those pipelines never read them.
  const bindGroup = cachedBindGroup(root, matmulLayoutFor(elem), {
    ...buffers,
    bias: buffers.bias ?? buffers.a,
    addend: buffers.addend ?? buffers.a,
    dims,
  });
  // One thread per 4-row × 4-col block: ceil(m/4) · ceil(n/4).
  return makeHandle(
    pipeline,
    'matmul',
    bindGroup,
    Math.ceil((Math.ceil(m / 4) * Math.ceil(n / 4)) / WORKGROUP_SIZE),
  );
}

// ── Shared-memory tiled variant (M ≥ MATMUL_TILED_MIN_M) ───────────────────
// The kernel above re-reads each b element once per 4-row block; this one
// stages a and b tiles in workgroup memory, so each weight element is read
// once per 64 output rows. Tiles zero-pad past every edge, so the inner loop
// needs no guards.

export const MATMUL_TILED_MIN_M = 8;

const TILE = 64; // output tile side (rows and cols)
const TILE_K = 32; // K-slab depth per barrier round
const TILED_THREADS = 256; // 16×16 logical threads, a 4×4 output block each
const A_LOADS = (TILE * TILE_K) / TILED_THREADS; // co-op loads per thread
const B_LOADS = (TILE_K * TILE) / 4 / TILED_THREADS; // vec4 loads per thread

const aTile = tgpu.workgroupVar(d.arrayOf(d.f32, TILE * TILE_K));
const bTile = tgpu.workgroupVar(d.arrayOf(d.vec4f, (TILE_K * TILE) / 4));

function makeMatmulTiledKernel(cfg: MatmulCfg) {
  const { k, n, hasBias, hasAdd } = cfg;
  const tiles = Math.ceil(k / TILE_K);
  return tgpu.computeFn({
    in: { lid: d.builtin.localInvocationId, wid: d.builtin.workgroupId },
    workgroupSize: [TILED_THREADS],
  })(({ lid, wid }) => {
    'use gpu';
    const m = matmulLayout.$.dims.m;
    const tid = lid.x;
    const tr = d.u32(tid / 16); // explicit integer division (codebase convention)
    const tc = tid % 16;
    const rowBase = wid.y * TILE;
    const colBase = wid.x * TILE;
    const r0 = rowBase + tr * 4;
    const c0 = colBase + tc * 4;

    let acc0 = d.vec4f();
    let acc1 = d.vec4f();
    let acc2 = d.vec4f();
    let acc3 = d.vec4f();

    for (const t of std.range(tiles)) {
      const k0 = t * TILE_K;
      // Consecutive threads hit consecutive addresses; out-of-range elements
      // become 0, so the inner loop needs no guards.
      for (const i of tgpu.unroll(std.range(A_LOADS))) {
        const e = tid + i * TILED_THREADS;
        const er = d.u32(e / TILE_K);
        const ec = e % TILE_K;
        const ar = rowBase + er;
        const ak = k0 + ec;
        const aIdx = std.min(ar, m - 1) * k + std.min(ak, k - 1);
        aTile.$[e] = std.select(d.f32(0), matmulLayout.$.a[aIdx]!, ar < m && ak < k);
      }
      for (const iv of tgpu.unroll(std.range(B_LOADS))) {
        const e = tid + iv * TILED_THREADS; // vec4 slot: row-major [TILE_K, TILE/4]
        const er = d.u32(e / (TILE / 4));
        const ec = (e % (TILE / 4)) * 4;
        const bk = k0 + er;
        const bc = colBase + ec;
        const rowOk = bk < k;
        const base = std.min(bk, k - 1) * n;
        bTile.$[e] = d.vec4f(
          std.select(d.f32(0), matmulLayout.$.b[base + std.min(bc, n - 1)]!, rowOk && bc < n),
          std.select(
            d.f32(0),
            matmulLayout.$.b[base + std.min(bc + 1, n - 1)]!,
            rowOk && bc + 1 < n,
          ),
          std.select(
            d.f32(0),
            matmulLayout.$.b[base + std.min(bc + 2, n - 1)]!,
            rowOk && bc + 2 < n,
          ),
          std.select(
            d.f32(0),
            matmulLayout.$.b[base + std.min(bc + 3, n - 1)]!,
            rowOk && bc + 3 < n,
          ),
        );
      }
      std.workgroupBarrier();

      for (const kk of std.range(TILE_K)) {
        const bv = bTile.$[kk * (TILE / 4) + tc]!;
        const aBase = tr * 4 * TILE_K + kk;
        acc0 += aTile.$[aBase]! * bv;
        acc1 += aTile.$[aBase + TILE_K]! * bv;
        acc2 += aTile.$[aBase + 2 * TILE_K]! * bv;
        acc3 += aTile.$[aBase + 3 * TILE_K]! * bv;
      }
      std.workgroupBarrier();
    }

    if (hasBias > 0) {
      const bc0 = std.min(c0, n - 1);
      const bias = d.vec4f(
        matmulLayout.$.bias[bc0]!,
        matmulLayout.$.bias[std.min(c0 + 1, n - 1)]!,
        matmulLayout.$.bias[std.min(c0 + 2, n - 1)]!,
        matmulLayout.$.bias[std.min(c0 + 3, n - 1)]!,
      );
      acc0 += bias;
      acc1 += bias;
      acc2 += bias;
      acc3 += bias;
    }
    acc0 = actSlot.$(acc0);
    acc1 = actSlot.$(acc1);
    acc2 = actSlot.$(acc2);
    acc3 = actSlot.$(acc3);

    // Column guard first: a tail tile's c0 can land past n, and row*n + c0
    // would then point into the next row.
    if (c0 < n) {
      for (const r of tgpu.unroll(std.range(4))) {
        const row = r0 + r;
        if (row < m) {
          const oBase = row * n + c0;
          let v = d.vec4f([acc0, acc1, acc2, acc3][r]!);
          if (hasAdd > 0) {
            v += d.vec4f(
              matmulLayout.$.addend[oBase]!,
              matmulLayout.$.addend[std.min(oBase + 1, m * n - 1)]!,
              matmulLayout.$.addend[std.min(oBase + 2, m * n - 1)]!,
              matmulLayout.$.addend[std.min(oBase + 3, m * n - 1)]!,
            );
          }
          matmulLayout.$.out[oBase] = v.x;
          if (c0 + 1 < n) {
            matmulLayout.$.out[oBase + 1] = v.y;
          }
          if (c0 + 2 < n) {
            matmulLayout.$.out[oBase + 2] = v.z;
          }
          if (c0 + 3 < n) {
            matmulLayout.$.out[oBase + 3] = v.w;
          }
        }
      }
    }
  });
}

export function createMatmulTiledPipeline(
  root: TgpuRoot,
  cfg: { k: number; n: number; hasBias?: number; hasAdd?: number; act?: number },
) {
  return root
    .with(actSlot, activationFor(cfg.act))
    .createComputePipeline({ compute: makeMatmulTiledKernel(cfgOf(cfg)) });
}

export function matmulTiledHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createMatmulTiledPipeline>,
  m: number,
  n: number,
  buffers: {
    a: FloatBuffer;
    b: FloatBuffer;
    out: FloatBuffer;
    bias?: FloatBuffer;
    addend?: FloatBuffer;
  },
): KernelHandle {
  const dims = cachedUniform(root, Dims, { m, baseRow: 0 });
  const bindGroup = cachedBindGroup(root, matmulLayout, {
    ...buffers,
    bias: buffers.bias ?? buffers.a,
    addend: buffers.addend ?? buffers.a,
    dims,
  });
  // One workgroup per 64×64 output tile.
  return makeHandle(pipeline, 'matmulTiled', bindGroup, [Math.ceil(n / TILE), Math.ceil(m / TILE)]);
}
