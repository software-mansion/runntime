import { describe, expect, it } from 'vitest';
import { toArray } from '../../../../src/core/gpu/eval.ts';
import { moeDown, moeUp, type QuantExpertWeight } from '../../../../src/core/graph/ops.ts';
import { MAX_WORKGROUPS_PER_DIM } from '../../../../src/core/kernels/common.ts';
import { MOE_SLOTS } from '../../../../src/core/kernels/matmul/moeQuantW.ts';
import { expectClose, gpu, randomF32, upload } from '../../../gpu.ts';
import { type QuantWeight, randomQuantWeight } from './quantW.ts';

const { root, ex } = await gpu();

/** gpt-oss SwiGLU from its definition: g·σ(1.702·g)·(l + 1), g = min(glu, 7),
 *  l = clamp(lin, ±7). */
function clampedSwiglu(glu: number, lin: number): number {
  const g = Math.min(glu, 7);
  const l = Math.min(Math.max(lin, -7), 7);
  return (g / (1 + Math.exp(-1.702 * g))) * (l + 1);
}

interface Shape {
  tokens: number;
  k: number;
  n: number;
  experts: number;
  bits: 4 | 8;
  groupSize: number;
}

/** topk's packed [T, 8] output: 4 expert ids, then 4 gate weights summing
 *  to 1. Ids cycle through the experts and repeat when there are fewer than 4.
 *  Gates rotate per token, so reading another token's gates fails. */
function routeFor(tokens: number, experts: number): Float32Array {
  const route = new Float32Array(tokens * 2 * MOE_SLOTS);
  for (let t = 0; t < tokens; t++) {
    for (let s = 0; s < MOE_SLOTS; s++) {
      route[t * 2 * MOE_SLOTS + s] = (t * 3 + s * 5) % experts;
      route[t * 2 * MOE_SLOTS + MOE_SLOTS + s] = [0.4, 0.3, 0.2, 0.1][(s + t) % MOE_SLOTS]!;
    }
  }
  return route;
}

/** One routed layer: the packed weight, its scales and bias as Values, and
 *  the dequantized weight and bias for the reference. */
function expertLayer(
  s: Shape,
  seed: number,
): { layer: QuantExpertWeight; q: QuantWeight; bias: Float32Array } {
  const q = randomQuantWeight(root, s, seed);
  const bias = randomF32(s.experts * s.n, seed + 2);
  const layer = {
    w: q.w,
    scales: upload(root, q.scales, [q.scales.length / s.n, s.n]),
    bias: upload(root, bias, [s.experts, s.n]),
  };
  return { layer, q, bias };
}

/** x[row] · W[e] + bias[e] for every column, in f64. */
function project(
  x: Float32Array,
  row: number,
  l: { q: QuantWeight; bias: Float32Array },
  e: number,
  s: Shape,
) {
  return Array.from({ length: s.n }, (_, c) => {
    let sum = l.bias[e * s.n + c]!;
    for (let j = 0; j < s.k; j++) sum += x[row * s.k + j]! * l.q.dequant[(e * s.k + j) * s.n + c]!;
    return sum;
  });
}

async function checkMoeUp(s: Shape) {
  // Scaled so some projections pass the ±7 clamps.
  const x = randomF32(s.tokens * s.k, 1).map((v) => 4 * v);
  const route = routeFor(s.tokens, s.experts);
  const glu = expertLayer(s, 10);
  const lin = expertLayer(s, 20);
  const got = await toArray(
    moeUp(
      upload(root, x, [s.tokens, s.k]),
      upload(root, route, [s.tokens, 2 * MOE_SLOTS]),
      glu.layer,
      lin.layer,
      s,
    ),
    ex,
  );
  const want = new Float64Array(s.tokens * MOE_SLOTS * s.n);
  for (let t = 0; t < s.tokens; t++) {
    for (let slot = 0; slot < MOE_SLOTS; slot++) {
      const e = route[t * 2 * MOE_SLOTS + slot]!;
      const g = project(x, t, glu, e, s);
      const l = project(x, t, lin, e, s);
      for (let c = 0; c < s.n; c++) {
        want[(t * MOE_SLOTS + slot) * s.n + c] = clampedSwiglu(g[c]!, l[c]!);
      }
    }
  }
  expectClose(got, want, { atol: 1e-5 * s.k, rtol: 1e-4 });
}

async function checkMoeDown(s: Shape) {
  const act = randomF32(s.tokens * MOE_SLOTS * s.k, 1);
  const route = routeFor(s.tokens, s.experts);
  const down = expertLayer(s, 30);
  const residual = randomF32(s.tokens * s.n, 4);
  const got = await toArray(
    moeDown(
      upload(root, act, [s.tokens * MOE_SLOTS, s.k]),
      upload(root, route, [s.tokens, 2 * MOE_SLOTS]),
      down.layer,
      upload(root, residual, [s.tokens, s.n]),
      s,
    ),
    ex,
  );
  const want = Float64Array.from(residual);
  for (let t = 0; t < s.tokens; t++) {
    for (let slot = 0; slot < MOE_SLOTS; slot++) {
      const e = route[t * 2 * MOE_SLOTS + slot]!;
      const gate = route[t * 2 * MOE_SLOTS + MOE_SLOTS + slot]!;
      const y = project(act, t * MOE_SLOTS + slot, down, e, s);
      for (let c = 0; c < s.n; c++) want[t * s.n + c] = want[t * s.n + c]! + gate * y[c]!;
    }
  }
  expectClose(got, want, { atol: 1e-5 * s.k, rtol: 1e-4 });
}

describe('moeQuantW kernels', () => {
  // One route each: a workgroup computes 8 units of 4 columns for one row,
  // with 32 lanes splitting K. bits, groupSize and the expert count are
  // pipeline config.
  const SHAPES = ([8, 4] as const).flatMap((bits) =>
    [
      { name: 'one tile, one expert', tokens: 1, k: 4, n: 8, experts: 1, groupSize: 4 },
      // 3 units leave 5 of the workgroup's 8 idle
      { name: 'fewer units than a workgroup', tokens: 3, k: 16, n: 24, experts: 4, groupSize: 8 },
      // 33 chunks of 4 rows: one lane takes two, the rest one
      { name: 'one K chunk past the lanes', tokens: 2, k: 132, n: 8, experts: 3, groupSize: 4 },
      { name: 'one group per expert', tokens: 2, k: 64, n: 72, experts: 5, groupSize: 64 },
      { name: 'more experts than slots', tokens: 5, k: 32, n: 16, experts: 8, groupSize: 16 },
    ].map((c) => ({ bits, ...c })),
  );
  it.for(SHAPES)('moeUp int$bits $name: T $tokens, K $k, N $n, E $experts', (c) => checkMoeUp(c));
  it.for(SHAPES)('moeDown int$bits $name: T $tokens, K $k, N $n, E $experts', (c) =>
    checkMoeDown(c),
  );

  // The grid is at most 65535 rows high; past it each workgroup loops.
  it('moeUp past 65535 rows of (token, slot)', async () => {
    const tokens = Math.ceil(MAX_WORKGROUPS_PER_DIM / MOE_SLOTS) + 1;
    await checkMoeUp({ tokens, k: 4, n: 8, experts: 2, bits: 8, groupSize: 4 });
  });

  it('moeDown past 65535 tokens', async () => {
    await checkMoeDown({
      tokens: MAX_WORKGROUPS_PER_DIM + 2,
      k: 4,
      n: 8,
      experts: 2,
      bits: 8,
      groupSize: 4,
    });
  });

  it('rejects an empty input with an INVALID_ARGUMENT', () => {
    expect(() => upload(root, new Float32Array(0), [0, 4])).toThrow(
      expect.objectContaining({ code: 'INVALID_ARGUMENT' }),
    );
  });

  const shape: Shape = { tokens: 2, k: 8, n: 8, experts: 2, bits: 8, groupSize: 4 };
  it('moeUp refuses a route that is not topk output', () => {
    const x = upload(root, new Float32Array(16), [2, 8]);
    const route = upload(root, new Float32Array(2 * 4), [2, 4]);
    const { layer } = expertLayer(shape, 10);
    expect(() => moeUp(x, route, layer, layer, shape)).toThrow(
      /route must be topk's f32 \[2, 8\] output/,
    );
  });

  it('moeUp refuses a groupSize that does not divide K', () => {
    const x = upload(root, new Float32Array(16), [2, 8]);
    const route = upload(root, routeFor(2, 2), [2, 8]);
    const { layer } = expertLayer(shape, 10);
    expect(() => moeUp(x, route, layer, layer, { bits: 8, groupSize: 3 })).toThrow(
      /groupSize 3 must be a multiple of 4 that divides K 8/,
    );
  });

  it('moeDown refuses act rows that are not 4 per token', () => {
    const act = upload(root, new Float32Array(6 * 8), [6, 8]);
    const route = upload(root, routeFor(2, 2), [2, 8]);
    const { layer } = expertLayer(shape, 10);
    const residual = upload(root, new Float32Array(16), [2, 8]);
    expect(() => moeDown(act, route, layer, residual, shape)).toThrow(
      /act rows 6 must be a multiple of 4/,
    );
  });

  it('moeDown refuses a residual of another shape', () => {
    const act = upload(root, new Float32Array(8 * 8), [8, 8]);
    const route = upload(root, routeFor(2, 2), [2, 8]);
    const { layer } = expertLayer(shape, 10);
    const residual = upload(root, new Float32Array(8), [1, 8]);
    expect(() => moeDown(act, route, layer, residual, shape)).toThrow(
      /residual must be f32 \[2, 8\]/,
    );
  });
});
