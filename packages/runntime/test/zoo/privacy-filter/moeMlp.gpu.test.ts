/** privacy-filter's MoE block against a CPU loop written from the formula:
 *  RMSNorm, router, top-4 softmax, clamped SwiGLU experts, residual. */
import { describe, it } from 'vitest';
import { toArray } from '../../../src/core/gpu/eval.ts';
import { EagerMoeMlp } from '../../../src/zoo/privacy-filter/moeMlp.ts';
import { randomQuantWeight } from '../../core/kernels/matmul/quantW.ts';
import { expectClose, gpu, randomF32, requireFeature, roundF16, upload } from '../../gpu.ts';

const { root } = await gpu();
const E = 8;
const H = 64;
const T = 5;
const G = 16;

interface Experts {
  glu: ArrayLike<number>;
  lin: ArrayLike<number>;
  out: ArrayLike<number>;
}

const clampedSwiglu = (glu: number, lin: number) => {
  const g = Math.min(glu, 7);
  const l = Math.min(Math.max(lin, -7), 7);
  return (g / (1 + Math.exp(-1.702 * g))) * (l + 1);
};

function reference(
  x: Float32Array,
  scale: Float32Array,
  gateW: Float32Array,
  gateB: Float32Array,
  w: Experts,
  b: Experts,
): Float64Array {
  const y = new Float64Array(T * H);
  for (let t = 0; t < T; t++) {
    const row = x.subarray(t * H, (t + 1) * H);
    const ms = row.reduce((s, v) => s + v * v, 0) / H;
    const xn = Array.from(row, (v, i) => (v / Math.sqrt(ms + 1e-5)) * scale[i]!);
    const logits = Array.from({ length: E }, (_, e) => {
      let s = gateB[e]!;
      for (let i = 0; i < H; i++) s += xn[i]! * gateW[i * E + e]!;
      return s;
    });
    const ids = [...logits.keys()].sort((a, c) => logits[c]! - logits[a]! || a - c).slice(0, 4);
    const top = ids.map((e) => logits[e]!);
    const max = Math.max(...top);
    const exps = top.map((v) => Math.exp(v - max));
    const sum = exps.reduce((a, c) => a + c, 0);
    ids.forEach((e, slot) => {
      const act = new Float64Array(H);
      for (let n = 0; n < H; n++) {
        let glu = b.glu[e * H + n]!;
        let lin = b.lin[e * H + n]!;
        for (let k = 0; k < H; k++) {
          glu += xn[k]! * w.glu[(e * H + k) * H + n]!;
          lin += xn[k]! * w.lin[(e * H + k) * H + n]!;
        }
        act[n] = clampedSwiglu(glu, lin);
      }
      for (let n = 0; n < H; n++) {
        let o = b.out[e * H + n]!;
        for (let k = 0; k < H; k++) o += act[k]! * w.out[(e * H + k) * H + n]!;
        y[t * H + n]! += (exps[slot]! / sum) * o;
      }
    });
    for (let n = 0; n < H; n++) y[t * H + n]! += row[n]!;
  }
  return y;
}

describe('privacy-filter EagerMoeMlp', () => {
  it.for(['f32', 'f16', 'quantInt8', 'quantInt4'] as const)('%s', async (format, ctx) => {
    const half = format === 'f16';
    if (half) requireFeature(ctx, root, 'shader-f16');
    const mlp = new EagerMoeMlp({
      experts: E,
      hidden: H,
      weightFormat: half ? 'f32' : format,
      groupSize: G,
    });
    if (half) mlp.half(root);
    const dtype = half ? 'f16' : 'f32';
    const round = (a: Float32Array) => (half ? roundF16(a) : a);
    const up = (data: Float32Array, dims: number[]) =>
      upload(root, data, { elems: data.length, dtype, dims });
    const x = round(randomF32(T * H, 1));
    const scale = round(randomF32(H, 2).map((v) => 1 + 0.2 * v));
    const gateW = round(randomF32(H * E, 3));
    const gateB = round(randomF32(E, 4).map((v) => 0.1 * v));
    const b = {
      glu: round(randomF32(E * H, 5).map((v) => 0.1 * v)),
      lin: round(randomF32(E * H, 6).map((v) => 0.1 * v)),
      out: round(randomF32(E * H, 7).map((v) => 0.1 * v)),
    };
    mlp.norm.scale.bind(up(scale, [H]));
    mlp.gate.weight.bind(up(gateW, [H, E]));
    mlp.gate.bias!.bind(up(gateB, [E]));
    mlp.gluBias.bind(up(b.glu, [E, H]));
    mlp.linBias.bind(up(b.lin, [E, H]));
    mlp.outBias.bind(up(b.out, [E, H]));
    let w: Experts;
    if (format === 'f32' || half) {
      const make = (seed: number) => round(randomF32(E * H * H, seed).map((v) => 0.2 * v));
      const [glu, lin, out] = [make(8), make(9), make(10)];
      mlp.gluWeight.bind(up(glu, [E * H, H]));
      mlp.linWeight.bind(up(lin, [E * H, H]));
      mlp.outWeight.bind(up(out, [E * H, H]));
      w = { glu, lin, out };
    } else {
      const bits = format === 'quantInt8' ? 8 : 4;
      const make = (seed: number) =>
        randomQuantWeight(root, { experts: E, k: H, n: H, bits, groupSize: G }, seed);
      const [glu, lin, out] = [make(8), make(9), make(10)];
      const scaleShape = [(E * H) / G, H] as const;
      mlp.gluWeight.bind(glu.w);
      mlp.linWeight.bind(lin.w);
      mlp.outWeight.bind(out.w);
      mlp.gluScales!.bind(upload(root, glu.scales, scaleShape));
      mlp.linScales!.bind(upload(root, lin.scales, scaleShape));
      mlp.outScales!.bind(upload(root, out.scales, scaleShape));
      w = { glu: glu.dequant, lin: lin.dequant, out: out.dequant };
    }
    const got = await toArray(mlp.forward(up(x, [T, H])));
    const tol = half ? 3e-2 : 2e-3;
    expectClose(got, reference(x, scale, gateW, gateB, w, b), { atol: tol, rtol: tol });
  });
});
