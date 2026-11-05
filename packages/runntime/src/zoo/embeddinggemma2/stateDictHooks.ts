/** Derives what the HF EmbeddingGemma2Model checkpoint stores differently
 *  from the text model's and the vision and audio towers' parameters, as
 *  lazy tensors; nothing is fetched here. Every other tensor binds under its
 *  own name, since the modules are named like the checkpoint.
 *
 *  An F16 checkpoint gives f16 derived tensors, except the text layer
 *  scalars and the audio tower's distance bias rows and q scales, which the
 *  CPU code and the attention kernel read as f32, and the scaled PLE
 *  projection and audio post norms, which stay f32 so an f32 model loads
 *  exact values. */

import {
  concatRows,
  derivedTensor,
  f32ArrayToF16Bits,
  permuteRows,
  reshapeWeight,
  scaleWeight,
  splitRows,
  transposeWeight,
  type LazyStateDict,
  type LazyTensor,
} from '../../core/index.ts';
import { checkpointMismatch } from '../errors.ts';
import type {
  EmbeddingGemma2AudioConfig,
  EmbeddingGemma2Config,
  EmbeddingGemma2VisionConfig,
} from './config.ts';

const qualify = (prefix: string, name: string): string => (prefix ? `${prefix}.${name}` : name);

/** Takes a tensor out of `sd`, for a derived one to replace. */
function take(sd: LazyStateDict, what: string, name: string): LazyTensor {
  const t = sd.tensors.get(name);
  if (!t) throw checkpointMismatch(what, `missing tensor '${name}'`);
  sd.tensors.delete(name);
  return t;
}

/** Row order for permuteRows that takes each `block` rows from the
 *  checkpoint's half-split rope layout to the kernel's interleaved pairs:
 *  rows (p, p + block/2) become (2p, 2p + 1). */
function ropeOrder(rows: number, block: number): number[] {
  return Array.from({ length: rows }, (_, r) => {
    const i = r % block;
    return r - i + (i % 2 ? (i - 1) / 2 + block / 2 : i / 2);
  });
}

/** A weightless norm's ones, as f16 halfwords: exact, and stacked with an F16
 *  checkpoint's norms they keep the result f16. */
const ones = (n: number) =>
  derivedTensor([n], () => Promise.resolve(f32ArrayToF16Bits(new Float32Array(n).fill(1)).data));

/** q, k and v in one weight. q and k rows go from the checkpoint's
 *  half-split rope layout to the kernel's interleaved pairs, per `block`
 *  rows. */
function fusedQkv(wq: LazyTensor, wk: LazyTensor, wv: LazyTensor, block: number): LazyTensor {
  const rope = (w: LazyTensor) => permuteRows(w, ropeOrder(w.shape[0]!, block));
  return concatRows([rope(wq), rope(wk), wv]);
}

/** q_norm and k_norm tiled over their heads and interleaved like
 *  fusedQkv, ones for v. */
function fusedQkvNorm(
  nq: LazyTensor,
  nk: LazyTensor,
  heads: number,
  kvHeads: number,
  block: number,
): LazyTensor {
  const d = nq.shape[0]!;
  const order = ropeOrder(d, block);
  const qn = permuteRows(nq, order);
  const kn = permuteRows(nk, order);
  const tiled = concatRows([
    ...Array<LazyTensor>(heads).fill(qn),
    ...Array<LazyTensor>(kvHeads).fill(kn),
    ones(kvHeads * d),
  ]);
  // concatRows reads a part listed several times once, so each norm is
  // downloaded once however many heads it is tiled over.
  return { ...tiled, byteLength: nq.byteLength + nk.byteLength };
}

/** The text model's mapping. PLE's hidden^-½ folds into its projection. */
export function transformEmbeddingGemma2StateDict(
  sd: LazyStateDict,
  prefix: string,
  cfg: EmbeddingGemma2Config,
): void {
  const what = 'embeddinggemma2 state dict';
  const q = (name: string) => qualify(prefix, `language_model.${name}`);
  const put = (name: string, t: LazyTensor) => sd.tensors.set(q(name), t);

  for (const [i, type] of cfg.layerTypes.entries()) {
    const L = `layers.${i}`;
    const { headDim: d, kvHeads } = cfg.attention[type];
    const attn = (n: string) => take(sd, what, q(`${L}.self_attn.${n}.weight`));
    put(`${L}.self_attn.qkv.weight`, fusedQkv(attn('q_proj'), attn('k_proj'), attn('v_proj'), d));
    put(
      `${L}.self_attn.qkv_norm`,
      fusedQkvNorm(attn('q_norm'), attn('k_norm'), cfg.heads, kvHeads, d),
    );
    const s = take(sd, what, q(`${L}.layer_scalar`));
    // Derived, so the CPU parameter binds f32 rather than raw bf16 bits.
    put(
      `${L}.layer_scalar`,
      derivedTensor([1], () => s.f32(), s.byteLength),
    );
  }
  const ple = 'ple.per_layer_model_projection.weight';
  put(ple, scaleWeight(take(sd, what, q(ple)), cfg.hidden ** -0.5));
}

/** The vision tower's mapping. Its 2D rope turns each half head on its
 *  own, so q and k interleave per half head. The position table splits
 *  into x and y. */
export function transformEmbeddingGemma2VisionStateDict(
  sd: LazyStateDict,
  prefix: string,
  cfg: EmbeddingGemma2VisionConfig,
): void {
  const what = 'embeddinggemma2 vision state dict';
  const q = (name: string) => qualify(prefix, `vision_tower.${name}`);
  const put = (name: string, t: LazyTensor) => sd.tensors.set(q(name), t);
  const { hidden, heads, headDim: d, positions } = cfg;

  // [2, positions, hidden]: the x table, then the y table.
  const [x, y] = splitRows(take(sd, what, q('patch_embedder.position_embedding_table')), [1, 1]);
  put('patch_embedder.pos_x.weight', reshapeWeight(x!, [positions, hidden]));
  put('patch_embedder.pos_y.weight', reshapeWeight(y!, [positions, hidden]));

  for (let i = 0; i < cfg.layers; i++) {
    const A = `encoder.layers.${i}.self_attn`;
    const attn = (n: string) => take(sd, what, q(`${A}.${n}.weight`));
    const proj = (n: string) => take(sd, what, q(`${A}.${n}.linear.weight`));
    put(`${A}.qkv.weight`, fusedQkv(proj('q_proj'), proj('k_proj'), proj('v_proj'), d / 2));
    put(`${A}.qkv_norm`, fusedQkvNorm(attn('q_norm'), attn('k_norm'), heads, heads, d / 2));
  }
}

/** The audio tower's mapping. The two conv weights take the taps-major
 *  layout the convs read as matmuls. q, k and v fuse into one weight and
 *  share their bounds. Each layer's distance bias rows and q column scales
 *  are computed here, and the feed-forward residual weight folds into its
 *  post norm. */
export function transformEmbeddingGemma2AudioStateDict(
  sd: LazyStateDict,
  prefix: string,
  cfg: EmbeddingGemma2AudioConfig,
): void {
  const what = 'embeddinggemma2 audio state dict';
  const q = (name: string) => qualify(prefix, `audio_tower.${name}`);
  const put = (name: string, t: LazyTensor) => sd.tensors.set(q(name), t);
  const { hidden, headDim, convKernel: K, window } = cfg;
  const [c0, c1] = cfg.convChannels;
  const scalar = async (t: LazyTensor) => (await t.f32())[0]!;
  const BOUNDS = ['input_min', 'input_max', 'output_min', 'output_max'];

  // The first conv has one input channel: [c0, 9] as a linear over 3×3
  // taps. The second reads its taps (row, column, channel) major, so each
  // output's [c0, 9] block becomes [9, c0], one element at a time.
  const sub = 'subsample_conv_projection.';
  const conv0 = `${sub}layer0.conv.weight`;
  put(conv0, reshapeWeight(take(sd, what, q(conv0)), [c0, 9]));
  const tapsMajor = Array.from({ length: c1 * 9 * c0 }, (_, j) => {
    const [o, k, c] = [Math.floor(j / (9 * c0)), Math.floor(j / c0) % 9, j % c0];
    return (o * c0 + c) * 9 + k;
  });
  const conv1 = `${sub}layer1.conv.weight`;
  const taps = reshapeWeight(take(sd, what, q(conv1)), [c1 * c0 * 9]);
  put(conv1, reshapeWeight(permuteRows(taps, tapsMajor), [c1, 9 * c0]));

  // Sinusoidal rows for distances 0 … window − 1: sines, then cosines.
  const half = hidden / 2;
  const pe = new Float32Array(window * hidden);
  for (let j = 0; j < window; j++) {
    for (let i = 0; i < half; i++) {
      const a = j * Math.exp((-i * Math.log(10000)) / (half - 1));
      pe[j * hidden + i] = Math.sin(a);
      pe[j * hidden + half + i] = Math.cos(a);
    }
  }
  // q's scale before the per-column softplus, fixed by the architecture.
  const qScale = headDim ** -0.5 / Math.LN2;

  for (let i = 0; i < cfg.layers; i++) {
    const L = `layers.${i}`;
    for (const ff of ['feed_forward1', 'feed_forward2']) {
      const post = `${L}.${ff}.post_layer_norm.weight`;
      put(post, scaleWeight(take(sd, what, q(post)), cfg.residualWeight));
    }

    const A = `${L}.self_attn`;
    const parts = ['q_proj', 'k_proj', 'v_proj'];
    put(
      `${A}.qkv.linear.weight`,
      concatRows(parts.map((n) => take(sd, what, q(`${A}.${n}.linear.weight`)))),
    );
    // One set of bounds for the fused projection, checked to agree.
    for (const b of BOUNDS) {
      const bounds = parts.map((n) => take(sd, what, q(`${A}.${n}.${b}`)));
      put(
        `${A}.qkv.${b}`,
        derivedTensor(
          [1],
          async () => {
            const values = await Promise.all(bounds.map(scalar));
            if (values.some((v) => v !== values[0])) {
              throw checkpointMismatch(what, `${q(A)}: q, k and v have different ${b}`);
            }
            return Float32Array.of(values[0]!);
          },
          bounds[0]!.byteLength,
        ),
      );
    }
    // rel and q_scale stay f32 on purpose: the attention kernel reads them as
    // f32, so rounding them to f16 would only lose precision.
    const rk = take(sd, what, q(`${A}.relative_k_proj.weight`));
    put(
      `${A}.rel`,
      derivedTensor(
        [window, hidden],
        async () => {
          const w = await rk.f32();
          const out = new Float32Array(window * hidden);
          for (let j = 0; j < window; j++) {
            for (let o = 0; o < hidden; o++) {
              let s = 0;
              for (let n = 0; n < hidden; n++) s += pe[j * hidden + n]! * w[o * hidden + n]!;
              out[j * hidden + o] = s;
            }
          }
          return out;
        },
        rk.byteLength,
      ),
    );
    const dimScale = take(sd, what, q(`${A}.per_dim_scale`));
    put(
      `${A}.q_scale`,
      derivedTensor(
        [headDim],
        async () => (await dimScale.f32()).map((x) => qScale * Math.log1p(Math.exp(x))),
        dimScale.byteLength,
      ),
    );

    // [hidden, 1, K] → [K, hidden].
    const conv = `${L}.lconv1d.depthwise_conv1d.weight`;
    put(conv, transposeWeight(reshapeWeight(take(sd, what, q(conv)), [hidden, K])));
  }
}
