/** Derives what the HF EmbeddingGemma2Model checkpoint stores differently
 *  from the text model's and the vision and audio towers' parameters, as
 *  lazy tensors; nothing is fetched here. Every other tensor binds under its
 *  own name, since the modules are named like the checkpoint.
 *
 *  An F16 checkpoint gives f16 derived tensors, except the text layer
 *  scalars and the audio tower's clip bounds, distance bias rows and q
 *  scales, which the CPU code and the attention kernel read as f32, and the
 *  scaled PLE projection and audio post norms, which stay f32 so an f32
 *  model loads exact values. */

import {
  concatRows,
  derive,
  f32ArrayToF16Bits,
  inTorchShape,
  reshapeWeight,
  scaleWeight,
  splitRows,
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

/** A weightless norm's ones, as f16 halfwords: exact, and stacked with an F16
 *  checkpoint's norms they keep the result f16. */
const ones = (n: number) =>
  derive([n], [], () => f32ArrayToF16Bits(new Float32Array(n).fill(1)).data);

/** q_norm and k_norm tiled over their heads, ones for v. */
function fusedQkvNorm(nq: LazyTensor, nk: LazyTensor, heads: number, kvHeads: number): LazyTensor {
  const d = nq.shape[0]!;
  const tiled = concatRows([
    ...Array<LazyTensor>(heads).fill(nq),
    ...Array<LazyTensor>(kvHeads).fill(nk),
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
    const { kvHeads } = cfg.attention[type];
    const attn = (n: string) => take(sd, what, q(`${L}.self_attn.${n}.weight`));
    put(
      `${L}.self_attn.qkv_norm`,
      fusedQkvNorm(attn('q_norm'), attn('k_norm'), cfg.heads, kvHeads),
    );
    const s = take(sd, what, q(`${L}.layer_scalar`));
    // Derived, so the CPU parameter binds f32 rather than raw bf16 bits.
    put(
      `${L}.layer_scalar`,
      derive([1], [s], ([value]) => value),
    );
  }
  const ple = 'ple.per_layer_model_projection.weight';
  put(ple, scaleWeight(take(sd, what, q(ple)), cfg.hidden ** -0.5));
}

/** The vision tower's mapping. The position table splits into x and y. */
export function transformEmbeddingGemma2VisionStateDict(
  sd: LazyStateDict,
  prefix: string,
  cfg: EmbeddingGemma2VisionConfig,
): void {
  const what = 'embeddinggemma2 vision state dict';
  const q = (name: string) => qualify(prefix, `vision_tower.${name}`);
  const put = (name: string, t: LazyTensor) => sd.tensors.set(q(name), t);
  const { hidden, heads, positions } = cfg;

  // [2, positions, hidden]: the x table, then the y table.
  const table = take(sd, what, q('patch_embedder.position_embedding_table'));
  const [x, y] = splitRows(inTorchShape(table, [2, positions, hidden]), [1, 1]);
  put('patch_embedder.pos_x.weight', reshapeWeight(x!, [positions, hidden]));
  put('patch_embedder.pos_y.weight', reshapeWeight(y!, [positions, hidden]));

  for (let i = 0; i < cfg.layers; i++) {
    const A = `encoder.layers.${i}.self_attn`;
    const attn = (n: string) => take(sd, what, q(`${A}.${n}.weight`));
    put(`${A}.qkv_norm`, fusedQkvNorm(attn('q_norm'), attn('k_norm'), heads, heads));
  }
}

/** The audio tower's mapping. q, k and v run as one matmul, so they must
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
  const { hidden, headDim, window } = cfg;
  const BOUNDS = ['input_min', 'input_max', 'output_min', 'output_max'];
  /** Rebuilds a clipped linear's bounds as f32: the CPU parameters they bind
   *  to would hold a BF16 bound as raw bits. */
  const f32Bounds = (linear: string) => {
    for (const b of BOUNDS) {
      const t = take(sd, what, q(`${linear}.${b}`));
      put(
        `${linear}.${b}`,
        derive([1], [t], ([value]) => value),
      );
    }
  };

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
      f32Bounds(`${L}.${ff}.ffw_layer_1`);
      f32Bounds(`${L}.${ff}.ffw_layer_2`);
    }

    const A = `${L}.self_attn`;
    // The fused projection clamps with q's bounds, so all three must agree.
    const parts = ['q_proj', 'k_proj', 'v_proj'];
    for (const b of BOUNDS) {
      const bounds = parts.map((n) => take(sd, what, q(`${A}.${n}.${b}`)));
      for (const [j, n] of parts.entries()) {
        put(
          `${A}.${n}.${b}`,
          derive(
            [1],
            bounds,
            (each) => {
              const values = each.map((bound) => bound[0]!);
              if (values.some((v) => v !== values[0])) {
                throw checkpointMismatch(what, `${q(A)}: q, k and v have different ${b}`);
              }
              return Float32Array.of(values[0]!);
            },
            bounds[j]!.byteLength,
          ),
        );
      }
    }
    f32Bounds(`${A}.post`);
    // rel and q_scale stay f32 on purpose: the attention kernel reads them as
    // f32, so rounding them to f16 would only lose precision.
    const rk = take(sd, what, q(`${A}.relative_k_proj.weight`));
    put(
      `${A}.rel`,
      derive([window, hidden], [rk], ([w]) => {
        const out = new Float32Array(window * hidden);
        for (let j = 0; j < window; j++) {
          for (let o = 0; o < hidden; o++) {
            let s = 0;
            for (let n = 0; n < hidden; n++) s += pe[j * hidden + n]! * w[o * hidden + n]!;
            out[j * hidden + o] = s;
          }
        }
        return out;
      }),
    );
    const dimScale = take(sd, what, q(`${A}.per_dim_scale`));
    put(
      `${A}.q_scale`,
      derive([headDim], [dimScale], ([scale]) =>
        scale.map((x) => qScale * Math.log1p(Math.exp(x))),
      ),
    );

    f32Bounds(`${L}.lconv1d.linear_start`);
    f32Bounds(`${L}.lconv1d.linear_end`);
  }
}
