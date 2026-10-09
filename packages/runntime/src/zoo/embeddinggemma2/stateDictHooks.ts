/** Maps the HF EmbeddingGemma2Model checkpoint onto the text model as lazy
 *  derived tensors; nothing is fetched here.
 *
 *  Keeps only `language_model.*` (the vision and audio towers go), renames
 *  RMSNorm `weight` to `scale`, folds PLE's hidden^-½ into its projection,
 *  and flattens each layer's attention, MLP and PLE blocks into the layer.
 *  q, k and v fuse into one weight, q and k rows permuted from HF's
 *  half-split rope layout to the kernel's interleaved pairs, and q_norm,
 *  k_norm and v_norm's ones fuse into one tiled norm weight. */

import { derivedTensor, type LazyStateDict, type LazyTensor } from '../../core/index.ts';
import { checkpointMismatch } from '../errors.ts';
import { ATTN, type EmbeddingGemma2Config } from './model.ts';

const SOURCE = 'language_model.';

/** Row r of each d-row head moves so (p, p + d/2) become (2p, 2p + 1). */
export function interleaveHeads(
  src: Float32Array,
  rows: number,
  cols: number,
  d: number,
): Float32Array {
  const out = new Float32Array(src.length);
  for (let r = 0; r < rows; r++) {
    const head = r - (r % d);
    const i = r % d;
    const to = head + (i < d / 2 ? 2 * i : 2 * (i - d / 2) + 1);
    out.set(src.subarray(r * cols, (r + 1) * cols), to * cols);
  }
  return out;
}

export function transformEmbeddingGemma2StateDict(
  sd: LazyStateDict,
  prefix: string,
  cfg: EmbeddingGemma2Config,
): void {
  const q = (name: string) => (prefix ? `${prefix}.${name}` : name);
  const src = new Map<string, LazyTensor>();
  for (const [key, t] of sd.tensors)
    if (key.startsWith(SOURCE)) src.set(key.slice(SOURCE.length), t);
  sd.tensors.clear();
  const take = (name: string): LazyTensor => {
    const t = src.get(name);
    if (!t) throw checkpointMismatch('embeddinggemma2 state dict', `missing tensor '${name}'`);
    src.delete(name);
    return t;
  };
  const put = (name: string, t: LazyTensor) => sd.tensors.set(q(name), t);

  for (let i = 0; i < cfg.layers; i++) {
    const L = `layers.${i}`;
    const { headDim: d, kvHeads } = ATTN[cfg.layerTypes[i]!];
    const wq = take(`${L}.self_attn.q_proj.weight`);
    const wk = take(`${L}.self_attn.k_proj.weight`);
    const wv = take(`${L}.self_attn.v_proj.weight`);
    const nq = take(`${L}.self_attn.q_norm.weight`);
    const nk = take(`${L}.self_attn.k_norm.weight`);
    const qRows = cfg.heads * d;
    const kvRows = kvHeads * d;
    const width = qRows + 2 * kvRows;
    put(
      `${L}.qkv.weight`,
      derivedTensor(
        [width, cfg.hidden],
        async () => {
          const out = new Float32Array(width * cfg.hidden);
          out.set(interleaveHeads(await wq.f32(), qRows, cfg.hidden, d), 0);
          out.set(interleaveHeads(await wk.f32(), kvRows, cfg.hidden, d), qRows * cfg.hidden);
          out.set(await wv.f32(), (qRows + kvRows) * cfg.hidden);
          return out;
        },
        wq.byteLength + wk.byteLength + wv.byteLength,
      ),
    );
    put(
      `${L}.qkv_norm`,
      derivedTensor(
        [width],
        async () => {
          const out = new Float32Array(width).fill(1);
          const qn = interleaveHeads(await nq.f32(), d, 1, d);
          const kn = interleaveHeads(await nk.f32(), d, 1, d);
          for (let h = 0; h < cfg.heads; h++) out.set(qn, h * d);
          for (let h = 0; h < kvHeads; h++) out.set(kn, qRows + h * d);
          return out;
        },
        nq.byteLength + nk.byteLength,
      ),
    );
    put(`${L}.o_proj.weight`, take(`${L}.self_attn.o_proj.weight`));
    for (const n of ['gate_proj', 'up_proj', 'down_proj']) {
      put(`${L}.${n}.weight`, take(`${L}.mlp.${n}.weight`));
    }
    for (const n of ['per_layer_input_gate', 'per_layer_projection']) {
      put(`${L}.${n}.weight`, take(`${L}.ple_block.${n}.weight`));
    }
    put(
      `${L}.post_per_layer_input_norm.scale`,
      take(`${L}.ple_block.post_per_layer_input_norm.weight`),
    );
    for (const n of [
      'input_layernorm',
      'post_attention_layernorm',
      'pre_feedforward_layernorm',
      'post_feedforward_layernorm',
    ]) {
      put(`${L}.${n}.scale`, take(`${L}.${n}.weight`));
    }
    const s = take(`${L}.layer_scalar`);
    // Derived, so the CPU parameter binds f32 rather than raw bf16 bits.
    put(
      `${L}.layer_scalar`,
      derivedTensor([1], () => s.f32(), s.byteLength),
    );
  }
  const p = take('ple.per_layer_model_projection.weight');
  const scale = cfg.hidden ** -0.5;
  put(
    'ple.per_layer_model_projection.weight',
    derivedTensor(p.shape, async () => (await p.f32()).map((v) => v * scale), p.byteLength),
  );
  put('ple.per_layer_projection_norm.scale', take('ple.per_layer_projection_norm.weight'));
  put('embed_tokens.weight', take('embed_tokens.weight'));
  put('norm.scale', take('norm.weight'));
  put('embedding_projection.weight', take('embedding_projection.weight'));
  if (src.size) {
    throw checkpointMismatch(
      'embeddinggemma2 state dict',
      `unused tensors: ${[...src.keys()].slice(0, 5)}`,
    );
  }
}
