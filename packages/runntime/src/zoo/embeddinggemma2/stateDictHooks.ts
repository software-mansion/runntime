/** Maps the HF EmbeddingGemma2Model checkpoint onto the text model and the
 *  vision tower as lazy derived tensors; nothing is fetched here. Each
 *  keeps only its own tensors, so one checkpoint with every tower loads
 *  into either. */

import { derivedTensor, type LazyStateDict, type LazyTensor } from '../../core/index.ts';
import { checkpointMismatch } from '../errors.ts';
import type { EmbeddingGemma2Config, EmbeddingGemma2VisionConfig } from './config.ts';

const SOURCE = 'language_model.';
const VISION_SOURCES = ['vision_tower.', 'embed_vision.'] as const;

/** Row r of each d-row head moves so (p, p + d/2) become (2p, 2p + 1). */
function interleaveHeads(src: Float32Array, rows: number, cols: number, d: number): Float32Array {
  const out = new Float32Array(src.length);
  for (let r = 0; r < rows; r++) {
    const head = r - (r % d);
    const i = r % d;
    const to = head + (i < d / 2 ? 2 * i : 2 * (i - d / 2) + 1);
    out.set(src.subarray(r * cols, (r + 1) * cols), to * cols);
  }
  return out;
}

/** Each layer's RMSNorms, renamed `weight` → `scale`. */
const LAYER_NORMS = [
  'input_layernorm',
  'post_attention_layernorm',
  'pre_feedforward_layernorm',
  'post_feedforward_layernorm',
] as const;

/** Moves the tensors under `sources` out of `sd`. take() throws for a
 *  missing one, done() for any left unused. */
function remap(sd: LazyStateDict, prefix: string, sources: readonly string[], what: string) {
  const src = new Map<string, LazyTensor>();
  for (const [key, t] of sd.tensors) {
    if (sources.some((s) => key.startsWith(s))) src.set(key, t);
  }
  sd.tensors.clear();
  return {
    take(name: string): LazyTensor {
      const t = src.get(name);
      if (!t) throw checkpointMismatch(what, `missing tensor '${name}'`);
      src.delete(name);
      return t;
    },
    put(name: string, t: LazyTensor): void {
      sd.tensors.set(prefix ? `${prefix}.${name}` : name, t);
    },
    done(): void {
      if (src.size)
        throw checkpointMismatch(what, `unused tensors: ${[...src.keys()].slice(0, 5)}`);
    },
  };
}

/** q, k and v in one weight. q and k rows go from the checkpoint's
 *  half-split rope layout to the kernel's interleaved pairs, per `block`
 *  rows. */
function fusedQkv(
  wq: LazyTensor,
  wk: LazyTensor,
  wv: LazyTensor,
  qRows: number,
  kvRows: number,
  hidden: number,
  block: number,
): LazyTensor {
  const width = qRows + 2 * kvRows;
  return derivedTensor(
    [width, hidden],
    async () => {
      const out = new Float32Array(width * hidden);
      out.set(interleaveHeads(await wq.f32(), qRows, hidden, block), 0);
      out.set(interleaveHeads(await wk.f32(), kvRows, hidden, block), qRows * hidden);
      out.set(await wv.f32(), (qRows + kvRows) * hidden);
      return out;
    },
    wq.byteLength + wk.byteLength + wv.byteLength,
  );
}

/** q_norm and k_norm tiled over their heads and interleaved like
 *  fusedQkv, ones for v. */
function fusedQkvNorm(
  nq: LazyTensor,
  nk: LazyTensor,
  heads: number,
  kvHeads: number,
  d: number,
  block: number,
): LazyTensor {
  const width = (heads + 2 * kvHeads) * d;
  return derivedTensor(
    [width],
    async () => {
      const out = new Float32Array(width).fill(1);
      const qn = interleaveHeads(await nq.f32(), d, 1, block);
      const kn = interleaveHeads(await nk.f32(), d, 1, block);
      for (let h = 0; h < heads; h++) out.set(qn, h * d);
      for (let h = 0; h < kvHeads; h++) out.set(kn, (heads + h) * d);
      return out;
    },
    nq.byteLength + nk.byteLength,
  );
}

/** The text model's mapping. PLE's hidden^-½ folds into its projection. */
export function transformEmbeddingGemma2StateDict(
  sd: LazyStateDict,
  prefix: string,
  cfg: EmbeddingGemma2Config,
): void {
  const io = remap(sd, prefix, [SOURCE], 'embeddinggemma2 state dict');
  const take = (name: string) => io.take(SOURCE + name);
  const { put } = io;

  for (const [i, type] of cfg.layerTypes.entries()) {
    const L = `layers.${i}`;
    const { headDim: d, kvHeads } = cfg.attention[type];
    const attn = (n: string) => take(`${L}.self_attn.${n}.weight`);
    put(
      `${L}.qkv.weight`,
      fusedQkv(
        attn('q_proj'),
        attn('k_proj'),
        attn('v_proj'),
        cfg.heads * d,
        kvHeads * d,
        cfg.hidden,
        d,
      ),
    );
    put(`${L}.qkv_norm`, fusedQkvNorm(attn('q_norm'), attn('k_norm'), cfg.heads, kvHeads, d, d));
    put(`${L}.o_proj.weight`, attn('o_proj'));
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
    for (const n of LAYER_NORMS) put(`${L}.${n}.scale`, take(`${L}.${n}.weight`));
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
  io.done();
}

/** The vision tower's mapping. Its 2D rope turns each half head on its
 *  own, so q and k interleave per half head. The position table splits
 *  into x and y, and the weightless norm gets ones. */
export function transformEmbeddingGemma2VisionStateDict(
  sd: LazyStateDict,
  prefix: string,
  cfg: EmbeddingGemma2VisionConfig,
): void {
  const { take, put, done } = remap(
    sd,
    prefix,
    VISION_SOURCES,
    'embeddinggemma2 vision state dict',
  );
  const { hidden, heads, headDim: d, positions } = cfg;

  put('input_proj.weight', take('vision_tower.patch_embedder.input_proj.weight'));
  const table = take('vision_tower.patch_embedder.position_embedding_table');
  // [2, positions, hidden], read once for both axes.
  let both: Promise<Float32Array> | undefined;
  const axis = (i: number) =>
    derivedTensor(
      [positions, hidden],
      async () =>
        (await (both ??= table.f32())).slice(i * positions * hidden, (i + 1) * positions * hidden),
      table.byteLength / 2,
    );
  put('pos_x.weight', axis(0));
  put('pos_y.weight', axis(1));

  for (let i = 0; i < cfg.layers; i++) {
    const S = `vision_tower.encoder.layers.${i}`;
    const L = `layers.${i}`;
    const attn = (n: string) => take(`${S}.self_attn.${n}.weight`);
    const proj = (n: string) => take(`${S}.self_attn.${n}.linear.weight`);
    const rows = heads * d;
    put(
      `${L}.qkv.weight`,
      fusedQkv(proj('q_proj'), proj('k_proj'), proj('v_proj'), rows, rows, hidden, d / 2),
    );
    put(`${L}.qkv_norm`, fusedQkvNorm(attn('q_norm'), attn('k_norm'), heads, heads, d, d / 2));
    put(`${L}.o_proj.weight`, proj('o_proj'));
    for (const n of ['gate_proj', 'up_proj', 'down_proj']) {
      put(`${L}.${n}.weight`, take(`${S}.mlp.${n}.linear.weight`));
    }
    for (const n of LAYER_NORMS) put(`${L}.${n}.scale`, take(`${S}.${n}.weight`));
  }
  put(
    'embedding_pre_projection_norm.scale',
    derivedTensor([hidden], () => Promise.resolve(new Float32Array(hidden).fill(1)), 0),
  );
  put('embedding_projection.weight', take('embed_vision.embedding_projection.weight'));
  done();
}
