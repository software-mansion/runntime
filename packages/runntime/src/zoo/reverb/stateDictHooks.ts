/** Maps a Reverb release checkpoint onto ReverbModel as lazy derived tensors;
 *  nothing is fetched here.
 *
 *  q and k rows go from the checkpoint's half-split rope layout to the
 *  kernel's interleaved pairs and fuse into one projection, as do up and
 *  gate. q_norm takes 1/√d and is tiled over the q heads, k_norm over the kv
 *  heads. The audio heads split into codebooks 0–1 and 2–7, each codebook's
 *  rows centered and each half padded with zero rows to a multiple of 4. */

import {
  concatRows,
  derive,
  floatTensor,
  permuteRows,
  scaleWeight,
  type LazyStateDict,
  type LazyTensor,
} from '../../core/index.ts';
import { checkpointMismatch } from '../errors.ts';
import { remap, ropeOrder } from '../shared/stateDictHooks.ts';
import { COARSE_CODEBOOKS, type ReverbConfig } from './config.ts';

const WHAT = 'reverb state dict';
const SOURCES = ['llm.', 'audio_embeddings.', 'audio_heads.', 'codebook_layer_offsets'] as const;

/** Rows of a head over `books` codebooks, padded to a multiple of 4 for the
 *  vec4 matmul. */
export const headRows = (books: number, codes: number) => Math.ceil((books * codes) / 4) * 4;

/** Codebooks [from, from + books) of the heads [codebooks·codes, hidden],
 *  each centered over its codes, then zero rows up to headRows. */
function centeredHead(t: LazyTensor, from: number, books: number, cfg: ReverbConfig): LazyTensor {
  const { codes, hidden, codebooks } = cfg;
  const rows = headRows(books, codes);
  return derive(
    [rows, hidden],
    [floatTensor(WHAT, t)],
    ([src]) => {
      const out = new Float32Array(rows * hidden);
      const mean = new Float64Array(hidden);
      for (let b = 0; b < books; b++) {
        const base = (from + b) * codes * hidden;
        mean.fill(0);
        for (let c = 0; c < codes; c++) {
          for (let i = 0; i < hidden; i++) mean[i]! += src[base + c * hidden + i]!;
        }
        for (let i = 0; i < hidden; i++) mean[i]! /= codes;
        for (let c = 0; c < codes; c++) {
          const row = (b * codes + c) * hidden;
          for (let i = 0; i < hidden; i++) out[row + i] = src[base + c * hidden + i]! - mean[i]!;
        }
      }
      return out;
    },
    (t.byteLength * books) / codebooks,
  );
}

export function transformReverbStateDict(
  sd: LazyStateDict,
  prefix: string,
  cfg: ReverbConfig,
): void {
  const { take, put, done } = remap(sd, prefix, SOURCES, WHAT);
  const { heads, kvHeads, headDim: d, codebooks, codes, hidden } = cfg;

  put('text_table.weight', take('llm.embed_tokens.table.weight'));
  put('text_project.weight', take('llm.embed_tokens.project.weight'));
  put('audio_table.weight', take('audio_embeddings.weight'));
  // Each codebook's offset into audio_table is fixed at codebook·codes.
  take('codebook_layer_offsets');

  const audioHeads = take('audio_heads.weight');
  if (audioHeads.shape.join() !== `${codebooks * codes},${hidden}`) {
    throw checkpointMismatch(
      WHAT,
      `audio_heads.weight has shape [${audioHeads.shape.join(', ')}], expected [${codebooks * codes}, ${hidden}]`,
    );
  }
  put('heads_coarse.weight', centeredHead(audioHeads, 0, COARSE_CODEBOOKS, cfg));
  put(
    'heads_fine.weight',
    centeredHead(audioHeads, COARSE_CODEBOOKS, codebooks - COARSE_CODEBOOKS, cfg),
  );
  put('norm.weight', take('llm.norm.weight'));

  const rope = (w: LazyTensor) => permuteRows(w, ropeOrder(w.shape[0]!, d));
  for (let i = 0; i < cfg.layers; i++) {
    const S = `llm.layers.${i}`;
    const L = `layers.${i}`;
    const attn = (n: string) => take(`${S}.self_attn.${n}.weight`);
    const mlp = (n: string) => take(`${S}.mlp.${n}.weight`);
    put(`${L}.qk.weight`, concatRows([rope(attn('q_proj')), rope(attn('k_proj'))]));
    const qn = attn('q_norm');
    const kn = attn('k_norm');
    const q = scaleWeight(rope(qn), d ** -0.5);
    const k = rope(kn);
    const norms = concatRows([
      ...Array<LazyTensor>(heads).fill(q),
      ...Array<LazyTensor>(kvHeads).fill(k),
    ]);
    // concatRows reads a part listed several times once.
    put(`${L}.qk_norm`, { ...norms, byteLength: qn.byteLength + kn.byteLength });
    put(`${L}.v.weight`, attn('v_proj'));
    put(`${L}.o.weight`, attn('o_proj'));
    put(`${L}.up_gate.weight`, concatRows([mlp('up_proj'), mlp('gate_proj')]));
    put(`${L}.down.weight`, mlp('down_proj'));
    put(`${L}.input_norm.weight`, take(`${S}.input_layernorm.weight`));
    put(`${L}.post_norm.weight`, take(`${S}.post_attention_layernorm.weight`));
  }
  done();
}
