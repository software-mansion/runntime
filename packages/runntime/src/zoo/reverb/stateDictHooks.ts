/** Derives what a Reverb release checkpoint stores differently from
 *  ReverbModel's parameters, as lazy tensors; nothing is fetched here. Every
 *  other tensor binds under its own name, since the modules are named like
 *  the checkpoint. `codebook_layer_offsets`, which the model never reads, is
 *  dropped.
 *
 *  q and k rows go from the checkpoint's half-split rope layout to the
 *  kernel's interleaved pairs and fuse into one projection, as do up and
 *  gate. q_norm takes 1/√d and is tiled over the q heads, k_norm over the kv
 *  heads. The audio heads split into codebooks 0–1 and 2–7, each codebook's
 *  rows centered and each half padded with zero rows to a multiple of 4. */

import {
  concatRows,
  derive,
  permuteRows,
  scaleWeight,
  type LazyStateDict,
  type LazyTensor,
} from '../../core/index.ts';
import { checkpointMismatch } from '../errors.ts';
import { ropeOrder } from '../shared/rope.ts';
import { COARSE_CODEBOOKS, type ReverbConfig } from './config.ts';

const WHAT = 'reverb state dict';

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
    [t],
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

const qualify = (prefix: string, name: string): string => (prefix ? `${prefix}.${name}` : name);

export function transformReverbStateDict(
  sd: LazyStateDict,
  prefix: string,
  cfg: ReverbConfig,
): void {
  const q = (name: string) => qualify(prefix, name);
  const take = (name: string): LazyTensor => {
    const t = sd.tensors.get(q(name));
    if (!t) throw checkpointMismatch(WHAT, `missing tensor '${q(name)}'`);
    sd.tensors.delete(q(name));
    return t;
  };
  const put = (name: string, t: LazyTensor) => sd.tensors.set(q(name), t);
  const { heads, kvHeads, headDim: d, codebooks, codes, hidden } = cfg;

  // Each codebook's offset into audio_embeddings is fixed at codebook·codes.
  sd.tensors.delete(q('codebook_layer_offsets'));

  const audioHeads = take('audio_heads.weight');
  if (audioHeads.shape.join() !== `${codebooks * codes},${hidden}`) {
    throw checkpointMismatch(
      WHAT,
      `audio_heads.weight has shape [${audioHeads.shape.join(', ')}], expected [${codebooks * codes}, ${hidden}]`,
    );
  }
  put('audio_heads.coarse.weight', centeredHead(audioHeads, 0, COARSE_CODEBOOKS, cfg));
  put(
    'audio_heads.fine.weight',
    centeredHead(audioHeads, COARSE_CODEBOOKS, codebooks - COARSE_CODEBOOKS, cfg),
  );

  const rope = (w: LazyTensor) => permuteRows(w, ropeOrder(w.shape[0]!, d));
  for (let i = 0; i < cfg.layers; i++) {
    const attn = `llm.layers.${i}.self_attn`;
    const mlp = `llm.layers.${i}.mlp`;
    put(
      `${attn}.qk.weight`,
      concatRows([rope(take(`${attn}.q_proj.weight`)), rope(take(`${attn}.k_proj.weight`))]),
    );
    const qn = take(`${attn}.q_norm.weight`);
    const kn = take(`${attn}.k_norm.weight`);
    const qScaled = scaleWeight(rope(qn), d ** -0.5);
    const k = rope(kn);
    const norms = concatRows([
      ...Array<LazyTensor>(heads).fill(qScaled),
      ...Array<LazyTensor>(kvHeads).fill(k),
    ]);
    // concatRows reads a part listed several times once.
    put(`${attn}.qk_norm`, { ...norms, byteLength: qn.byteLength + kn.byteLength });
    put(
      `${mlp}.up_gate.weight`,
      concatRows([take(`${mlp}.up_proj.weight`), take(`${mlp}.gate_proj.weight`)]),
    );
  }
}
