/** Folds the score scale into the HF Moonshine checkpoint's cross-attention
 *  q as a lazy derived tensor; nothing is fetched here. Every other tensor
 *  binds under its own name, since the modules are named like the checkpoint,
 *  except the tied `proj_out.weight`, which the model never reads and is
 *  dropped.
 *
 *  No rotary permutation is needed, since HF's convention already matches the
 *  kernel's. Cross-attention q absorbs the full score scale, having no rope.
 *
 *  Weights stay in torch layout for the core Linear hook to transpose. The
 *  scaled q stays f32 even from an F16 checkpoint, so an f32 model loads
 *  exact values. */

import { scaleWeight, type LazyStateDict, type LazyTensor } from '../../core/index.ts';
import { checkpointMismatch } from '../errors.ts';
import type { MoonshineConfig } from './config.ts';

export const qualify = (prefix: string, name: string): string =>
  prefix ? `${prefix}.${name}` : name;

export function take(sd: LazyStateDict, name: string): LazyTensor {
  const t = sd.tensors.get(name);
  if (!t) throw checkpointMismatch('moonshine state dict', `missing tensor '${name}'`);
  sd.tensors.delete(name);
  return t;
}

/** Multiplies the `q_proj.weight` of the attention at `attn` by the score
 *  scale 1/√headDim. */
export function scaleQuery(sd: LazyStateDict, attn: string, headDim: number): void {
  const name = `${attn}.q_proj.weight`;
  sd.tensors.set(name, scaleWeight(take(sd, name), 1 / Math.sqrt(headDim)));
}

export function transformDecoder(
  sd: LazyStateDict,
  prefix: string,
  dec: { layers: number; headDim: number },
): void {
  // The LM head is the embedding weight, so the tied copy is never read.
  sd.tensors.delete(qualify(prefix, 'proj_out.weight'));
  for (let i = 0; i < dec.layers; i++) {
    scaleQuery(sd, qualify(prefix, `model.decoder.layers.${i}.encoder_attn`), dec.headDim);
  }
}

export function transformMoonshineStateDict(
  sd: LazyStateDict,
  prefix: string,
  cfg: MoonshineConfig,
): void {
  transformDecoder(sd, prefix, { layers: cfg.decLayers, headDim: cfg.headDim });
}
