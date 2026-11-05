/** Fuses the HF Moonshine checkpoint's attention projections as lazy derived
 *  tensors; nothing is fetched here. Every other tensor binds under its own
 *  name, since the modules are named like the checkpoint.
 *
 *  q/k/v row-stack into one fused weight, so the hot path is a single matmul.
 *  No rotary permutation is needed, since HF's convention already matches the
 *  kernel's. Cross-attention q absorbs the full score scale, having no rope.
 *
 *  Weights stay in torch layout for the core Linear hook to transpose. An F16
 *  checkpoint gives f16 derived tensors, except the scaled cross-attention q,
 *  which stays f32 so an f32 model loads exact values. Derived byteLengths
 *  carry the source file bytes, so load progress still sums to the
 *  checkpoint size. */

import { concatRows, scaleWeight, type LazyStateDict, type LazyTensor } from '../../core/index.ts';
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

/** Takes an attention projection weight, which must be [_, hidden]. */
function takeProjection(sd: LazyStateDict, name: string, hidden: number): LazyTensor {
  const t = take(sd, name);
  if (t.shape.length !== 2 || t.shape[1] !== hidden) {
    throw checkpointMismatch(
      'moonshine state dict',
      `'${name}' has shape [${t.shape.join(', ')}], expected [_, ${hidden}]`,
    );
  }
  return t;
}

/** Stacks `attn`'s q|k|v projections into its one torch-layout
 *  [3·hidden, hidden] `qkv.weight`, with q times `qScale` when given. */
export function fuseQkv(sd: LazyStateDict, attn: string, hidden: number, qScale?: number): void {
  const part = (p: string) => takeProjection(sd, `${attn}.${p}_proj.weight`, hidden);
  const query = part('q');
  sd.tensors.set(
    `${attn}.qkv.weight`,
    concatRows([qScale === undefined ? query : scaleWeight(query, qScale), part('k'), part('v')]),
  );
}

export function transformDecoder(
  sd: LazyStateDict,
  prefix: string,
  dec: { layers: number; hidden: number; headDim: number },
): void {
  const crossQScale = 1 / Math.sqrt(dec.headDim);
  for (let i = 0; i < dec.layers; i++) {
    const L = qualify(prefix, `model.decoder.layers.${i}`);
    fuseQkv(sd, `${L}.self_attn`, dec.hidden);
    const cross = `${L}.encoder_attn`;
    sd.tensors.set(
      `${cross}.q_proj.weight`,
      scaleWeight(takeProjection(sd, `${cross}.q_proj.weight`, dec.hidden), crossQScale),
    );
    sd.tensors.set(
      `${cross}.kv.weight`,
      concatRows(
        ['k', 'v'].map((p) => takeProjection(sd, `${cross}.${p}_proj.weight`, dec.hidden)),
      ),
    );
  }
}

export function transformMoonshineStateDict(
  sd: LazyStateDict,
  prefix: string,
  cfg: MoonshineConfig,
): void {
  for (let i = 0; i < cfg.encLayers; i++) {
    fuseQkv(sd, qualify(prefix, `model.encoder.layers.${i}.self_attn`), cfg.dModel);
  }
  transformDecoder(sd, prefix, { layers: cfg.decLayers, hidden: cfg.dModel, headDim: cfg.headDim });
}
