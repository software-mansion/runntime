/** Derives what the HF BertModel checkpoint stores differently from the
 *  model's parameters, as lazy tensors; nothing is fetched here. Every other
 *  tensor binds under its own name, since the modules are named like the
 *  checkpoint; the pooler and `position_ids` are left unread.
 *
 *  The constant token-type row folds into the position table, and q/k/v fuse
 *  into one weight with the score scale split across the q and k parts, so
 *  the scale-free kernel still computes q·kᵀ/√headDim.
 *
 *  Weights stay in torch layout for the core Linear hook to transpose. The
 *  fused q/k/v and the position sum are f32 even from an F16 checkpoint, so
 *  an f32 model loads exact values and an f16 model rounds once in the
 *  loader. Derived byteLengths carry the source file bytes, so load progress
 *  still sums to the checkpoint size. */

import {
  concatRows,
  derivedTensor,
  scaleWeight,
  type LazyStateDict,
  type LazyTensor,
} from '../../core/index.ts';
import { checkpointMismatch } from '../errors.ts';
import type { MinilmConfig } from './config.ts';

const qualify = (prefix: string, name: string): string => (prefix ? `${prefix}.${name}` : name);

function take(sd: LazyStateDict, name: string): LazyTensor {
  const t = sd.tensors.get(name);
  if (!t) throw checkpointMismatch('minilm state dict', `missing tensor '${name}'`);
  sd.tensors.delete(name);
  return t;
}

export function transformMinilmStateDict(
  sd: LazyStateDict,
  prefix: string,
  cfg: MinilmConfig,
): void {
  const q = (name: string) => qualify(prefix, name);
  // qkScale on q and k each — their product is the 1/√headDim score scale.
  const qkScale = cfg.headDim ** -0.25;

  // position' = position + tokenType[0] (every real input is token type 0).
  const positions = q('embeddings.position_embeddings.weight');
  const position = take(sd, positions);
  const tokenType = take(sd, q('embeddings.token_type_embeddings.weight'));
  sd.tensors.set(
    positions,
    derivedTensor(
      position.shape,
      async () => {
        const pos = await position.f32();
        const type = await tokenType.f32();
        const out = new Float32Array(pos.length);
        for (let i = 0; i < pos.length; i++) out[i] = pos[i]! + type[i % cfg.hidden]!;
        return out;
      },
      position.byteLength + tokenType.byteLength,
    ),
  );

  for (let i = 0; i < cfg.layers; i++) {
    const self = q(`encoder.layer.${i}.attention.self`);
    // query|key|value → one torch-layout [3·hidden, hidden] weight and
    // [3·hidden] bias: the three parts stacked row-wise, q/k pre-scaled.
    for (const suffix of ['weight', 'bias']) {
      const part = (p: string) => take(sd, `${self}.${p}.${suffix}`);
      sd.tensors.set(
        `${self}.qkv.${suffix}`,
        concatRows([
          scaleWeight(part('query'), qkScale),
          scaleWeight(part('key'), qkScale),
          part('value'),
        ]),
      );
    }
  }
}
