/** Derives what the HF BertModel checkpoint stores differently from the
 *  model's parameters, as lazy tensors; nothing is fetched here. Every other
 *  tensor binds under its own name, since the modules are named like the
 *  checkpoint. The pooler and `position_ids`, which the model never reads,
 *  are dropped.
 *
 *  The constant token-type row folds into the position table, and the score
 *  scale splits across the query and key projections, so the scale-free
 *  kernel still computes q·kᵀ/√headDim.
 *
 *  Weights stay in torch layout for the core Linear hook to transpose. The
 *  scaled query and key and the position sum are f32 even from an F16
 *  checkpoint, so an f32 model loads exact values and an f16 model rounds
 *  once in the loader. Derived byteLengths carry the source file bytes, so
 *  load progress counts every byte the model reads. */

import { derive, scaleWeight, type LazyStateDict, type LazyTensor } from '../../core/index.ts';
import { checkpointMismatch } from '../errors.ts';
import type { MinilmConfig } from './config.ts';

const qualify = (prefix: string, name: string): string => (prefix ? `${prefix}.${name}` : name);

const WHAT = 'minilm state dict';

function take(sd: LazyStateDict, name: string): LazyTensor {
  const t = sd.tensors.get(name);
  if (!t) throw checkpointMismatch(WHAT, `missing tensor '${name}'`);
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

  for (const unread of ['pooler.dense.weight', 'pooler.dense.bias', 'embeddings.position_ids']) {
    sd.tensors.delete(q(unread));
  }

  // position' = position + tokenType[0] (every real input is token type 0).
  const positions = q('embeddings.position_embeddings.weight');
  const position = take(sd, positions);
  const tokenType = take(sd, q('embeddings.token_type_embeddings.weight'));
  sd.tensors.set(
    positions,
    derive(position.shape, [position, tokenType], ([pos, type]) => {
      const out = new Float32Array(pos.length);
      for (let i = 0; i < pos.length; i++) out[i] = pos[i]! + type[i % cfg.hidden]!;
      return out;
    }),
  );

  for (let i = 0; i < cfg.layers; i++) {
    const self = q(`encoder.layer.${i}.attention.self`);
    for (const part of ['query.weight', 'query.bias', 'key.weight', 'key.bias']) {
      const name = `${self}.${part}`;
      sd.tensors.set(name, scaleWeight(take(sd, name), qkScale));
    }
  }
}
