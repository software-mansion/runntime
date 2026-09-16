/** Maps the HF BertModel checkpoint onto the model's parameter tree as lazy
 *  derived tensors; nothing is fetched here.
 *
 *  Drops the unused pooler keys, folds the constant token-type row into the
 *  position table, and fuses q/k/v into one weight with the score scale split
 *  across the q and k parts, so the scale-free kernel still computes
 *  q·kᵀ/√headDim.
 *
 *  Weights stay in torch layout for the core Linear hook to transpose. Derived
 *  byteLengths carry the source file bytes, so load progress still sums to the
 *  checkpoint size. */

import { derivedTensor, type LazyStateDict, type LazyTensor } from '../../core/index.ts';
import type { MinilmConfig } from './config.ts';

const qualify = (prefix: string, name: string): string => (prefix ? `${prefix}.${name}` : name);

function take(sd: LazyStateDict, name: string): LazyTensor {
  const t = sd.tensors.get(name);
  if (!t) throw new Error(`minilm state dict: missing tensor '${name}'`);
  sd.tensors.delete(name);
  return t;
}

function move(sd: LazyStateDict, from: string, to: string): void {
  sd.tensors.set(to, take(sd, from));
}

export function transformMinilmStateDict(
  sd: LazyStateDict,
  prefix: string,
  cfg: MinilmConfig,
): void {
  const q = (name: string) => qualify(prefix, name);
  // qkScale on q and k each — their product is the 1/√headDim score scale.
  const qkScale = cfg.headDim ** -0.25;

  sd.tensors.delete(q('pooler.dense.weight'));
  sd.tensors.delete(q('pooler.dense.bias'));
  sd.tensors.delete(q('embeddings.position_ids'));

  // position' = position + tokenType[0] (every real input is token type 0).
  const position = take(sd, q('embeddings.position_embeddings.weight'));
  const tokenType = take(sd, q('embeddings.token_type_embeddings.weight'));
  sd.tensors.set(
    q('embeddings.position.weight'),
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
  move(sd, q('embeddings.word_embeddings.weight'), q('embeddings.word.weight'));
  move(sd, q('embeddings.LayerNorm.weight'), q('embeddings.norm.weight'));
  move(sd, q('embeddings.LayerNorm.bias'), q('embeddings.norm.bias'));

  for (let i = 0; i < cfg.layers; i++) {
    const hf = (name: string) => q(`encoder.layer.${i}.${name}`);
    const ours = (name: string) => q(`layers.${i}.${name}`);

    // q|k|v → one torch-layout [3·hidden, hidden] weight: the three [out, in]
    // blocks stacked row-wise, q/k parts pre-scaled.
    const parts = (['query', 'key', 'value'] as const).map((p) => ({
      weight: take(sd, hf(`attention.self.${p}.weight`)),
      bias: take(sd, hf(`attention.self.${p}.bias`)),
      scale: p === 'value' ? 1 : qkScale,
    }));
    const H = cfg.hidden;
    sd.tensors.set(
      ours('qkv.weight'),
      derivedTensor(
        [3 * H, H],
        async () => {
          const out = new Float32Array(3 * H * H);
          for (let p = 0; p < 3; p++) {
            const { weight, scale } = parts[p]!;
            const w = await weight.f32();
            for (let i = 0; i < H * H; i++) out[p * H * H + i] = w[i]! * scale;
          }
          return out;
        },
        parts.reduce((sum, p) => sum + p.weight.byteLength, 0),
      ),
    );
    sd.tensors.set(
      ours('qkv.bias'),
      derivedTensor(
        [3 * H],
        async () => {
          const out = new Float32Array(3 * H);
          for (let p = 0; p < 3; p++) {
            const { bias, scale } = parts[p]!;
            const b = await bias.f32();
            for (let c = 0; c < H; c++) out[p * H + c] = b[c]! * scale;
          }
          return out;
        },
        parts.reduce((sum, p) => sum + p.bias.byteLength, 0),
      ),
    );

    move(sd, hf('attention.output.dense.weight'), ours('out.weight'));
    move(sd, hf('attention.output.dense.bias'), ours('out.bias'));
    move(sd, hf('attention.output.LayerNorm.weight'), ours('attnNorm.weight'));
    move(sd, hf('attention.output.LayerNorm.bias'), ours('attnNorm.bias'));

    move(sd, hf('intermediate.dense.weight'), ours('up.weight'));
    move(sd, hf('intermediate.dense.bias'), ours('up.bias'));
    move(sd, hf('output.dense.weight'), ours('down.weight'));
    move(sd, hf('output.dense.bias'), ours('down.bias'));
    move(sd, hf('output.LayerNorm.weight'), ours('ffnNorm.weight'));
    move(sd, hf('output.LayerNorm.bias'), ours('ffnNorm.bias'));
  }
}
