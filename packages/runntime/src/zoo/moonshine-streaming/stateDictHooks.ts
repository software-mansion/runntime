/** Maps the HF Moonshine v2 checkpoint onto the model's parameter tree as lazy
 *  derived tensors; nothing is fetched here.
 *
 *  The notable transforms: encoder norm gamma gains back its −1 offset, CMVN's
 *  missing weight is synthesized as ones, and q/k/v stack into fused weights —
 *  scaled on the encoder, which has no rope, and unscaled on the decoder, whose
 *  rope tables carry the scale. Everything else renames.
 *
 *  Weights stay in torch layout for the core Linear and Conv1d hooks to
 *  transpose. Derived byteLengths carry the source file bytes, so load progress
 *  still sums to the checkpoint size. */

import { derivedTensor, type LazyStateDict, type LazyTensor } from '../../core/index.ts';
import {
  fuseQkv,
  move,
  moveTiedEmbedding,
  qualify,
  take,
  transformDecoder,
} from '../moonshine/stateDictHooks.ts';
import type { MoonshineStreamingConfig } from './config.ts';

function gammaPlusOne(t: LazyTensor): LazyTensor {
  return derivedTensor(
    t.shape,
    async () => {
      const gamma = await t.f32();
      const out = new Float32Array(gamma.length);
      for (let j = 0; j < gamma.length; j++) out[j] = gamma[j]! + 1;
      return out;
    },
    t.byteLength,
  );
}

export function transformStreamingStateDict(
  sd: LazyStateDict,
  prefix: string,
  cfg: MoonshineStreamingConfig,
): void {
  const q = (name: string) => qualify(prefix, name);
  const [eh, dh] = [cfg.enc.hidden, cfg.dec.hidden];
  const gammaNorm = (from: string, to: string) => sd.tensors.set(to, gammaPlusOne(take(sd, from)));

  moveTiedEmbedding(sd, prefix);

  // CMVN is a LayerNorm with no learned scale, so its weight is all ones and
  // is not in the file.
  sd.tensors.set(
    q('frontend.cmvn.weight'),
    derivedTensor([cfg.frameLen], async () => new Float32Array(cfg.frameLen).fill(1), 0),
  );
  const logK = take(sd, q('model.encoder.embedder.comp.log_k'));
  sd.tensors.set(
    q('frontend.k'),
    derivedTensor(
      [1],
      async () => Float32Array.from([Math.exp((await logK.f32())[0]!)]),
      logK.byteLength,
    ),
  );
  move(sd, q('model.encoder.embedder.linear.weight'), q('frontend.linear.weight'));
  move(sd, q('model.encoder.embedder.conv1.weight'), q('frontend.conv1.weight'));
  move(sd, q('model.encoder.embedder.conv1.bias'), q('frontend.conv1.bias'));
  move(sd, q('model.encoder.embedder.conv2.weight'), q('frontend.conv2.weight'));
  move(sd, q('model.encoder.embedder.conv2.bias'), q('frontend.conv2.bias'));

  // Encoder: no rope, so the full score scale folds into the q rows.
  const encQScale = 1 / Math.sqrt(cfg.enc.headDim);
  for (let i = 0; i < cfg.enc.layers; i++) {
    const hf = q(`model.encoder.layers.${i}`);
    const en = q(`encoder.layers.${i}`);
    gammaNorm(`${hf}.input_layernorm.gamma`, `${en}.norm1.weight`);
    fuseQkv(sd, `${hf}.self_attn`, `${en}.attn.qkv.weight`, eh, encQScale);
    move(sd, `${hf}.self_attn.o_proj.weight`, `${en}.attn.out.weight`);
    gammaNorm(`${hf}.post_attention_layernorm.gamma`, `${en}.norm2.weight`);
    move(sd, `${hf}.mlp.fc1.weight`, `${en}.mlp.fc1.weight`);
    move(sd, `${hf}.mlp.fc1.bias`, `${en}.mlp.fc1.bias`);
    move(sd, `${hf}.mlp.fc2.weight`, `${en}.mlp.fc2.weight`);
    move(sd, `${hf}.mlp.fc2.bias`, `${en}.mlp.fc2.bias`);
  }
  gammaNorm(q('model.encoder.final_norm.gamma'), q('encoder.norm.weight'));

  move(sd, q('model.decoder.pos_emb.weight'), q('adapter.posEmb'));
  if (eh !== dh) move(sd, q('model.decoder.proj.weight'), q('adapter.proj.weight'));

  transformDecoder(sd, prefix, {
    layers: cfg.dec.layers,
    hidden: dh,
    headDim: cfg.dec.headDim,
  });
}
