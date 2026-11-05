/** Derives what the HF Moonshine v2 checkpoint stores differently from the
 *  model's parameters, as lazy tensors; nothing is fetched here. Every other
 *  tensor binds under its own name, since the modules are named like the
 *  checkpoint.
 *
 *  The encoder norm gammas gain back their −1 offset, `log_k` becomes k, and
 *  q/k/v stack into fused weights — scaled on the encoder, which has no
 *  rope, and unscaled on the decoder, whose rope tables carry the scale.
 *
 *  Weights stay in torch layout for the core Linear hook to transpose. An F16
 *  checkpoint gives f16 decoder q/k/v; the scaled encoder q/k/v and
 *  cross-attention q, the norm gammas and k stay f32. Derived byteLengths
 *  carry the source file bytes, so load progress still sums to the
 *  checkpoint size. */

import { derivedTensor, type LazyStateDict, type LazyTensor } from '../../core/index.ts';
import { fuseQkv, qualify, take, transformDecoder } from '../moonshine/stateDictHooks.ts';
import type { MoonshineStreamingConfig } from './config.ts';

/** Stays f32 for any checkpoint: rounding gamma + 1 to f16 would drop the low
 *  bits of gamma, which an f32 run keeps. */
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
  /** The norm at `norm` reads its `gamma` as a weight. */
  const gammaNorm = (norm: string) =>
    sd.tensors.set(`${norm}.weight`, gammaPlusOne(take(sd, `${norm}.gamma`)));

  const comp = q('model.encoder.embedder.comp');
  const logK = take(sd, `${comp}.log_k`);
  sd.tensors.set(
    `${comp}.k`,
    derivedTensor(
      [1],
      async () => Float32Array.from([Math.exp((await logK.f32())[0]!)]),
      logK.byteLength,
    ),
  );

  // Encoder: no rope, so the full score scale folds into the q rows.
  const encQScale = 1 / Math.sqrt(cfg.enc.headDim);
  for (let i = 0; i < cfg.enc.layers; i++) {
    const L = q(`model.encoder.layers.${i}`);
    gammaNorm(`${L}.input_layernorm`);
    fuseQkv(sd, `${L}.self_attn`, cfg.enc.hidden, encQScale);
    gammaNorm(`${L}.post_attention_layernorm`);
  }
  gammaNorm(q('model.encoder.final_norm'));

  transformDecoder(sd, prefix, {
    layers: cfg.dec.layers,
    hidden: cfg.dec.hidden,
    headDim: cfg.dec.headDim,
  });
}
