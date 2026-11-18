/** Derives what the HF Moonshine v2 checkpoint stores differently from the
 *  model's parameters, as lazy tensors; nothing is fetched here. Every other
 *  tensor binds under its own name, since the modules are named like the
 *  checkpoint.
 *
 *  The encoder norm gammas gain back their −1 offset and `log_k` becomes k.
 *
 *  The norm gammas and k stay f32 even from an F16 checkpoint.
 *  Derived byteLengths carry the source file bytes, so load progress counts
 *  every byte the model reads. */

import { derive, type LazyStateDict, type LazyTensor } from '../../core/index.ts';
import { qualify, take, transformDecoder } from '../moonshine/stateDictHooks.ts';
import type { MoonshineStreamingConfig } from './config.ts';

/** Stays f32 for any checkpoint: rounding gamma + 1 to f16 would drop the low
 *  bits of gamma, which an f32 run keeps. */
function gammaPlusOne(t: LazyTensor): LazyTensor {
  return derive(t.shape, [t], ([gamma]) => {
    const out = new Float32Array(gamma.length);
    for (let j = 0; j < gamma.length; j++) out[j] = gamma[j]! + 1;
    return out;
  });
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
    derive([1], [logK], ([log]) => Float32Array.from([Math.exp(log[0]!)])),
  );

  for (let i = 0; i < cfg.enc.layers; i++) {
    const L = q(`model.encoder.layers.${i}`);
    gammaNorm(`${L}.input_layernorm`);
    gammaNorm(`${L}.post_attention_layernorm`);
  }
  gammaNorm(q('model.encoder.final_norm'));

  transformDecoder(sd, prefix);
}
