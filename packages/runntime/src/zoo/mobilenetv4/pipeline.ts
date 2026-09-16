/** CPU pre/post-processing around the MobileNetV4 GPU forward.
 *  Preprocess: RGBA canvas pixels → [3, S, S] CHW f32, ImageNet-normalized.
 */

import { IMAGENET_MEAN, IMAGENET_STD } from './config.ts';

export interface Classification {
  classId: number;
  confidence: number;
}

/** RGBA (canvas ImageData) → CHW RGB f32, (x/255 − mean)/std per channel.
 *  Pass `out` to reuse one scratch array across frames. */
export function preprocessRgba(
  rgba: Uint8ClampedArray,
  size: number,
  out?: Float32Array,
): Float32Array {
  const hw = size * size;
  const result = out ?? new Float32Array(3 * hw);
  for (let ch = 0; ch < 3; ch++) {
    const m = IMAGENET_MEAN[ch]!;
    const s = IMAGENET_STD[ch]!;
    for (let i = 0; i < hw; i++) {
      result[ch * hw + i] = (rgba[i * 4 + ch]! / 255 - m) / s;
    }
  }
  return result;
}

/** Logits [numClasses] → top-k class probabilities (softmax over ALL classes). */
export function decodeTopK(logits: Float32Array, k = 5): Classification[] {
  let max = -Infinity;
  for (const v of logits) max = Math.max(max, v);
  let sum = 0;
  const exps = new Float64Array(logits.length);
  for (let i = 0; i < logits.length; i++) {
    exps[i] = Math.exp(logits[i]! - max);
    sum += exps[i]!;
  }
  return Array.from(exps.keys())
    .sort((a, b) => exps[b]! - exps[a]!)
    .slice(0, k)
    .map((classId) => ({ classId, confidence: exps[classId]! / sum }));
}
