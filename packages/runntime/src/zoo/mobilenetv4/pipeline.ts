/** CPU post-processing around the MobileNetV4 GPU forward. */

export interface Classification {
  classId: number;
  confidence: number;
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
