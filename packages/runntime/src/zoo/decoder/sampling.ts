/** Next-token selection on the CPU: greedy, or temperature sampling over
 *  the top-k tokens cut to a top-p nucleus, with a seedable generator. */

import { RunntimeError } from '../../core/index.ts';

/** mulberry32. */
export function rng(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Picks the next token from (already processed) logits. */
export function sampleToken(
  logits: Float32Array,
  candidates: readonly number[] | undefined,
  temperature: number,
  topK: number,
  topP: number,
  random: () => number,
): number {
  const n = candidates ? candidates.length : logits.length;
  const idAt = candidates ? (i: number) => candidates[i]! : (i: number) => i;
  if (temperature <= 0) {
    let best = -1;
    let bestV = -Infinity;
    for (let i = 0; i < n; i++) {
      const id = idAt(i);
      const v = logits[id]!;
      if (v > bestV) {
        bestV = v;
        best = id;
      }
    }
    if (best < 0) throw new RunntimeError('INVALID_ARGUMENT', 'generate: every token is masked');
    return best;
  }
  // Top-k by a small sorted buffer (k is small, V is large).
  const k = Math.max(1, Math.min(topK, n));
  const ids: number[] = [];
  const vals: number[] = [];
  for (let i = 0; i < n; i++) {
    const id = idAt(i);
    const v = logits[id]!;
    if (v === -Infinity || Number.isNaN(v)) continue;
    if (vals.length === k && v <= vals[k - 1]!) continue;
    let j = vals.length === k ? k - 1 : vals.length;
    if (vals.length < k) {
      vals.push(v);
      ids.push(id);
    }
    while (j > 0 && vals[j - 1]! < v) {
      vals[j] = vals[j - 1]!;
      ids[j] = ids[j - 1]!;
      j--;
    }
    vals[j] = v;
    ids[j] = id;
  }
  if (ids.length === 0)
    throw new RunntimeError('INVALID_ARGUMENT', 'generate: every token is masked');
  const max = vals[0]!;
  const probs = vals.map((v) => Math.exp((v - max) / temperature));
  const total = probs.reduce((a, b) => a + b, 0);
  let cut = probs.length;
  let acc = 0;
  for (let i = 0; i < probs.length; i++) {
    acc += probs[i]! / total;
    if (acc >= topP) {
      cut = i + 1;
      break;
    }
  }
  const kept = probs.slice(0, cut);
  const keptTotal = kept.reduce((a, b) => a + b, 0);
  let r = random() * keptTotal;
  for (let i = 0; i < kept.length; i++) {
    r -= kept[i]!;
    if (r <= 0) return ids[i]!;
  }
  return ids[kept.length - 1]!;
}
