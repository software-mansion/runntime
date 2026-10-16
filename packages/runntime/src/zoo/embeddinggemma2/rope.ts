/** EmbeddingGemma 2 rope tables, in the kernel's interleaved-pair layout:
 *  entries 2p and 2p+1 hold pair p's angle. The q and k weights are permuted
 *  at load so these pairs are HF's half split (stateDictHooks). */

/** cos and sin [positions, headDim] for theta. Row t depends only on t. */
export function ropeTables(positions: number, headDim: number, theta: number) {
  const cos = new Float32Array(positions * headDim);
  const sin = new Float32Array(positions * headDim);
  for (let t = 0; t < positions; t++) {
    for (let p = 0; p < headDim / 2; p++) {
      const angle = t / theta ** ((2 * p) / headDim);
      cos[t * headDim + 2 * p] = cos[t * headDim + 2 * p + 1] = Math.cos(angle);
      sin[t * headDim + 2 * p] = sin[t * headDim + 2 * p + 1] = Math.sin(angle);
    }
  }
  return { cos, sin };
}
