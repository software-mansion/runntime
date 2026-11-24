/** Rope tables in HF's half-split layout, the one `rope` runs as 'half'. */

/** cos and sin [positions, headDim] for theta, in HF's half-split layout:
 *  entries p and p + headDim/2 hold pair p's angle, `cat(freqs, freqs)`. Row
 *  t depends only on t. */
export function ropeTables(positions: number, headDim: number, theta: number) {
  const half = headDim / 2;
  const cos = new Float32Array(positions * headDim);
  const sin = new Float32Array(positions * headDim);
  for (let t = 0; t < positions; t++) {
    for (let p = 0; p < half; p++) {
      const angle = t / theta ** ((2 * p) / headDim);
      cos[t * headDim + p] = cos[t * headDim + half + p] = Math.cos(angle);
      sin[t * headDim + p] = sin[t * headDim + half + p] = Math.sin(angle);
    }
  }
  return { cos, sin };
}
