/** EmbeddingGemma 2 rope tables. */

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

/** The vision tower's 2D rope: cos and sin [cols · rows, headDim] for a grid
 *  of patches in row-major order. The first half of each head turns with the
 *  patch's column, the second half with its row, each half a half-split rope
 *  of headDim / 2: within a half, entries p and p + headDim/4 hold pair p's
 *  angle. */
export function visionRopeTables(cols: number, rows: number, headDim: number, theta: number) {
  const ropeDim = headDim / 2;
  const ropePairs = ropeDim / 2;
  const cos = new Float32Array(cols * rows * headDim);
  const sin = new Float32Array(cols * rows * headDim);
  for (let k = 0; k < cols * rows; k++) {
    for (const [start, pos] of [
      [0, k % cols],
      [ropeDim, Math.floor(k / cols)],
    ] as const) {
      for (let p = 0; p < ropePairs; p++) {
        const angle = pos / theta ** (p / ropePairs);
        const first = k * headDim + start + p;
        cos[first] = cos[first + ropePairs] = Math.cos(angle);
        sin[first] = sin[first + ropePairs] = Math.sin(angle);
      }
    }
  }
  return { cos, sin };
}
