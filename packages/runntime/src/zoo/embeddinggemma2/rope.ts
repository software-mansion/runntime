/** EmbeddingGemma 2's vision rope tables. The text model's are
 *  zoo/shared/rope.ts. */

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
