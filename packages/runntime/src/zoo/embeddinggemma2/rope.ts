/** EmbeddingGemma 2's vision rope tables, in the interleaved-pair layout of
 *  zoo/shared/rope.ts. */

/** The vision tower's 2D rope: cos and sin [cols · rows, headDim] for a grid
 *  of patches in row-major order. The first half of each head turns with the
 *  patch's column, the second half with its row, each half a rope of
 *  headDim / 2. */
export function visionRopeTables(cols: number, rows: number, headDim: number, theta: number) {
  // Rotation pairs per head, and per half head.
  const pairs = headDim / 2;
  const halfPairs = pairs / 2;
  const cos = new Float32Array(cols * rows * headDim);
  const sin = new Float32Array(cols * rows * headDim);
  for (let k = 0; k < cols * rows; k++) {
    for (let p = 0; p < pairs; p++) {
      const pos = p < halfPairs ? k % cols : Math.floor(k / cols);
      const angle = pos / theta ** ((p % halfPairs) / halfPairs);
      cos[k * headDim + 2 * p] = cos[k * headDim + 2 * p + 1] = Math.cos(angle);
      sin[k * headDim + 2 * p] = sin[k * headDim + 2 * p + 1] = Math.sin(angle);
    }
  }
  return { cos, sin };
}
