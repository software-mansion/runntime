/** Rope in the kernel's interleaved-pair layout: table entries 2p and 2p+1
 *  hold pair p's angle, and the hooks permute q and k at load (ropeOrder)
 *  so these pairs are HF's half split. */

/** Row order for permuteRows that takes each `block` rows from the
 *  checkpoint's half-split rope layout to the kernel's interleaved pairs:
 *  rows (p, p + block/2) become (2p, 2p + 1). */
export function ropeOrder(rows: number, block: number): number[] {
  return Array.from({ length: rows }, (_, r) => {
    const i = r % block;
    return r - i + (i % 2 ? (i - 1) / 2 + block / 2 : i / 2);
  });
}

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
