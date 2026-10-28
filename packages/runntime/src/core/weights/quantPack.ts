/** Packs int weights into the quantW layout that kernels/quantCommon.ts
 *  documents. */

/** Packs `ints`, a row-major input-first [k, n] matrix of signed int`bits`
 *  values, into quantW words for one expert: 16-value tiles per (4-column
 *  unit, 4-row chunk), column-major in the tile at `bits` per value, lowest
 *  bits first, tiles k4-major per unit. k and n must be multiples of 4.
 *
 *  The f32 scales [k / groupSize, n] need no packing: the kernels read them
 *  row-major as they are, so the group size does not change the words. */
export function packQuantW(ints: Int8Array, k: number, n: number, bits: 4 | 8): Uint32Array {
  const words = new Uint32Array((k * n * bits) / 32);
  const mask = (1 << bits) - 1;
  for (let unit = 0; unit < n / 4; unit++) {
    for (let k4 = 0; k4 < k / 4; k4++) {
      const tileBit = (unit * (k / 4) + k4) * 16 * bits;
      for (let c = 0; c < 4; c++) {
        for (let r = 0; r < 4; r++) {
          const v = ints[(k4 * 4 + r) * n + unit * 4 + c]!;
          const bit = tileBit + (c * 4 + r) * bits;
          words[bit >> 5] = (words[bit >> 5]! | ((v & mask) << (bit & 31))) >>> 0;
        }
      }
    }
  }
  return words;
}
