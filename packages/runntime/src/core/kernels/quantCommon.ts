import { d, std } from 'typegpu';

/** The physical layout contract for quantW weight buffers, shared by the
 *  packer and both quant matmuls. Change it in one place or not at all.
 *
 *  The logical [E·K, N] weight is stored as 16-value tiles, one per (expert,
 *  4-column unit, 4-row chunk). Within a tile values are column-major and
 *  packed at `bits` stride, so a tile spans whole u32 words and a column's four
 *  values never straddle one. Tiles are k4-major per (e, n), giving a word base
 *  of ((e·N/4 + n)·K/4 + k4)·(bits/2).
 *
 *  The point: one thread owning a 4-column unit streams its weight run
 *  K-contiguously. Scales and biases are unaffected. */

export const deqVec4 = (word: number, off: number, bits: number) => {
  'use gpu';
  const w = std.bitcastU32toI32(word);
  return d.vec4f(
    d.f32(std.extractBits(w, off, bits)),
    d.f32(std.extractBits(w, off + bits, bits)),
    d.f32(std.extractBits(w, off + 2 * bits, bits)),
    d.f32(std.extractBits(w, off + 3 * bits, bits)),
  );
};
