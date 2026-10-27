import { d, std } from 'typegpu';

/** The physical layout contract for quantW weight buffers, shared by the
 *  packer and the matmulQuantW, moeQuantW and matmulQuantTiled kernels.
 *  Change it in one place or not at all.
 *
 *  The logical [E·K, N] weight is stored as 16-value tiles, one per (expert,
 *  4-column unit, 4-row chunk). Within a tile values are column-major and
 *  packed at `bits` stride, so a tile spans whole u32 words and a column's four
 *  values never straddle one. Tiles are k4-major per (e, n), giving a word base
 *  of ((e·N/4 + n)·K/4 + k4)·(bits/2).
 *
 *  The point: one thread owning a 4-column unit streams its weight run
 *  K-contiguously. Scales and biases are unaffected. */

/** The four values of one column of a tile, from the word that holds it and
 *  the column's bit offset in that word. */
export const deqVec4 = (word: number, off: number, bits: number) => {
  'use gpu';
  const w = std.bitcast(d.u32, d.i32)(word);
  return d.vec4f(
    d.f32(std.extractBits(w, off, bits)),
    d.f32(std.extractBits(w, off + bits, bits)),
    d.f32(std.extractBits(w, off + 2 * bits, bits)),
    d.f32(std.extractBits(w, off + 3 * bits, bits)),
  );
};

/** The word of a tile that holds column c, from its first word. */
export const columnWord = (c: number, bits: number) => Math.floor((c * 4 * bits) / 32);

/** Row r of a tile, one value per column, from the words that hold columns
 *  0 to 3. */
export const deqRow = (c0: number, c1: number, c2: number, c3: number, r: number, bits: number) => {
  'use gpu';
  const off = r * bits;
  return d.vec4f(
    d.f32(std.extractBits(std.bitcast(d.u32, d.i32)(c0), off, bits)),
    d.f32(std.extractBits(std.bitcast(d.u32, d.i32)(c1), (4 * bits + off) % 32, bits)),
    d.f32(std.extractBits(std.bitcast(d.u32, d.i32)(c2), (8 * bits + off) % 32, bits)),
    d.f32(std.extractBits(std.bitcast(d.u32, d.i32)(c3), (12 * bits + off) % 32, bits)),
  );
};
