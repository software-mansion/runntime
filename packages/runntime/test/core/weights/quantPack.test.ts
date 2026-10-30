import { describe, expect, it } from 'vitest';
import { columnWord } from '../../../src/core/kernels/quantCommon.ts';
import { packQuantW } from '../../../src/core/weights/quantPack.ts';

// The layout (kernels/quantCommon.ts): 16-value tiles, one per (4-column unit,
// 4-row chunk). Inside a tile, column by column, each column's 4 rows in a
// row. Values sit back to back at `bits` width, lowest bits first.
//
// Expected words below are worked out by hand from that description.

/** Output-first [n, k] ints where row r, column c of the logical [k, n] weight
 *  holds r·n + c, so each value names its spot. */
const numbered = (k: number, n: number) =>
  Int8Array.from({ length: k * n }, (_, i) => (i % k) * n + Math.floor(i / k));

describe('packQuantW', () => {
  it('packs one int8 tile column by column', () => {
    // [K=4, N=4]; value r*4 + c sits at row r, column c
    const words = packQuantW(numbered(4, 4), 4, 4, 8);
    // one word per column, its 4 rows from the lowest byte up
    expect([...words]).toEqual([
      0x0c080400, // column 0: rows hold 0, 4, 8, 12
      0x0d090501, // column 1: 1, 5, 9, 13
      0x0e0a0602, // column 2
      0x0f0b0703, // column 3
    ]);
  });

  it('packs one int4 tile, two columns per word', () => {
    const ints = numbered(4, 4).map((v) => (v >= 8 ? v - 16 : v)); // keep in int4 range
    expect([...packQuantW(ints, 4, 4, 4)]).toEqual([
      0xd951c840, // columns 0 and 1: nibbles 0, 4, 8, 12, 1, 5, 9, 13 from the bottom
      0xfb73ea62, // columns 2 and 3
    ]);
  });

  it.each([
    { bits: 8 as const, mask: 0xff },
    { bits: 4 as const, mask: 0xf },
  ])('keeps negative int$bits values as two’s complement bits', ({ bits, mask }) => {
    const ints = Int8Array.from({ length: 16 }, (_, i) => (i % 2 ? -1 : -(2 ** (bits - 1))));
    const words = packQuantW(ints, 4, 4, bits);
    const unpacked = Array.from(
      { length: 16 },
      (_, i) => (words[(i * bits) >> 5]! >>> ((i * bits) & 31)) & mask,
    );
    // a tile is column-major like the output-first input, so values keep their order
    const want = Array.from(ints, (v) => v & mask);
    expect(unpacked).toEqual(want);
  });

  it('orders tiles by column unit, then row chunk', () => {
    // K=8, N=8: 2 column units × 2 row chunks = 4 tiles of 4 words
    const k = 8;
    const n = 8;
    const words = packQuantW(numbered(k, n), k, n, 8);
    expect(words).toHaveLength(16);
    // first byte of each tile = row 0 of the chunk's first column
    const firstByte = (tile: number) => words[tile * 4]! & 0xff;
    expect(firstByte(0)).toBe(0 * n + 0); // columns 0..3, rows 0..3
    expect(firstByte(1)).toBe(4 * n + 0); // columns 0..3, rows 4..7
    expect(firstByte(2)).toBe(0 * n + 4); // columns 4..7, rows 0..3
    expect(firstByte(3)).toBe(4 * n + 4);
  });

  // What the kernels read: a tile starts at word (unit·K/4 + k4)·bits/2, column
  // c sits in word columnWord(c) at bit (c·4·bits) mod 32, and row r of it
  // r·bits higher, sign-extended from `bits`.
  it.each([
    { bits: 8 as const, k: 8, n: 12 },
    { bits: 4 as const, k: 12, n: 8 },
    { bits: 4 as const, k: 16, n: 16 },
  ])('reads back as the kernels read it: int$bits [$k, $n]', ({ bits, k, n }) => {
    const half = 2 ** (bits - 1);
    const ints = Int8Array.from({ length: k * n }, (_, i) => ((i * 37 + 5) % (2 * half)) - half);
    const words = packQuantW(ints, k, n, bits);
    expect(words).toHaveLength((k * n * bits) / 32);
    const read = new Int8Array(k * n);
    for (let unit = 0; unit < n / 4; unit++) {
      for (let k4 = 0; k4 < k / 4; k4++) {
        const tile = (unit * (k / 4) + k4) * (bits / 2);
        for (let c = 0; c < 4; c++) {
          const word = words[tile + columnWord(c, bits)]!;
          for (let r = 0; r < 4; r++) {
            const off = ((c * 4 * bits) % 32) + r * bits;
            const raw = (word >>> off) & ((1 << bits) - 1);
            read[(unit * 4 + c) * k + k4 * 4 + r] = raw >= half ? raw - 2 * half : raw;
          }
        }
      }
    }
    expect([...read]).toEqual([...ints]);
  });
});
