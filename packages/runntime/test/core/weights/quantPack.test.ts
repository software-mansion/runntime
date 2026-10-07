import { describe, expect, it } from 'vitest';
import { packQuantColSlice } from '../../../src/core/weights/quantPack.ts';

// The layout (kernels/quantCommon.ts): 16-value tiles, one per (expert, 4-column
// unit, 4-row chunk). Inside a tile, column by column, each column's 4 rows in
// a row. Values sit back to back at `bits` width, lowest bits first.
//
// Expected words below are worked out by hand from that description.

/** Signed int8 values as the raw bytes a checkpoint stores. */
const int8 = (values: number[]) => new Uint8Array(new Int8Array(values).buffer);

/** Signed int4 values, two per byte, the even index in the low nibble. */
function int4(values: number[]): Uint8Array {
  const out = new Uint8Array(values.length / 2);
  values.forEach((v, i) => (out[i >> 1]! |= (v & 0xf) << (i & 1 ? 4 : 0)));
  return out;
}

/** Row-major [K, O] values numbered 0, 1, 2, ... so each value names its spot. */
const numbered = (k: number, o: number) => Array.from({ length: k * o }, (_, i) => i);

const noScales = (E: number, KG: number, O: number) => new Float32Array(E * KG * O);

describe('packQuantColSlice', () => {
  it('packs one int8 tile column by column', () => {
    // [K=4, O=4]; value r*4 + c sits at row r, column c
    const q = int8(numbered(4, 4));
    const { qdata } = packQuantColSlice(q, noScales(1, 1, 4), 8, 4, 1, 4, 4, 0, 4);
    // one word per column, its 4 rows from the lowest byte up
    expect([...qdata]).toEqual([
      0x0c080400, // column 0: rows hold 0, 4, 8, 12
      0x0d090501, // column 1: 1, 5, 9, 13
      0x0e0a0602, // column 2
      0x0f0b0703, // column 3
    ]);
  });

  it('packs one int4 tile, two columns per word', () => {
    const q = int4(numbered(4, 4).map((v) => (v >= 8 ? v - 16 : v))); // keep in int4 range
    const { qdata } = packQuantColSlice(q, noScales(1, 1, 4), 4, 4, 1, 4, 4, 0, 4);
    expect([...qdata]).toEqual([
      0xd951c840, // columns 0 and 1: nibbles 0, 4, 8, 12, 1, 5, 9, 13 from the bottom
      0xfb73ea62, // columns 2 and 3
    ]);
  });

  it.each([
    { bits: 8 as const, pack: int8, mask: 0xff },
    { bits: 4 as const, pack: int4, mask: 0xf },
  ])('keeps negative int$bits values as two’s complement bits', ({ bits, pack, mask }) => {
    const values = Array.from({ length: 16 }, (_, i) => (i % 2 ? -1 : -(bits === 8 ? 128 : 8)));
    const { qdata } = packQuantColSlice(pack(values), noScales(1, 1, 4), bits, 4, 1, 4, 4, 0, 4);
    const unpacked = Array.from(
      { length: 16 },
      (_, i) => (qdata[(i * bits) >> 5]! >>> ((i * bits) & 31)) & mask,
    );
    // column-major in, so value (row r, column c) comes out at c*4 + r
    const want = Array.from({ length: 16 }, (_, i) => values[(i % 4) * 4 + (i >> 2)]! & mask);
    expect(unpacked).toEqual(want);
  });

  it('takes only the requested columns', () => {
    // [K=4, O=8], slice columns 4..7: the same tile as columns 0..3 shifted by 4
    const q = int8(numbered(4, 8));
    const { qdata } = packQuantColSlice(q, noScales(1, 1, 8), 8, 4, 1, 4, 8, 4, 8);
    expect([...qdata]).toEqual([
      0x1c140c04, // column 4: rows hold 4, 12, 20, 28
      0x1d150d05,
      0x1e160e06,
      0x1f170f07,
    ]);
  });

  it('orders tiles by expert, then column unit, then row chunk', () => {
    // E=2, K=8, O=8: per expert 2 column units × 2 row chunks = 4 tiles of 4 words
    const E = 2;
    const K = 8;
    const O = 8;
    const q = int8(Array.from({ length: E * K * O }, (_, i) => i - 128));
    const { qdata } = packQuantColSlice(q, noScales(E, 2, O), 8, 4, E, K, O, 0, O);
    expect(qdata).toHaveLength(E * 4 * 4);
    // first word of each tile = rows 0..3 of the chunk's first column
    const firstByte = (tile: number) => ((qdata[tile * 4]! << 24) >> 24) + 128;
    const index = (e: number, row: number, col: number) => (e * K + row) * O + col;
    expect(firstByte(0)).toBe(index(0, 0, 0)); // expert 0, columns 0..3, rows 0..3
    expect(firstByte(1)).toBe(index(0, 4, 0)); // expert 0, columns 0..3, rows 4..7
    expect(firstByte(2)).toBe(index(0, 0, 4)); // expert 0, columns 4..7, rows 0..3
    expect(firstByte(3)).toBe(index(0, 4, 4));
    expect(firstByte(4)).toBe(index(1, 0, 0)); // expert 1 starts here
    expect(firstByte(7)).toBe(index(1, 4, 4));
  });

  it('puts every value of the slice in exactly once', () => {
    const E = 2;
    const K = 8;
    const O = 12;
    const values = Array.from({ length: E * K * O }, (_, i) => (i % 251) - 125);
    const { qdata } = packQuantColSlice(int8(values), noScales(E, 2, O), 8, 4, E, K, O, 4, 12);
    const got = [...new Int8Array(qdata.buffer)].sort((a, b) => a - b);
    const want = values
      .filter((_, i) => i % O >= 4) // columns 4..11 of every row
      .sort((a, b) => a - b);
    expect(got).toEqual(want);
  });

  it('slices the scales to the same columns', () => {
    // scales [E=1, KG=2, O=8]: value g*10 + column
    const scales = Float32Array.from({ length: 16 }, (_, i) => Math.floor(i / 8) * 10 + (i % 8));
    const q = int8(new Array(8 * 8).fill(0));
    const out = packQuantColSlice(q, scales, 8, 4, 1, 8, 8, 4, 8);
    expect([...out.scales]).toEqual([4, 5, 6, 7, 14, 15, 16, 17]);
  });

  it.each([
    { name: 'an empty slice', args: [8, 4, 1, 4, 4, 2, 2], error: /width 0/ },
    { name: 'a slice width not a multiple of 4', args: [8, 4, 1, 4, 8, 0, 6], error: /width 6/ },
    { name: 'K not a multiple of 4', args: [8, 2, 1, 6, 4, 0, 4], error: /K 6/ },
    {
      name: 'a group size that does not divide K',
      args: [8, 3, 1, 4, 4, 0, 4],
      error: /groupSize 3/,
    },
  ] as const)('rejects $name', ({ args, error }) => {
    const [bits, groupSize, E, K, O, oStart, oEnd] = args;
    const q = new Uint8Array(E * K * O);
    expect(() =>
      packQuantColSlice(q, noScales(E, 1, O), bits, groupSize, E, K, O, oStart, oEnd),
    ).toThrow(error);
  });

  it('rejects quantized data or scales of the wrong length', () => {
    expect(() =>
      packQuantColSlice(new Uint8Array(15), noScales(1, 1, 4), 8, 4, 1, 4, 4, 0, 4),
    ).toThrow(/qbytes 15 != 16/);
    // int4 holds two values per byte
    expect(() =>
      packQuantColSlice(new Uint8Array(16), noScales(1, 1, 4), 4, 4, 1, 4, 4, 0, 4),
    ).toThrow(/qbytes 16 != 8/);
    expect(() =>
      packQuantColSlice(new Uint8Array(16), new Float32Array(3), 8, 4, 1, 4, 4, 0, 4),
    ).toThrow(/scales 3 != 4/);
  });
});
