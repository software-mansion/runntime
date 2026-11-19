import { describe, expect, it } from 'vitest';
import { visionRopeTables } from '../../../src/zoo/embeddinggemma2/rope.ts';
import { ropeTables } from '../../../src/zoo/shared/rope.ts';

const close = (want: number) => expect.closeTo(want, 6);

/** Row `t` of a [rows, headDim] table. */
const row = (table: Float32Array, t: number, headDim: number) =>
  Array.from(table.subarray(t * headDim, (t + 1) * headDim));

describe('visionRopeTables', () => {
  const headDim = 16;
  const theta = 100;
  const cols = 3;
  const rows = 2;
  const { cos, sin } = visionRopeTables(cols, rows, headDim, theta);
  // A 1D rope of half a head, for every column and row index.
  const half = ropeTables(Math.max(cols, rows), headDim / 2, theta);

  it('is [cols · rows, headDim]', () => {
    expect(cos.length).toBe(cols * rows * headDim);
    expect(sin.length).toBe(cols * rows * headDim);
  });

  it('turns the first half of a head by the column and the second by the row', () => {
    for (let k = 0; k < cols * rows; k++) {
      const col = k % cols;
      const r = Math.floor(k / cols);
      const want = (table: Float32Array) => [
        ...row(table, col, headDim / 2),
        ...row(table, r, headDim / 2),
      ];
      expect(row(cos, k, headDim)).toEqual(want(half.cos).map(close));
      expect(row(sin, k, headDim)).toEqual(want(half.sin).map(close));
    }
  });
});
