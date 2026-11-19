import { describe, expect, it } from 'vitest';
import { ropeOrder, ropeTables } from '../../../src/zoo/shared/rope.ts';

const close = (want: number) => expect.closeTo(want, 6);

/** Row `t` of a [rows, headDim] table. */
const row = (table: Float32Array, t: number, headDim: number) =>
  Array.from(table.subarray(t * headDim, (t + 1) * headDim));

describe('ropeTables', () => {
  const headDim = 8;
  const theta = 10_000;
  const { cos, sin } = ropeTables(5, headDim, theta);

  it('is [positions, headDim]', () => {
    expect(cos.length).toBe(5 * headDim);
    expect(sin.length).toBe(5 * headDim);
  });

  it('gives a position the same row whatever the table length', () => {
    const short = ropeTables(3, headDim, theta);
    expect(short.cos).toEqual(cos.subarray(0, 3 * headDim));
    expect(short.sin).toEqual(sin.subarray(0, 3 * headDim));
  });

  it('does not turn position 0', () => {
    expect(row(cos, 0, headDim)).toEqual(Array(headDim).fill(1));
    expect(row(sin, 0, headDim)).toEqual(Array(headDim).fill(0));
  });

  it('turns pair p of position t by t / theta^(2p / headDim)', () => {
    for (const t of [1, 4]) {
      for (let p = 0; p < headDim / 2; p++) {
        const angle = t / theta ** ((2 * p) / headDim);
        // Entries 2p and 2p + 1 hold the same angle.
        for (const i of [2 * p, 2 * p + 1]) {
          expect(cos[t * headDim + i]).toEqual(close(Math.cos(angle)));
          expect(sin[t * headDim + i]).toEqual(close(Math.sin(angle)));
        }
      }
    }
  });
});

describe('ropeOrder', () => {
  it('takes each block from half-split to interleaved pairs', () => {
    // Rows (p, p + 2) of each block of 4 become (2p, 2p + 1).
    expect(ropeOrder(8, 4)).toEqual([0, 2, 1, 3, 4, 6, 5, 7]);
  });
});
