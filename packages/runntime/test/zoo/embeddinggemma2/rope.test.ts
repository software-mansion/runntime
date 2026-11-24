import { describe, expect, it } from 'vitest';
import { visionRopeTables } from '../../../src/zoo/embeddinggemma2/rope.ts';

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
  // HF's Gemma vision attention ropes each half of a head, 8 columns, on its
  // own: inv_freq = 1 / theta^(arange(0, 8, 2) / 8), written out for theta
  // 100.
  const invFreq = [1, 100 ** -0.25, 0.1, 100 ** -0.75];

  it('is [cols · rows, headDim]', () => {
    expect(cos.length).toBe(cols * rows * headDim);
    expect(sin.length).toBe(cols * rows * headDim);
  });

  // The first half of a head turns with the patch's column, the second with
  // its row, each cat(freqs, freqs): entries p and p + 4 of a half hold pair
  // p's angle. Patches are row-major on the 3-wide grid.
  it.each([
    { k: 0, col: 0, r: 0 },
    { k: 2, col: 2, r: 0 },
    { k: 3, col: 0, r: 1 },
    { k: 5, col: 2, r: 1 },
  ])('turns patch $k by column $col in the first half and row $r in the second', (patch) => {
    const x = invFreq.map((f) => patch.col * f);
    const y = invFreq.map((f) => patch.r * f);
    const angles = [...x, ...x, ...y, ...y];
    expect(row(cos, patch.k, headDim)).toEqual(angles.map((a) => close(Math.cos(a))));
    expect(row(sin, patch.k, headDim)).toEqual(angles.map((a) => close(Math.sin(a))));
  });
});
