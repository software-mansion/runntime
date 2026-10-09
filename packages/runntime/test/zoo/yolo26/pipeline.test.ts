import { describe, expect, it } from 'vitest';
import {
  COCO_LANDMARKS,
  COCO_NAMES,
  COCO_SKELETON,
  decodeDetections,
  decodeSegmentations,
  preprocessRgba,
  type RawLevel,
} from '../../../src/zoo/yolo26/pipeline.ts';

/** A logit no threshold keeps: sigmoid(−20) ≈ 2e-9. */
const NEVER = -20;

/** One head output over an h×w grid: 4 box planes (l, t, r, b in cells), then
 *  `numClasses` logit planes, then `extra` planes. Every class logit starts at
 *  NEVER, so only the cells a test fills can be detected. */
function level(h: number, w: number, stride: number, numClasses: number, extra = 0): RawLevel {
  const data = new Float32Array((4 + numClasses + extra) * h * w);
  data.fill(NEVER, 4 * h * w, (4 + numClasses) * h * w);
  return { data, h, w, stride };
}

/** Writes `values` into consecutive planes of one cell, from plane `first`. */
function put(lvl: RawLevel, cell: number, first: number, values: readonly number[]): void {
  values.forEach((v, j) => {
    lvl.data[(first + j) * lvl.h * lvl.w + cell] = v;
  });
}

const logit = (p: number) => Math.log(p / (1 - p));

describe('preprocessRgba', () => {
  // Red, green, blue and (51, 102, 204), each with alpha 7
  const rgba = new Uint8ClampedArray([255, 0, 0, 7, 0, 255, 0, 7, 0, 0, 255, 7, 51, 102, 204, 7]);
  const want = [1, 0, 0, 0.2, 0, 1, 0, 0.4, 0, 0, 1, 0.8];

  it('splits RGBA into R, G and B planes in [0, 1] and drops alpha', () => {
    expect([...preprocessRgba(rgba, 2)]).toEqual(want.map((v) => expect.closeTo(v, 6)));
  });

  it('writes into the given array', () => {
    const out = new Float32Array(12);
    expect(preprocessRgba(rgba, 2, out)).toBe(out);
    expect([...out]).toEqual(want.map((v) => expect.closeTo(v, 6)));
  });
});

describe('decodeDetections', () => {
  it('turns a cell’s ltrb distances into a box around the cell center', () => {
    // Cell (2, 1) of a 4×4 grid at stride 8 is centered on (2.5, 1.5) cells
    const lvl = level(4, 4, 8, 3);
    put(lvl, 1 * 4 + 2, 0, [1, 1, 2, 3, -1, 2, 0]);
    expect(decodeDetections([lvl], { numClasses: 3 })).toEqual([
      {
        x1: (2.5 - 1) * 8,
        y1: (1.5 - 1) * 8,
        x2: (2.5 + 2) * 8,
        y2: (1.5 + 3) * 8,
        score: expect.closeTo(1 / (1 + Math.exp(-2)), 6),
        classId: 1,
        label: 'bicycle',
      },
    ]);
  });

  it.each([
    { name: 'keeps a score exactly at the threshold', score: 0.5, kept: 1 },
    { name: 'drops a score just under it', score: 0.499, kept: 0 },
  ])('$name', ({ score, kept }) => {
    const lvl = level(2, 2, 8, 1);
    put(lvl, 0, 4, [logit(score)]);
    expect(decodeDetections([lvl], { numClasses: 1, confThreshold: 0.5 })).toHaveLength(kept);
  });

  it('sorts by score across levels and keeps the best maxDet', () => {
    const fine = level(4, 4, 8, 1);
    const coarse = level(2, 2, 16, 1);
    put(fine, 0, 4, [logit(0.6)]);
    put(fine, 5, 4, [logit(0.8)]);
    put(coarse, 3, 4, [logit(0.9)]);
    const scores = (dets: { score: number }[]) => dets.map((d) => d.score);
    expect(scores(decodeDetections([fine, coarse], { numClasses: 1 }))).toEqual(
      [0.9, 0.8, 0.6].map((p) => expect.closeTo(p, 6)),
    );
    expect(scores(decodeDetections([fine, coarse], { numClasses: 1, maxDet: 2 }))).toEqual(
      [0.9, 0.8].map((p) => expect.closeTo(p, 6)),
    );
  });

  it('finds nothing when no cell passes the threshold', () => {
    expect(decodeDetections([level(4, 4, 8, 80)])).toEqual([]);
  });

  it('labels a class past the COCO names by its id', () => {
    const lvl = level(2, 2, 8, 81);
    put(lvl, 0, 4 + 80, [logit(0.9)]);
    expect(decodeDetections([lvl], { numClasses: 81 })[0]?.label).toBe('cls80');
  });
});

describe('decodeSegmentations', () => {
  // A 32×32 input, 4×4 grid at stride 8, 2 mask coefficients, and a 4×4 proto
  // grid: one proto pixel per cell.
  const opts = { numClasses: 1, numMasks: 2, inputSize: 32 };

  /** A level with one confident cell at `cell`, its box and mask coefficients. */
  function oneHit(cell: number, ltrb: readonly number[], coeffs = [0, 0]): RawLevel {
    const lvl = level(4, 4, 8, 1, 2);
    put(lvl, cell, 0, [...ltrb, logit(0.9), ...coeffs]);
    return lvl;
  }

  /** Two proto planes over the 4×4 grid, filled from `planes[c][y·4 + x]`. */
  function proto(planes: Record<number, Record<number, number>>) {
    const data = new Float32Array(2 * 16);
    for (const [c, pixels] of Object.entries(planes)) {
      for (const [i, v] of Object.entries(pixels)) data[Number(c) * 16 + Number(i)] = v;
    }
    return { data, h: 4, w: 4 };
  }

  it('crops the mask to the box and scores each pixel as sigmoid(coeffs · proto)', () => {
    // Cell (2, 2), ltrb 1: box (12, 12)–(28, 28), rounded out to proto pixels
    // 1..3. With coeffs (2, −1), proto (2, 2) sums to 10 and (3, 3) to −10;
    // every other pixel sums to 0, sigmoid 0.5, 128. Pixel (0, 0) is outside.
    const [det] = decodeSegmentations(
      [oneHit(2 * 4 + 2, [1, 1, 1, 1], [2, -1])],
      proto({ 0: { 0: 10, 10: 5 }, 1: { 15: 10 } }),
      opts,
    );
    expect(det).toMatchObject({ x1: 12, y1: 12, x2: 28, y2: 28 });
    expect(det).toMatchObject({ maskX: 1, maskY: 1, maskW: 3, maskH: 3 });
    expect([...det!.mask]).toEqual([128, 128, 128, 128, 255, 128, 128, 128, 0]);
  });

  it.each([
    {
      name: 'clips a box hanging off the top-left corner',
      // Cell 0: box (−12, −12)–(12, 12) covers proto pixels 0..1
      cell: 0,
      ltrb: [2, 2, 1, 1],
      want: { maskX: 0, maskY: 0, maskW: 2, maskH: 2 },
    },
    {
      name: 'clips a box hanging off the bottom-right corner',
      // Cell (3, 3): box (20, 20)–(44, 44) covers proto pixels 2..3
      cell: 15,
      ltrb: [1, 1, 2, 2],
      want: { maskX: 2, maskY: 2, maskW: 2, maskH: 2 },
    },
    {
      name: 'gives an empty mask for a box with no area',
      // Cell (1, 1): box (8, 8)–(8, 8) sits on a proto pixel edge
      cell: 5,
      ltrb: [0.5, 0.5, -0.5, -0.5],
      want: { maskX: 1, maskY: 1, maskW: 0, maskH: 0 },
    },
  ])('$name', ({ cell, ltrb, want }) => {
    const [det] = decodeSegmentations([oneHit(cell, ltrb)], proto({}), opts);
    expect(det).toMatchObject(want);
    expect(det!.mask).toHaveLength(want.maskW * want.maskH);
  });
});

describe('COCO tables', () => {
  it('has 80 class names and 17 landmarks', () => {
    expect(COCO_NAMES).toHaveLength(80);
    expect(COCO_LANDMARKS).toHaveLength(17);
    expect(COCO_LANDMARKS[0]).toBe('nose');
  });

  it('joins only existing landmarks in the skeleton', () => {
    for (const [a, b] of COCO_SKELETON) {
      expect(a).toBeLessThan(COCO_LANDMARKS.length);
      expect(b).toBeLessThan(COCO_LANDMARKS.length);
    }
  });
});
