import { describe, expect, it } from 'vitest';
import {
  COCO_LANDMARKS,
  COCO_NAMES,
  COCO_SKELETON,
  decodeDetections,
  decodePoses,
  decodeSegmentations,
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
  // A 32×32 input with 2 mask coefficients and an 8×8 proto grid, so a proto
  // pixel is 4 input pixels: half a cell at stride 8, a quarter at stride 16.
  const opts = { numClasses: 1, numMasks: 2, inputSize: 32 };
  const fine = () => level(4, 4, 8, 1, 2);

  /** Marks `cell` of `lvl` confident and writes its box and mask coefficients. */
  function hit(lvl: RawLevel, cell: number, ltrb: readonly number[], coeffs = [0, 0]): RawLevel {
    put(lvl, cell, 0, [...ltrb, logit(0.9), ...coeffs]);
    return lvl;
  }

  /** Two proto planes over the 8×8 grid, filled from `planes[c][y·8 + x]`. */
  function proto(planes: Record<number, Record<number, number>>) {
    const data = new Float32Array(2 * 64);
    for (const [c, pixels] of Object.entries(planes)) {
      for (const [i, v] of Object.entries(pixels)) data[Number(c) * 64 + Number(i)] = v;
    }
    return { data, h: 8, w: 8 };
  }

  it('crops the mask to the box and scores each pixel as sigmoid(coeffs · proto)', () => {
    // Fine cell (2, 2), ltrb 1: box (12, 12)–(28, 28), proto pixels 3..6.
    // With coeffs (2, −1): proto (3, 3) sums to 10, 255; (x 5, y 4) to 2,
    // round(255·sigmoid(2)) = 225; (6, 6) to −10, 0. Every other pixel sums
    // to 0, sigmoid 0.5, 128. Pixel (0, 0) is outside the box.
    const [det] = decodeSegmentations(
      [hit(fine(), 2 * 4 + 2, [1, 1, 1, 1], [2, -1])],
      proto({ 0: { 0: 10, 27: 5, 37: 1 }, 1: { 54: 10 } }),
      opts,
    );
    expect(det).toMatchObject({ x1: 12, y1: 12, x2: 28, y2: 28 });
    expect(det).toMatchObject({ maskX: 3, maskY: 3, maskW: 4, maskH: 4 });
    // prettier-ignore
    expect([...det!.mask]).toEqual([
      255, 128, 128, 128,
      128, 128, 225, 128,
      128, 128, 128, 128,
      128, 128, 128, 0,
    ]);
  });

  it('reads the coefficients of a coarser-level hit from after the class planes', () => {
    // Coarse cell (1, 0) is centered on (24, 8). ltrb (0.5, 0.25, 0.25, 0.5)
    // cells of 16: box (16, 4)–(28, 16), proto pixels x 4..6, y 1..3.
    // With coeffs (−1, 3): proto (4, 1) sums to −1 + 3 = 2, 225; (5, 2) to 6,
    // round(255·0.99753) = 254; (6, 3) to −4, round(255·0.01799) = 5.
    // Three classes, class 2 (car) confident, so the coeffs are planes 7 and 8.
    const lvl = level(2, 2, 16, 3, 2);
    put(lvl, 1, 0, [0.5, 0.25, 0.25, 0.5]);
    put(lvl, 1, 4 + 2, [logit(0.9)]);
    put(lvl, 1, 4 + 3, [-1, 3]);
    const [det] = decodeSegmentations(
      [level(4, 4, 8, 3, 2), lvl],
      proto({ 0: { 12: 1, 30: 4 }, 1: { 12: 1, 21: 2 } }),
      { ...opts, numClasses: 3 },
    );
    expect(det).toMatchObject({ x1: 16, y1: 4, x2: 28, y2: 16, classId: 2, label: 'car' });
    expect(det).toMatchObject({ maskX: 4, maskY: 1, maskW: 3, maskH: 3 });
    // prettier-ignore
    expect([...det!.mask]).toEqual([
      225, 128, 128,
      128, 254, 128,
      128, 128, 5,
    ]);
  });

  it.each([
    {
      name: 'clips a box hanging off the top-left corner',
      // Cell 0: box (−12, −12)–(12, 12) covers proto pixels 0..2
      cell: 0,
      ltrb: [2, 2, 1, 1],
      want: { maskX: 0, maskY: 0, maskW: 3, maskH: 3 },
    },
    {
      name: 'clips a box hanging off the bottom-right corner',
      // Cell (3, 3): box (20, 20)–(44, 44) covers proto pixels 5..7
      cell: 15,
      ltrb: [1, 1, 2, 2],
      want: { maskX: 5, maskY: 5, maskW: 3, maskH: 3 },
    },
    {
      name: 'gives an empty mask for a box with no area',
      // Cell (1, 1): box (8, 8)–(8, 8) sits on a proto pixel edge
      cell: 5,
      ltrb: [0.5, 0.5, -0.5, -0.5],
      want: { maskX: 2, maskY: 2, maskW: 0, maskH: 0 },
    },
  ])('$name', ({ cell, ltrb, want }) => {
    const [det] = decodeSegmentations([hit(fine(), cell, ltrb)], proto({}), opts);
    expect(det).toMatchObject(want);
    expect(det!.mask).toHaveLength(want.maskW * want.maskH);
  });
});

describe('decodePoses', () => {
  it('places keypoints as (offset + cell center) · stride, visibility as sigmoid', () => {
    // Same decode as Ultralytics Pose26.kpts_decode. Cell (1, 1) of the
    // stride-16 level is centered on (1.5, 1.5) cells. Two classes, class 1
    // confident, so the keypoint planes start at 6.
    const fine = level(4, 4, 8, 2, 6);
    const coarse = level(2, 2, 16, 2, 6);
    put(coarse, 3, 0, [1, 0.5, 0.5, 1]);
    put(coarse, 3, 4 + 1, [logit(0.8)]);
    put(coarse, 3, 4 + 2, [-0.25, 0.5, 2, 0.75, -1, logit(0.3)]);
    expect(decodePoses([fine, coarse], { numClasses: 2, numKeypoints: 2 })).toEqual([
      {
        x1: (1.5 - 1) * 16,
        y1: (1.5 - 0.5) * 16,
        x2: (1.5 + 0.5) * 16,
        y2: (1.5 + 1) * 16,
        score: expect.closeTo(0.8, 6),
        classId: 1,
        label: 'bicycle',
        kpts: [
          {
            x: (-0.25 + 1.5) * 16,
            y: (0.5 + 1.5) * 16,
            v: expect.closeTo(1 / (1 + Math.exp(-2)), 6),
          },
          { x: (0.75 + 1.5) * 16, y: (-1 + 1.5) * 16, v: expect.closeTo(0.3, 6) },
        ],
      },
    ]);
  });
});

describe('COCO tables', () => {
  // Spot checks against Ultralytics coco.yaml, coco-pose.yaml and the
  // 1-based pose skeleton in utils/plotting.py.
  it.each([
    [0, 'person'],
    [56, 'chair'],
    [79, 'toothbrush'],
  ])('names class %i %s', (id, name) => {
    expect(COCO_NAMES[id]).toBe(name);
  });

  it.each([
    [0, 'nose'],
    [9, 'leftWrist'],
    [16, 'rightAnkle'],
  ])('names landmark %i %s', (id, name) => {
    expect(COCO_LANDMARKS[id]).toBe(name);
  });

  it.each([
    [15, 13],
    [5, 6],
    [2, 4],
  ])('joins landmarks %i and %i', (a, b) => {
    expect(COCO_SKELETON).toContainEqual([a, b]);
  });
});
