/** Comparing a GPU detection list with the CPU method's: pairing entries the
 *  two may rank differently, and how far a video frame's may differ. */
import type { BoundingBox } from '../../../../src/zoo/tasks/cv/ops/box.ts';

/** How far a video frame's GPU detections may sit from the CPU method's on
 *  the same frame. Chrome hands the import the same bytes a canvas reads
 *  from the frame (measured: none differ), but a frame resampled into the
 *  input picks up rounding differences between the GPU and CPU bilinear
 *  fits, which the random f16 net turns into up to 0.03 px on a landmark,
 *  0.008 px on a box and 1.6e-5 on a confidence over 18 frames of six
 *  sizes. About three times that. */
export const FRAME_TOLERANCE = { pixels: 0.1, confidence: 5e-5 };

interface Boxed {
  readonly box: Pick<BoundingBox<'xyxy'>, 'xmin' | 'ymin' | 'xmax' | 'ymax'>;
}

/** The first `want.length` of `detections`, reordered so entry `i` is the
 *  one whose box is nearest `want[i]`'s. The two paths sort by confidence,
 *  and two near-tied scores can swap when the model input differs by
 *  rounding, as it does for a resampled video frame. */
export function byNearestBox<G extends Boxed>(
  detections: readonly G[],
  want: readonly Boxed[],
): G[] {
  const left = new Set(detections);
  return want.map((w) => {
    let best: G | undefined;
    let bestDistance = Infinity;
    for (const g of left) {
      const distance =
        Math.abs(g.box.xmin - w.box.xmin) +
        Math.abs(g.box.ymin - w.box.ymin) +
        Math.abs(g.box.xmax - w.box.xmax) +
        Math.abs(g.box.ymax - w.box.ymax);
      if (distance < bestDistance) [best, bestDistance] = [g, distance];
    }
    left.delete(best!);
    return best!;
  });
}
