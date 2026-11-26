/** Shared pieces of the YOLO26 encode<Verb> tests: the random checkpoint, the
 *  letterbox model input, and comparing a GPU detection list with the CPU
 *  method's: pairing entries the two may rank differently, and how far a
 *  video frame's may differ. */
import type { TgpuRoot } from 'typegpu';
import { expect, onTestFinished } from 'vitest';
import type { ImageBuffer } from '../../../../src/zoo/tasks/cv/image.ts';
import type { BoundingBox } from '../../../../src/zoo/tasks/cv/ops/box.ts';
import { createImagePreprocessor } from '../../../../src/zoo/tasks/cv/utils/imagePreprocessor.ts';
import type { ImageInput } from '../../../../src/zoo/tasks/gpu.ts';
import { Yolo26Model, type Yolo26Task } from '../../../../src/zoo/yolo26/model.ts';
import { randomSafetensors } from '../../randomWeights.ts';

/** Small enough to run fast, a multiple of 32: 84 cells. */
export const YOLO_INPUT_SIZE = 64;

/** A YOLO26-n `task` checkpoint with random weights, in memory. */
export function yoloWeights(task: Yolo26Task) {
  const template = new Yolo26Model('n', { task });
  // Scaled down so f16 activations stay finite through the random layers.
  const weights = randomSafetensors(template, { scale: 0.1 });
  template.dispose();
  return weights;
}

/** A source buffer holding `image` as a YOLO_INPUT_SIZE runner's letterbox
 *  makes it, destroyed when the test ends. */
export function modelInput(root: TgpuRoot, schema: ImageInput, image: ImageBuffer) {
  const pixels = createImagePreprocessor(
    { resizeMode: 'letterbox' },
    { width: YOLO_INPUT_SIZE, height: YOLO_INPUT_SIZE },
  ).process(image);
  const source = root.createBuffer(schema).$usage('storage');
  source.write(new Float32Array(pixels).buffer);
  onTestFinished(() => source.destroy());
  return source;
}

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
  expect(detections.length).toBeGreaterThanOrEqual(want.length);
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
