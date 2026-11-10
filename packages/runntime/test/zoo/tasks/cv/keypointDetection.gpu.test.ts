/** encodeDetectKeypoints against detectKeypoints on the same model input:
 *  the GPU decode lists the people and landmarks the CPU one returns.
 *  Random weights, so the values mean nothing; the two paths must agree. */
import type { d } from 'typegpu';
import { describe, expect, it, onTestFinished } from 'vitest';
import { imageBuffer, type ImageBuffer } from '../../../../src/zoo/tasks/cv/image.ts';
import {
  createKeypointDetector,
  type KeypointDetection,
  type KeypointDetectionList,
} from '../../../../src/zoo/tasks/cv/keypointDetection.ts';
import { createImagePreprocessor } from '../../../../src/zoo/tasks/cv/utils/imagePreprocessor.ts';
import type { ImageInput } from '../../../../src/zoo/tasks/gpu.ts';
import { COCO_LANDMARKS } from '../../../../src/zoo/yolo26/pipeline.ts';
import { Yolo26Model } from '../../../../src/zoo/yolo26/model.ts';
import { expectClose, gpu, requireFeature } from '../../../gpu.ts';
import { randomSafetensors } from '../../randomWeights.ts';
import { ENCODERS } from '../gpu.ts';
import { byNearestBox, FRAME_TOLERANCE } from './detections.ts';
import { frameImage, randomRgba, videoFrame } from './images.ts';

const { root } = await gpu();
/** Small enough to run fast, a multiple of 32: 84 cells. */
const INPUT_SIZE = 64;

async function detector() {
  const template = new Yolo26Model('n', { task: 'pose' });
  // Scaled down so f16 activations stay finite through the random layers.
  const weights = randomSafetensors(template, { scale: 0.1 });
  template.dispose();
  const runner = await createKeypointDetector({ modelPath: weights, inputSize: INPUT_SIZE });
  onTestFinished(() => runner.dispose());
  return runner;
}

/** A source buffer holding `image` as the runner's letterbox makes it. */
function modelInput(schema: ImageInput, image: ImageBuffer) {
  const pixels = createImagePreprocessor(
    { resizeMode: 'letterbox' },
    { width: INPUT_SIZE, height: INPUT_SIZE },
  ).process(image);
  const source = root.createBuffer(schema).$usage('storage');
  source.write(new Float32Array(pixels).buffer);
  onTestFinished(() => source.destroy());
  return source;
}

function outputBuffer(schema: KeypointDetectionList) {
  const target = root.createBuffer(schema).$usage('storage');
  onTestFinished(() => target.destroy());
  return target;
}

/** The GPU list holds the CPU detections, in order, within `pixels` on the
 *  boxes and landmarks and `confidence` on the scores. */
function expectSameDetections(
  got: d.Infer<KeypointDetectionList>,
  want: readonly KeypointDetection[],
  { pixels, confidence } = { pixels: 1e-3, confidence: 1e-3 },
): void {
  expect(want.length).toBeGreaterThan(0);
  expect(got.count).toBe(want.length);
  want.forEach((w, i) => {
    const g = got.detections[i]!;
    const landmarks = COCO_LANDMARKS.map((name) => w.landmarks[name]);
    expectClose(
      [g.box.xmin, g.box.ymin, g.box.xmax, g.box.ymax, ...g.landmarks.flatMap((l) => [l.x, l.y])],
      [w.box.xmin, w.box.ymin, w.box.xmax, w.box.ymax, ...landmarks.flatMap((l) => [l.x, l.y])],
      { atol: pixels, rtol: 1e-4 },
    );
    expectClose(
      [g.confidence, ...g.landmarks.map((l) => l.confidence)],
      [w.confidence, ...landmarks.map((l) => l.confidence)],
      { atol: confidence, rtol: 1e-4 },
    );
  });
}

describe('encodeDetectKeypoints', () => {
  it.for([
    { confidenceThreshold: 0.3, maxDetections: 84 },
    { confidenceThreshold: 0.4, maxDetections: 5 },
  ])(
    'lists what detectKeypoints returns, $confidenceThreshold and $maxDetections',
    async (opts, ctx) => {
      requireFeature(ctx, root, 'shader-f16');
      const runner = await detector();
      // The input size, so the letterbox is the identity on both paths.
      const image = imageBuffer(randomRgba(INPUT_SIZE, INPUT_SIZE, 3), INPUT_SIZE, INPUT_SIZE);
      const source = modelInput(runner.schemas.input, image);
      const target = outputBuffer(runner.schemas.output(84));

      const want = await runner.detectKeypoints(image, opts);
      runner.encodeDetectKeypoints({ source, target, ...opts });
      expectSameDetections(await target.read(), want);
    },
  );

  // The route the docs show: a camera frame imported as an external texture.
  it.for([
    { name: 'the input size', w: 64, h: 64 },
    { name: 'a wide frame, letterboxed', w: 96, h: 48 },
  ])('lists what detectKeypoints returns for a video frame of $name', async ({ w, h }, ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const runner = await detector();
    const frame = videoFrame(w, h, 11);
    const target = outputBuffer(runner.schemas.output(84));

    const want = await runner.detectKeypoints(frameImage(frame), { confidenceThreshold: 0.3 });
    runner.encodeDetectKeypoints({
      source: root.device.importExternalTexture({ source: frame }),
      target,
      confidenceThreshold: 0.3,
    });
    const got = await target.read();
    expectSameDetections(
      { count: got.count, detections: byNearestBox(got.detections.slice(0, got.count), want) },
      want,
      FRAME_TOLERANCE,
    );
  });

  it.for(ENCODERS)('gives the same list recorded into $name', async ({ record }, ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const runner = await detector();
    const image = imageBuffer(randomRgba(INPUT_SIZE, INPUT_SIZE, 3), INPUT_SIZE, INPUT_SIZE);
    const source = modelInput(runner.schemas.input, image);
    const own = outputBuffer(runner.schemas.output(84));
    const target = outputBuffer(runner.schemas.output(84));

    runner.encodeDetectKeypoints({ source, target: own });
    record(root, (encoder) => runner.encodeDetectKeypoints({ source, target, encoder }));
    expect(await target.read()).toEqual(await own.read());
  });

  // Two calls in one command buffer share the runner's input and decode
  // scratch, so the first must be done with them before the second starts.
  it('runs two calls recorded into one encoder', async (ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const runner = await detector();
    const images = [3, 4].map((seed) =>
      imageBuffer(randomRgba(INPUT_SIZE, INPUT_SIZE, seed), INPUT_SIZE, INPUT_SIZE),
    );
    const sources = images.map((image) => modelInput(runner.schemas.input, image));
    const targets = images.map(() => outputBuffer(runner.schemas.output(84)));

    const encoder = root.device.createCommandEncoder();
    sources.forEach((source, i) =>
      runner.encodeDetectKeypoints({ source, target: targets[i]!, encoder }),
    );
    root.device.queue.submit([encoder.finish()]);
    for (const [i, image] of images.entries()) {
      expectSameDetections(await targets[i]!.read(), await runner.detectKeypoints(image));
    }
  });
});
