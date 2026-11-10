/** encodeClassify against classify on the same model input: the GPU list
 *  has the classes and confidences classify() returns, best first. Random
 *  weights, so the values mean nothing; the two paths must agree. */
import type { d } from 'typegpu';
import { describe, expect, it, onTestFinished } from 'vitest';
import {
  IMAGENET_MEAN,
  IMAGENET_STD,
  MOBILENETV4_CONV_S,
} from '../../../../src/zoo/mobilenetv4/config.ts';
import { MobileNetV4Model } from '../../../../src/zoo/mobilenetv4/model.ts';
import type { Classification } from '../../../../src/zoo/mobilenetv4/pipeline.ts';
import { imageBuffer, type ImageBuffer } from '../../../../src/zoo/tasks/cv/image.ts';
import {
  createImageClassifier,
  type ClassificationList,
} from '../../../../src/zoo/tasks/cv/imageClassification.ts';
import { createImagePreprocessor } from '../../../../src/zoo/tasks/cv/utils/imagePreprocessor.ts';
import type { ImageInput } from '../../../../src/zoo/tasks/gpu.ts';
import { expectClose, gpu, requireFeature } from '../../../gpu.ts';
import { randomSafetensors } from '../../randomWeights.ts';
import { ENCODERS, freed } from '../gpu.ts';
import { frameImage, randomRgba, videoFrame } from './images.ts';

const { root } = await gpu();
const { inputSize: SIZE, numClasses, headHidden } = MOBILENETV4_CONV_S;

async function classifier() {
  const template = new MobileNetV4Model(MOBILENETV4_CONV_S, { root });
  const weights = randomSafetensors(template, {
    scale: 0.1,
    // The checkpoint holds the classifier as [out, in]; the model keeps it [in, out].
    shapes: { 'classifier.weight': [numClasses, headHidden] },
  });
  template.dispose();
  // Stretch at the input size, so the fit is the identity on both paths.
  const runner = await createImageClassifier({ modelPath: weights, resizeMode: 'stretch' });
  onTestFinished(() => runner.dispose());
  return runner;
}

/** A source buffer holding `image` as the runner's preprocessing makes it:
 *  stretched, then (byte / 255 − mean) / std. */
function modelInput(schema: ImageInput, image: ImageBuffer) {
  const pixels = createImagePreprocessor(
    {
      resizeMode: 'stretch',
      normalizeOpts: {
        alpha: IMAGENET_STD.map((s) => 1 / (255 * s)),
        beta: IMAGENET_MEAN.map((m, c) => -m / IMAGENET_STD[c]!),
      },
    },
    { width: SIZE, height: SIZE },
  ).process(image);
  const source = root.createBuffer(schema).$usage('storage');
  source.write(new Float32Array(pixels).buffer);
  onTestFinished(() => source.destroy());
  return source;
}

/** The first `want.length` GPU entries are the CPU classes, in order, with
 *  confidences within `rtol`. */
function expectSameClasses(
  got: d.Infer<ClassificationList>,
  want: readonly Classification[],
  rtol = 1e-4,
): void {
  const top = got.slice(0, want.length);
  expect(top.map((c) => c.classId)).toEqual(want.map((c) => c.classId));
  expectClose(
    top.map((c) => c.confidence),
    want.map((c) => c.confidence),
    { atol: 1e-7, rtol },
  );
}

/** Chrome hands the import the same bytes a canvas reads from the frame
 *  (measured: none differ), but a frame stretched into the input picks up
 *  rounding differences between the GPU and CPU bilinear fits, which the
 *  random f16 net turns into up to 5.1e-4 on a confidence, relative, over
 *  12 frames of four sizes. Twice that. */
const FRAME_RTOL = 1e-3;

describe('encodeClassify', () => {
  it.for([
    { name: 'the 5 best', capacity: 5, topk: undefined },
    { name: 'topk under what the target holds', capacity: 10, topk: 3 },
    { name: 'every class', capacity: numClasses, topk: undefined },
  ])('lists what classify returns, $name', async ({ capacity, topk }, ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const runner = await classifier();
    const image = imageBuffer(randomRgba(SIZE, SIZE, 3), SIZE, SIZE);
    const source = modelInput(runner.schemas.input, image);
    const target = freed(root.createBuffer(runner.schemas.output(capacity)).$usage('storage'));

    const want = await runner.classify(image, { topk: topk ?? capacity });
    runner.encodeClassify({ source, target, topk });
    expectSameClasses(await target.read(), want);
  });

  // The route the docs show: a camera frame imported as an external texture.
  it.for([
    { name: 'the input size', w: SIZE, h: SIZE },
    { name: 'a wide frame, stretched', w: 320, h: 180 },
  ])('lists what classify returns for a video frame of $name', async ({ w, h }, ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const runner = await classifier();
    const frame = videoFrame(w, h, 11);
    const target = freed(root.createBuffer(runner.schemas.output(5)).$usage('storage'));

    const want = await runner.classify(frameImage(frame), { topk: 5 });
    runner.encodeClassify({ source: root.device.importExternalTexture({ source: frame }), target });
    expectSameClasses(await target.read(), want, FRAME_RTOL);
  });

  it.for(ENCODERS)('gives the same list recorded into $name', async ({ open }, ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const runner = await classifier();
    const image = imageBuffer(randomRgba(SIZE, SIZE, 3), SIZE, SIZE);
    const source = modelInput(runner.schemas.input, image);
    const own = freed(root.createBuffer(runner.schemas.output(5)).$usage('storage'));
    const target = freed(root.createBuffer(runner.schemas.output(5)).$usage('storage'));

    runner.encodeClassify({ source, target: own });
    const { encoder, submit } = open(root);
    runner.encodeClassify({ source, target, encoder });
    // Recorded, not run: nothing reaches the target until the caller submits.
    expect((await target.read()).every((c) => c.confidence === 0)).toBe(true);
    submit();
    expect(await target.read()).toEqual(await own.read());
  });

  // Two calls in one command buffer share the runner's input and logits, so
  // the first must be done with them before the second starts.
  it('runs two calls recorded into one encoder', async (ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const runner = await classifier();
    const images = [3, 4].map((seed) => imageBuffer(randomRgba(SIZE, SIZE, seed), SIZE, SIZE));
    const sources = images.map((image) => modelInput(runner.schemas.input, image));
    const targets = images.map(() =>
      freed(root.createBuffer(runner.schemas.output(5)).$usage('storage')),
    );

    const encoder = root.device.createCommandEncoder();
    sources.forEach((source, i) => runner.encodeClassify({ source, target: targets[i]!, encoder }));
    root.device.queue.submit([encoder.finish()]);
    for (const [i, image] of images.entries()) {
      expectSameClasses(await targets[i]!.read(), await runner.classify(image, { topk: 5 }));
    }
  });

  it('rejects a topk past what the target holds', async (ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const runner = await classifier();
    const source = freed(root.createBuffer(runner.schemas.input).$usage('storage'));
    const target = freed(root.createBuffer(runner.schemas.output(5)).$usage('storage'));
    expect(() => runner.encodeClassify({ source, target, topk: 6 })).toThrow(
      expect.objectContaining({ code: 'INVALID_ARGUMENT' }),
    );
  });
});
