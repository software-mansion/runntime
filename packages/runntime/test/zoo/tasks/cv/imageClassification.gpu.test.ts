/** encodeClassify against classify on the same model input: the GPU list
 *  has the classes and confidences classify() returns, best first. Random
 *  weights, so the values mean nothing; the two paths must agree. */
import { describe, expect, it, onTestFinished } from 'vitest';
import {
  IMAGENET_MEAN,
  IMAGENET_STD,
  MOBILENETV4_CONV_S,
} from '../../../../src/zoo/mobilenetv4/config.ts';
import { MobileNetV4Model } from '../../../../src/zoo/mobilenetv4/model.ts';
import { imageBuffer } from '../../../../src/zoo/tasks/cv/image.ts';
import { createImageClassifier } from '../../../../src/zoo/tasks/cv/imageClassification.ts';
import { createImagePreprocessor } from '../../../../src/zoo/tasks/cv/utils/imagePreprocessor.ts';
import { expectClose, gpu, requireFeature } from '../../../gpu.ts';
import { randomSafetensors } from '../../randomWeights.ts';
import { randomRgba } from './images.ts';

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

describe('encodeClassify', () => {
  it.for([
    { name: 'the 5 best', capacity: 5, topk: undefined },
    { name: 'topk under what the target holds', capacity: 10, topk: 3 },
    { name: 'every class', capacity: numClasses, topk: undefined },
  ])('lists what classify returns, $name', async ({ capacity, topk }, ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const runner = await classifier();
    const image = imageBuffer(randomRgba(SIZE, SIZE, 3), SIZE, SIZE);
    // The runner's normalization: (byte / 255 − mean) / std.
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
    const source = root.createBuffer(runner.schemas.input).$usage('storage');
    source.write(new Float32Array(pixels).buffer);
    const target = root.createBuffer(runner.schemas.output(capacity)).$usage('storage');
    onTestFinished(() => {
      source.destroy();
      target.destroy();
    });

    const k = topk ?? capacity;
    const want = await runner.classify(image, { topk: k });
    runner.encodeClassify({ source, target, topk });
    const got = (await target.read()).slice(0, k);

    expect(got.map((c) => c.classId)).toEqual(want.map((c) => c.classId));
    expectClose(
      got.map((c) => c.confidence),
      want.map((c) => c.confidence),
      { atol: 1e-7, rtol: 1e-4 },
    );
  });

  it('rejects a topk past what the target holds', async (ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const runner = await classifier();
    const source = root.createBuffer(runner.schemas.input).$usage('storage');
    const target = root.createBuffer(runner.schemas.output(5)).$usage('storage');
    onTestFinished(() => {
      source.destroy();
      target.destroy();
    });
    expect(() => runner.encodeClassify({ source, target, topk: 6 })).toThrow(
      expect.objectContaining({ code: 'INVALID_ARGUMENT' }),
    );
  });
});
