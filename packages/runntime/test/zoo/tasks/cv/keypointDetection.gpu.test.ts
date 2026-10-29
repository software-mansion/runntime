/** encodeDetectKeypoints against detectKeypoints on the same model input:
 *  the GPU decode lists the people and landmarks the CPU one returns.
 *  Random weights, so the values mean nothing; the two paths must agree. */
import { describe, expect, it, onTestFinished } from 'vitest';
import { imageBuffer } from '../../../../src/zoo/tasks/cv/image.ts';
import { createKeypointDetector } from '../../../../src/zoo/tasks/cv/keypointDetection.ts';
import { createImagePreprocessor } from '../../../../src/zoo/tasks/cv/utils/imagePreprocessor.ts';
import { COCO_LANDMARKS } from '../../../../src/zoo/yolo26/pipeline.ts';
import { Yolo26Model } from '../../../../src/zoo/yolo26/model.ts';
import { expectClose, gpu, requireFeature } from '../../../gpu.ts';
import { randomSafetensors } from '../../randomWeights.ts';
import { randomRgba } from './images.ts';

const { root } = await gpu();
/** Small enough to run fast, a multiple of 32: 84 cells. */
const INPUT_SIZE = 64;

describe('encodeDetectKeypoints', () => {
  it.for([
    { confidenceThreshold: 0.3, maxDetections: 84 },
    { confidenceThreshold: 0.4, maxDetections: 5 },
  ])(
    'lists what detectKeypoints returns, $confidenceThreshold and $maxDetections',
    async (opts, ctx) => {
      requireFeature(ctx, root, 'shader-f16');
      const template = new Yolo26Model('n', { task: 'pose' });
      // Scaled down so f16 activations stay finite through the random layers.
      const weights = randomSafetensors(template, { scale: 0.1 });
      template.dispose();
      const runner = await createKeypointDetector({ modelPath: weights, inputSize: INPUT_SIZE });
      // The input size, so the letterbox is the identity on both paths.
      const image = imageBuffer(randomRgba(INPUT_SIZE, INPUT_SIZE, 3), INPUT_SIZE, INPUT_SIZE);
      const pixels = createImagePreprocessor(
        { resizeMode: 'letterbox' },
        { width: INPUT_SIZE, height: INPUT_SIZE },
      ).process(image);
      const source = root.createBuffer(runner.schemas.input).$usage('storage');
      source.write(new Float32Array(pixels).buffer);
      const target = root.createBuffer(runner.schemas.output(84)).$usage('storage');
      onTestFinished(() => {
        source.destroy();
        target.destroy();
        runner.dispose();
      });

      const want = await runner.detectKeypoints(image, opts);
      runner.encodeDetectKeypoints({ source, target, ...opts });
      const got = await target.read();

      expect(want.length).toBeGreaterThan(0);
      expect(got.count).toBe(want.length);
      want.forEach((w, i) => {
        const g = got.detections[i]!;
        expectClose(
          [g.box.xmin, g.box.ymin, g.box.xmax, g.box.ymax, g.confidence],
          [w.box.xmin, w.box.ymin, w.box.xmax, w.box.ymax, w.confidence],
          { atol: 1e-3, rtol: 1e-4 },
        );
        expectClose(
          g.landmarks.flatMap((l) => [l.x, l.y, l.confidence]),
          COCO_LANDMARKS.flatMap((name) => {
            const l = w.landmarks[name];
            return [l.x, l.y, l.confidence];
          }),
          { atol: 1e-3, rtol: 1e-4 },
        );
      });
    },
  );
});
