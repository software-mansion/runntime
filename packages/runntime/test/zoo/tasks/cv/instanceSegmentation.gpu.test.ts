/** encodeSegmentInstances against segmentInstances on the same model input:
 *  the GPU decode lists the objects the CPU one returns, with masks that
 *  cover the same pixels. Random weights, so the values mean nothing; the
 *  two paths must agree. */
import { describe, expect, it, onTestFinished } from 'vitest';
import { imageBuffer, type ImageBuffer } from '../../../../src/zoo/tasks/cv/image.ts';
import {
  createInstanceSegmenter,
  GPU_MASK_SIZE,
} from '../../../../src/zoo/tasks/cv/instanceSegmentation.ts';
import { createImagePreprocessor } from '../../../../src/zoo/tasks/cv/utils/imagePreprocessor.ts';
import { Yolo26Model } from '../../../../src/zoo/yolo26/model.ts';
import { expectClose, gpu, requireFeature } from '../../../gpu.ts';
import { randomSafetensors } from '../../randomWeights.ts';
import { randomRgba } from './images.ts';

const { root } = await gpu();
/** Small enough to run fast, a multiple of 32: 84 cells. */
const INPUT_SIZE = 64;
/** Mean difference allowed between the two masks, 0 to 1. The CPU mask is
 *  bytes at prototype resolution, resampled once more here, so it differs
 *  by rounding and along the box edge. */
const MASK_TOLERANCE = 0.02;

/** `mask` read bilinearly at the GPU mask's value centers, 0 to 1. */
function resample(mask: ImageBuffer, size: number): number[] {
  const { width: w, height: h, data } = mask;
  const at = (x: number, y: number) =>
    data[Math.min(h - 1, Math.max(0, y)) * w + Math.min(w - 1, Math.max(0, x))]! / 255;
  const out: number[] = [];
  for (let j = 0; j < size; j++) {
    for (let i = 0; i < size; i++) {
      const u = ((i + 0.5) / size) * w - 0.5;
      const v = ((j + 0.5) / size) * h - 0.5;
      const x = Math.floor(u);
      const y = Math.floor(v);
      const top = at(x, y) * (x + 1 - u) + at(x + 1, y) * (u - x);
      const bot = at(x, y + 1) * (x + 1 - u) + at(x + 1, y + 1) * (u - x);
      out.push(top * (y + 1 - v) + bot * (v - y));
    }
  }
  return out;
}

describe('encodeSegmentInstances', () => {
  it.for([
    { confidenceThreshold: 0.3, maxDetections: 20 },
    { confidenceThreshold: 0.5, maxDetections: 3 },
  ])(
    'lists what segmentInstances returns, $confidenceThreshold and $maxDetections',
    async (opts, ctx) => {
      requireFeature(ctx, root, 'shader-f16');
      const template = new Yolo26Model('n', { task: 'segment' });
      // Scaled down so f16 activations stay finite through the random layers.
      const weights = randomSafetensors(template, { scale: 0.1 });
      template.dispose();
      const runner = await createInstanceSegmenter({ modelPath: weights, inputSize: INPUT_SIZE });
      // The input size, so the letterbox is the identity on both paths.
      const image = imageBuffer(randomRgba(INPUT_SIZE, INPUT_SIZE, 3), INPUT_SIZE, INPUT_SIZE);
      const pixels = createImagePreprocessor(
        { resizeMode: 'letterbox' },
        { width: INPUT_SIZE, height: INPUT_SIZE },
      ).process(image);
      const source = root.createBuffer(runner.schemas.input).$usage('storage');
      source.write(new Float32Array(pixels).buffer);
      const target = root.createBuffer(runner.schemas.output(20)).$usage('storage');
      onTestFinished(() => {
        source.destroy();
        target.destroy();
        runner.dispose();
      });

      const want = await runner.segmentInstances(image, opts);
      runner.encodeSegmentInstances({ source, target, ...opts });
      const got = await target.read();

      expect(want.length).toBeGreaterThan(0);
      expect(got.count).toBe(want.length);
      want.forEach((w, i) => {
        const g = got.detections[i]!;
        expect(g.classId).toBe(w.classId);
        expectClose(
          [g.box.xmin, g.box.ymin, g.box.xmax, g.box.ymax, g.confidence],
          [w.box.xmin, w.box.ymin, w.box.xmax, w.box.ymax, w.confidence],
          { atol: 1e-3, rtol: 1e-4 },
        );
        const cpu = resample(w.mask, GPU_MASK_SIZE);
        const diff = g.mask.reduce((sum, v, k) => sum + Math.abs(v - cpu[k]!), 0) / cpu.length;
        expect(diff).toBeLessThan(MASK_TOLERANCE);
      });
    },
  );
});
