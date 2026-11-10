/** encodeSegmentInstances against segmentInstances on the same model input:
 *  the GPU decode lists the objects the CPU one returns, with masks that
 *  cover the same pixels. Random weights, so the values mean nothing; the
 *  two paths must agree. */
import type { d } from 'typegpu';
import { describe, expect, it, onTestFinished } from 'vitest';
import { imageBuffer, type ImageBuffer } from '../../../../src/zoo/tasks/cv/image.ts';
import {
  createInstanceSegmenter,
  GPU_MASK_SIZE,
  type InstanceSegmentation,
  type InstanceSegmentationList,
} from '../../../../src/zoo/tasks/cv/instanceSegmentation.ts';
import { expectClose, gpu, requireFeature } from '../../../gpu.ts';
import { ENCODERS, freed } from '../gpu.ts';
import {
  byNearestBox,
  FRAME_TOLERANCE,
  modelInput,
  YOLO_INPUT_SIZE as INPUT_SIZE,
  yoloWeights,
} from './detections.ts';
import { frameImage, randomRgba, videoFrame } from './images.ts';

const { root } = await gpu();
/** Mean difference allowed between the two masks, 0 to 1. The CPU mask is
 *  bytes at prototype resolution, resampled once more here, so it differs
 *  by rounding and along the box edge. */
const MASK_TOLERANCE = 0.02;

async function segmenter() {
  const runner = await createInstanceSegmenter({
    modelPath: yoloWeights('segment'),
    inputSize: INPUT_SIZE,
  });
  onTestFinished(() => runner.dispose());
  return runner;
}

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

/** The GPU list holds the CPU detections, in order, within `pixels` on the
 *  boxes and `confidence` on the scores, with masks that cover the same
 *  pixels. */
function expectSameDetections(
  got: d.Infer<InstanceSegmentationList>,
  want: readonly InstanceSegmentation[],
  { pixels, confidence } = { pixels: 1e-3, confidence: 1e-3 },
): void {
  expect(want.length).toBeGreaterThan(0);
  expect(got.count).toBe(want.length);
  want.forEach((w, i) => {
    const g = got.detections[i]!;
    expect(g.classId).toBe(w.classId);
    expectClose(
      [g.box.xmin, g.box.ymin, g.box.xmax, g.box.ymax],
      [w.box.xmin, w.box.ymin, w.box.xmax, w.box.ymax],
      { atol: pixels, rtol: 1e-4 },
    );
    expectClose([g.confidence], [w.confidence], { atol: confidence, rtol: 1e-4 });
    const cpu = resample(w.mask, GPU_MASK_SIZE);
    const diff = g.mask.reduce((sum, v, k) => sum + Math.abs(v - cpu[k]!), 0) / cpu.length;
    expect(diff).toBeLessThan(MASK_TOLERANCE);
  });
}

describe('encodeSegmentInstances', () => {
  it.for([
    { confidenceThreshold: 0.3, maxDetections: 20 },
    { confidenceThreshold: 0.5, maxDetections: 3 },
  ])(
    'lists what segmentInstances returns, $confidenceThreshold and $maxDetections',
    async (opts, ctx) => {
      requireFeature(ctx, root, 'shader-f16');
      const runner = await segmenter();
      // The input size, so the letterbox is the identity on both paths.
      const image = imageBuffer(randomRgba(INPUT_SIZE, INPUT_SIZE, 3), INPUT_SIZE, INPUT_SIZE);
      const source = modelInput(root, runner.schemas.input, image);
      const target = freed(root.createBuffer(runner.schemas.output(20)).$usage('storage'));

      const want = await runner.segmentInstances(image, opts);
      runner.encodeSegmentInstances({ source, target, ...opts });
      expectSameDetections(await target.read(), want);
    },
  );

  // The route the docs show: a camera frame imported as an external texture.
  it.for([
    { name: 'the input size', w: 64, h: 64 },
    { name: 'a wide frame, letterboxed', w: 96, h: 48 },
  ])('lists what segmentInstances returns for a video frame of $name', async ({ w, h }, ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const runner = await segmenter();
    const frame = videoFrame(w, h, 11);
    const target = freed(root.createBuffer(runner.schemas.output(20)).$usage('storage'));
    const opts = { confidenceThreshold: 0.3, maxDetections: 20 };

    const want = await runner.segmentInstances(frameImage(frame), opts);
    runner.encodeSegmentInstances({
      source: root.device.importExternalTexture({ source: frame }),
      target,
      ...opts,
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
    const runner = await segmenter();
    const image = imageBuffer(randomRgba(INPUT_SIZE, INPUT_SIZE, 3), INPUT_SIZE, INPUT_SIZE);
    const source = modelInput(root, runner.schemas.input, image);
    const own = freed(root.createBuffer(runner.schemas.output(20)).$usage('storage'));
    const target = freed(root.createBuffer(runner.schemas.output(20)).$usage('storage'));

    runner.encodeSegmentInstances({ source, target: own });
    record(root, (encoder) => runner.encodeSegmentInstances({ source, target, encoder }));
    expect(await target.read()).toEqual(await own.read());
  });

  // Two calls in one command buffer share the runner's input and decode
  // scratch, so the first must be done with them before the second starts.
  it('runs two calls recorded into one encoder', async (ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const runner = await segmenter();
    const images = [3, 4].map((seed) =>
      imageBuffer(randomRgba(INPUT_SIZE, INPUT_SIZE, seed), INPUT_SIZE, INPUT_SIZE),
    );
    const sources = images.map((image) => modelInput(root, runner.schemas.input, image));
    const targets = images.map(() =>
      freed(root.createBuffer(runner.schemas.output(20)).$usage('storage')),
    );
    const opts = { confidenceThreshold: 0.3, maxDetections: 20 };

    const encoder = root.device.createCommandEncoder();
    sources.forEach((source, i) =>
      runner.encodeSegmentInstances({ source, target: targets[i]!, encoder, ...opts }),
    );
    root.device.queue.submit([encoder.finish()]);
    for (const [i, image] of images.entries()) {
      expectSameDetections(await targets[i]!.read(), await runner.segmentInstances(image, opts));
    }
  });
});
