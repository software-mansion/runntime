/** Instance segmentation task: an image in, the objects in it out, each
 *  with a box and a pixel mask. Runs YOLO26 segment on the initRunntime() device. */

import { createResourceScope } from '../../../core/index.ts';
import { openWeights, throwIfAborted, type LoadOptions, type ModelPath } from '../../load.ts';
import type { Yolo26Variant } from '../../yolo26/config.ts';
import { createDetector, SEGMENT_MASKS } from '../../yolo26/detector.ts';
import { COCO_NAMES, decodeSegmentations } from '../../yolo26/pipeline.ts';
import { models } from '../../models.ts';
import type { ImageBuffer, ResizeMode } from './image.ts';
import { decodeBox, scaleBox, type BoundingBox } from './ops/box.ts';
import { resizeTransform } from './ops/point.ts';
import { createImagePreprocessor } from './utils/imagePreprocessor.ts';

/** What to load. Every field is optional; a missing `modelPath` comes from
 *  models.instanceSegmentation.YOLO26_SEG.DEFAULT. */
export interface InstanceSegmenterModel {
  /** The weights: a safetensors export of a YOLO26 segment checkpoint. */
  readonly modelPath?: ModelPath;
  /** Model size, `n` to `x`. Default: read from the weights. */
  readonly variant?: Yolo26Variant;
  /** Input width and height in pixels, a multiple of 32. Default 640.
   *  Smaller runs faster and misses more small objects. */
  readonly inputSize?: number;
  /** How an image is fitted into the model input. Default `letterbox`. */
  readonly resizeMode?: ResizeMode;
}

export interface SegmentInstancesOptions {
  /** Drops objects scored below this, 0 to 1. Default 0.3. */
  readonly confidenceThreshold?: number;
  /** At most this many objects, best first. Default 300. */
  readonly maxDetections?: number;
}

/** One object found in the image. */
export interface InstanceSegmentation {
  /** The class name, like `person` or `bus`. */
  readonly label: string;
  /** The class index into `labels`. */
  readonly classId: number;
  /** The score, 0 to 1. */
  readonly confidence: number;
  /** Where it sits, in pixels of the input image. */
  readonly box: BoundingBox<'xyxy'>;
  /** Which pixels of `box` are the object: a `gray` image that covers the
   *  box, at about a quarter of the model input's resolution. 0 is outside
   *  the object, 255 inside, in between along the edge. Stretch it over
   *  `box` when drawing. */
  readonly mask: ImageBuffer;
}

export interface InstanceSegmenter {
  /** The class names, by class index. */
  readonly labels: readonly string[];
  /** Image in, the objects in it out, best first. */
  segmentInstances(
    image: ImageBuffer,
    options?: SegmentInstancesOptions,
  ): Promise<InstanceSegmentation[]>;
  /** Frees the weights and every GPU buffer the runner holds. */
  dispose(): void;
}

/** Loads the checkpoint, warms the GPU path once, and returns the runner.
 *  Needs a device with `shader-f16`. */
export async function createInstanceSegmenter(
  config: InstanceSegmenterModel = {},
  opts: LoadOptions = {},
): Promise<InstanceSegmenter> {
  const scope = createResourceScope();
  try {
    const {
      modelPath = models.instanceSegmentation.YOLO26_SEG.DEFAULT.modelPath,
      variant,
      inputSize: size,
      resizeMode = 'letterbox',
    } = config;

    // Cache key. The whole path, so same-named files of two sizes stay
    // apart. Bump the version to drop bytes cached under the old one.
    const path = typeof modelPath === 'string' ? modelPath.replace(/^https?:\/\//, '') : undefined;
    const sd = await openWeights(modelPath, {
      ...opts,
      cacheId: opts.cacheId ?? (path && `yolo26-seg-v1/${path}`),
    });

    throwIfAborted(opts.signal);
    const detector = scope.track(
      await createDetector(sd, {
        task: 'segment',
        variant,
        inputSize: size,
        onProgress: opts.onProgress,
      }),
    );
    const { inputSize, numClasses } = detector;
    const labels =
      numClasses === COCO_NAMES.length
        ? COCO_NAMES
        : Array.from({ length: numClasses }, (_, i) => `class ${i}`);
    const preprocessor = createImagePreprocessor(
      { resizeMode },
      { width: inputSize, height: inputSize },
    );
    // One scratch array, rewritten each call; the detector copies it to the GPU.
    const pixels = new Float32Array(3 * inputSize * inputSize);

    let disposed = false;
    // Calls run one after another: the next preprocess overwrites the scratch
    // array only after the previous run took it.
    let queue: Promise<unknown> = Promise.resolve();
    return {
      labels,
      async segmentInstances(image, options = {}) {
        if (disposed) throw new Error('instance segmenter is disposed');
        const { confidenceThreshold = 0.3, maxDetections = 300 } = options;
        const run = queue.then(() => detector.run(preprocessor.process(image, pixels)));
        queue = run.catch(() => undefined);
        const { levels, proto } = await run;
        if (!proto) throw new Error('yolo26 segment: the forward pass gave no mask prototypes');
        const scale = preprocessor.scaleOptions(image);
        const toInput = resizeTransform(scale);
        // Mask maps are at the prototype resolution, a quarter of the input.
        const protoStride = inputSize / proto.w;
        return decodeSegmentations(levels, proto, {
          numClasses,
          numMasks: SEGMENT_MASKS,
          confThreshold: confidenceThreshold,
          maxDet: maxDetections,
          inputSize,
        }).map((d) => {
          const raw = scaleBox(decodeBox([d.x1, d.y1, d.x2, d.y2], 'xyxy'), scale);
          // Boxes can spill into the padding; cut them at the image edge.
          const box: BoundingBox<'xyxy'> = {
            format: 'xyxy',
            xmin: clamp(raw.xmin, 0, image.width),
            ymin: clamp(raw.ymin, 0, image.height),
            xmax: clamp(raw.xmax, 0, image.width),
            ymax: clamp(raw.ymax, 0, image.height),
          };
          // The decoded map covers whole prototype cells around the box;
          // resample it so the mask covers exactly the box the caller gets.
          const cell = (v: number, axis: 'x' | 'y') =>
            axis === 'x'
              ? (v * toInput.scaleX + toInput.offsetX) / protoStride - d.maskX
              : (v * toInput.scaleY + toInput.offsetY) / protoStride - d.maskY;
          const mask = resampleMask(
            d.mask,
            d.maskW,
            d.maskH,
            cell(box.xmin, 'x'),
            cell(box.ymin, 'y'),
            cell(box.xmax, 'x'),
            cell(box.ymax, 'y'),
          );
          return { label: labels[d.classId]!, classId: d.classId, confidence: d.score, box, mask };
        });
      },
      dispose() {
        disposed = true;
        scope.dispose();
      },
    };
  } catch (err) {
    scope.dispose();
    throw err;
  }
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

/** Bilinear resample of the cell window [u0,u1)×[v0,v1) of a `w`×`h`
 *  coverage map into a gray image of the same cell count. */
function resampleMask(
  src: Uint8Array,
  w: number,
  h: number,
  u0: number,
  v0: number,
  u1: number,
  v1: number,
): ImageBuffer {
  const width = Math.max(1, Math.round(u1 - u0));
  const height = Math.max(1, Math.round(v1 - v0));
  const data = new Uint8Array(width * height);
  const at = (x: number, y: number) => src[clamp(y, 0, h - 1) * w + clamp(x, 0, w - 1)]!;
  const du = (u1 - u0) / width;
  const dv = (v1 - v0) / height;
  for (let j = 0; j < height; j++) {
    const v = v0 + (j + 0.5) * dv - 0.5;
    const y = Math.floor(v);
    const fy = v - y;
    for (let i = 0; i < width; i++) {
      const u = u0 + (i + 0.5) * du - 0.5;
      const x = Math.floor(u);
      const fx = u - x;
      const top = at(x, y) * (1 - fx) + at(x + 1, y) * fx;
      const bottom = at(x, y + 1) * (1 - fx) + at(x + 1, y + 1) * fx;
      data[j * width + i] = Math.round(top * (1 - fy) + bottom * fy);
    }
  }
  return { data, width, height, format: 'gray', layout: 'hwc' };
}
