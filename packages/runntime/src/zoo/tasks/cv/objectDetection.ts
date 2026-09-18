/** Object detection task: an image in, the objects in it out, as labeled
 *  boxes. Runs YOLO26 on the initRunntime() device. */

import { createResourceScope, RunntimeError } from '../../../core/index.ts';
import { openWeights, throwIfAborted, type LoadOptions, type ModelPath } from '../../load.ts';
import { asLoadError, rethrowRunError } from '../../errors.ts';
import type { Yolo26Variant } from '../../yolo26/config.ts';
import { createDetector } from '../../yolo26/detector.ts';
import { COCO_NAMES, decodeDetections } from '../../yolo26/pipeline.ts';
import { models } from '../../models.ts';
import type { ImageBuffer, ResizeMode } from './image.ts';
import { decodeBox, scaleBox, type BoundingBox } from './ops/box.ts';
import { createImagePreprocessor } from './utils/imagePreprocessor.ts';

/** What to load. Every field is optional; a missing `modelPath` comes from
 *  models.objectDetection.YOLO26.DEFAULT. */
export interface ObjectDetectorModel {
  /** The weights: a safetensors export of a YOLO26 detect checkpoint. */
  readonly modelPath?: ModelPath;
  /** Model size, `n` to `x`. Default: read from the weights. */
  readonly variant?: Yolo26Variant;
  /** Input width and height in pixels, a multiple of 32. Default 640.
   *  Smaller runs faster and misses more small objects. */
  readonly inputSize?: number;
  /** How an image is fitted into the model input. Default `letterbox`. */
  readonly resizeMode?: ResizeMode;
}

export interface DetectObjectsOptions {
  /** Drops objects scored below this, 0 to 1. Default 0.3. */
  readonly confidenceThreshold?: number;
  /** At most this many objects, best first. Default 300. */
  readonly maxDetections?: number;
}

/** One object found in the image. */
export interface ObjectDetection {
  /** The class name, like `person` or `bus`. */
  readonly label: string;
  /** The class index into `labels`. */
  readonly classId: number;
  /** The score, 0 to 1. */
  readonly confidence: number;
  /** Where it sits, in pixels of the input image. */
  readonly box: BoundingBox<'xyxy'>;
}

export interface ObjectDetector {
  /** The class names, by class index. */
  readonly labels: readonly string[];
  /** Image in, the objects in it out, best first. */
  detectObjects(image: ImageBuffer, options?: DetectObjectsOptions): Promise<ObjectDetection[]>;
  /** Frees the weights and every GPU buffer the runner holds. */
  dispose(): void;
}

/** Loads the checkpoint, warms the GPU path once, and returns the runner.
 *  Needs a device with `shader-f16`. */
export async function createObjectDetector(
  config: ObjectDetectorModel = {},
  opts: LoadOptions = {},
): Promise<ObjectDetector> {
  const scope = createResourceScope();
  try {
    const {
      modelPath = models.objectDetection.YOLO26.DEFAULT.modelPath,
      variant,
      inputSize: size,
      resizeMode = 'letterbox',
    } = config;

    // Cache key. The whole path, so same-named files of two sizes stay
    // apart. Bump the version to drop bytes cached under the old one.
    const path = typeof modelPath === 'string' ? modelPath.replace(/^https?:\/\//, '') : undefined;
    const sd = await openWeights(modelPath, {
      ...opts,
      cacheId: opts.cacheId ?? (path && `yolo26-v1/${path}`),
    });

    throwIfAborted(opts.signal);
    const detector = scope.track(
      await createDetector(sd, {
        task: 'detect',
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
      async detectObjects(image, options = {}) {
        if (disposed) throw new RunntimeError('RESOURCE_DISPOSED', 'object detector is disposed');
        const { confidenceThreshold = 0.3, maxDetections = 300 } = options;
        const run = queue.then(() => detector.run(preprocessor.process(image, pixels)));
        queue = run.catch(() => undefined);
        const { levels } = await run.catch(rethrowRunError);
        const scale = preprocessor.scaleOptions(image);
        return decodeDetections(levels, {
          numClasses,
          confThreshold: confidenceThreshold,
          maxDet: maxDetections,
        }).map((d) => {
          const box = scaleBox(decodeBox([d.x1, d.y1, d.x2, d.y2], 'xyxy'), scale);
          return {
            label: labels[d.classId]!,
            classId: d.classId,
            confidence: d.score,
            // Boxes can spill into the padding; cut them at the image edge.
            box: {
              format: 'xyxy',
              xmin: clamp(box.xmin, 0, image.width),
              ymin: clamp(box.ymin, 0, image.height),
              xmax: clamp(box.xmax, 0, image.width),
              ymax: clamp(box.ymax, 0, image.height),
            },
          };
        });
      },
      dispose() {
        disposed = true;
        scope.dispose();
      },
    };
  } catch (err) {
    scope.dispose();
    throw asLoadError(err);
  }
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
