/** Keypoint detection task: an image in, the people in it out, each with a
 *  box and 17 COCO landmarks. Runs YOLO26 pose on the initRunntime() device. */

import { createResourceScope, RunntimeError } from '../../../core/index.ts';
import { openWeights, throwIfAborted, type LoadOptions, type ModelPath } from '../../load.ts';
import { asLoadError, rethrowRunError } from '../../errors.ts';
import type { Yolo26Variant } from '../../yolo26/config.ts';
import { createDetector, POSE_KEYPOINTS } from '../../yolo26/detector.ts';
import {
  COCO_LANDMARKS,
  COCO_SKELETON,
  decodePoses,
  type CocoLandmark,
} from '../../yolo26/pipeline.ts';
import { models } from '../../models.ts';
import type { ImageBuffer, ResizeMode } from './image.ts';
import { decodeBox, scaleBox, type BoundingBox } from './ops/box.ts';
import { scalePoint, type Point } from './ops/point.ts';
import { createImagePreprocessor } from './utils/imagePreprocessor.ts';

/** What to load. Every field is optional; a missing `modelPath` comes from
 *  models.keypointDetection.YOLO26_POSE.DEFAULT. */
export interface KeypointDetectorModel {
  /** The weights: a safetensors export of a YOLO26 pose checkpoint. */
  readonly modelPath?: ModelPath;
  /** Model size, `n` to `x`. Default: read from the weights. */
  readonly variant?: Yolo26Variant;
  /** Input width and height in pixels, a multiple of 32. Default 640.
   *  Smaller runs faster and misses more small people. */
  readonly inputSize?: number;
  /** How an image is fitted into the model input. Default `letterbox`. */
  readonly resizeMode?: ResizeMode;
}

export interface DetectKeypointsOptions {
  /** Drops people scored below this, 0 to 1. Default 0.3. */
  readonly confidenceThreshold?: number;
  /** At most this many people, best first. Default 300. */
  readonly maxDetections?: number;
}

/** One landmark of a person, in pixels of the input image. */
export interface Landmark extends Point {
  /** How sure the model is the landmark is visible, 0 to 1. A hidden one
   *  keeps a position guess with a low value. */
  readonly confidence: number;
}

/** One person found in the image. */
export interface KeypointDetection {
  /** Where the person sits, in pixels of the input image. */
  readonly box: BoundingBox<'xyxy'>;
  /** The score, 0 to 1. */
  readonly confidence: number;
  /** The 17 COCO landmarks by name, `nose` to `rightAnkle`. A landmark
   *  outside the image keeps its position. */
  readonly landmarks: Readonly<Record<CocoLandmark, Landmark>>;
}

export interface KeypointDetector {
  /** The landmark names, in `COCO_LANDMARKS` order. */
  readonly landmarkNames: readonly CocoLandmark[];
  /** Pairs of landmark names to draw as limbs. */
  readonly skeleton: readonly (readonly [CocoLandmark, CocoLandmark])[];
  /** Image in, the people in it out, best first. */
  detectKeypoints(
    image: ImageBuffer,
    options?: DetectKeypointsOptions,
  ): Promise<KeypointDetection[]>;
  /** Frees the weights and every GPU buffer the runner holds. */
  dispose(): void;
}

/** Loads the checkpoint, warms the GPU path once, and returns the runner.
 *  Needs a device with `shader-f16`. */
export async function createKeypointDetector(
  config: KeypointDetectorModel = {},
  opts: LoadOptions = {},
): Promise<KeypointDetector> {
  const scope = createResourceScope();
  try {
    const {
      modelPath = models.keypointDetection.YOLO26_POSE.DEFAULT.modelPath,
      variant,
      inputSize: size,
      resizeMode = 'letterbox',
    } = config;

    // Cache key. The whole path, so same-named files of two sizes stay
    // apart. Bump the version to drop bytes cached under the old one.
    const path = typeof modelPath === 'string' ? modelPath.replace(/^https?:\/\//, '') : undefined;
    const sd = await openWeights(modelPath, {
      ...opts,
      cacheId: opts.cacheId ?? (path && `yolo26-pose-v1/${path}`),
    });

    throwIfAborted(opts.signal);
    const detector = scope.track(
      await createDetector(sd, {
        task: 'pose',
        variant,
        inputSize: size,
        onProgress: opts.onProgress,
      }),
    );
    const { inputSize, numClasses } = detector;
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
      landmarkNames: COCO_LANDMARKS,
      skeleton: COCO_SKELETON.map(([a, b]) => [COCO_LANDMARKS[a]!, COCO_LANDMARKS[b]!] as const),
      async detectKeypoints(image, options = {}) {
        if (disposed) throw new RunntimeError('RESOURCE_DISPOSED', 'keypoint detector is disposed');
        const { confidenceThreshold = 0.3, maxDetections = 300 } = options;
        const run = queue.then(() => detector.run(preprocessor.process(image, pixels)));
        queue = run.catch(() => undefined);
        const { levels } = await run.catch(rethrowRunError);
        const scale = preprocessor.scaleOptions(image);
        return decodePoses(levels, {
          numClasses,
          numKeypoints: POSE_KEYPOINTS,
          confThreshold: confidenceThreshold,
          maxDet: maxDetections,
        }).map((p) => {
          const box = scaleBox(decodeBox([p.x1, p.y1, p.x2, p.y2], 'xyxy'), scale);
          const landmarks = {} as Record<CocoLandmark, Landmark>;
          p.kpts.forEach((k, i) => {
            landmarks[COCO_LANDMARKS[i]!] = { ...scalePoint(k, scale), confidence: k.v };
          });
          return {
            // Boxes can spill into the padding; cut them at the image edge.
            box: {
              format: 'xyxy',
              xmin: clamp(box.xmin, 0, image.width),
              ymin: clamp(box.ymin, 0, image.height),
              xmax: clamp(box.xmax, 0, image.width),
              ymax: clamp(box.ymax, 0, image.height),
            },
            confidence: p.score,
            landmarks,
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
