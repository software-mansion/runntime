/** Depth estimation task: an image in, a relative depth map out. Runs
 *  DepthART on the initRunntime() device. */

import { createResourceScope, supportsF16 } from '../../../core/index.ts';
import { openWeights, throwIfAborted, type LoadOptions, type ModelPath } from '../../load.ts';
import {
  createEstimator,
  DEPTHART_CONFIGS,
  type DepthartVariant,
} from '../../depthart/estimator.ts';
import { models } from '../../models.ts';
import type { ImageBuffer, ResizeMode } from './image.ts';
import { createImagePreprocessor } from './utils/imagePreprocessor.ts';

/** What to load. Every field is optional; a missing `modelPath` comes from
 *  models.depthEstimation.DEPTHART.DEFAULT. */
export interface DepthEstimatorModel {
  /** The weights: a safetensors export of a DepthART relative checkpoint. */
  readonly modelPath?: ModelPath;
  /** Model size, `b` or `s`. Default: read from the weights. */
  readonly variant?: DepthartVariant;
  /** How an image is fitted into the model input. Default `stretch`, so
   *  the map covers the whole image. */
  readonly resizeMode?: ResizeMode;
}

/** Relative depth of the model input, one value per pixel, row by row.
 *  Smaller = nearer. Values have no unit: compare them within one map. */
export interface DepthMap {
  readonly width: number;
  readonly height: number;
  readonly data: Float32Array;
}

export interface DepthEstimator {
  /** Width and height of every returned map. */
  readonly inputSize: number;
  /** Image in, its depth map out. */
  estimateDepth(image: ImageBuffer): Promise<DepthMap>;
  /** Frees the weights and every GPU buffer the runner holds. */
  dispose(): void;
}

/** Loads the checkpoint, warms the GPU path once, and returns the runner. */
export async function createDepthEstimator(
  config: DepthEstimatorModel = {},
  opts: LoadOptions = {},
): Promise<DepthEstimator> {
  if (!supportsF16()) {
    throw new Error(
      "createDepthEstimator: this device has no shader-f16, which the conv kernels need. Request it with tgpu.init({ device: { optionalFeatures: ['shader-f16'] } })",
    );
  }
  const scope = createResourceScope();
  try {
    const {
      modelPath = models.depthEstimation.DEPTHART.DEFAULT.modelPath,
      variant,
      resizeMode = 'stretch',
    } = config;

    // Cache key. The whole path, so same-named files of two sizes stay
    // apart. Bump the version to drop bytes cached under the old one.
    const path = typeof modelPath === 'string' ? modelPath.replace(/^https?:\/\//, '') : undefined;
    const sd = await openWeights(modelPath, {
      ...opts,
      cacheId: opts.cacheId ?? (path && `depthart-v1/${path}`),
    });

    throwIfAborted(opts.signal);
    const estimator = scope.track(
      await createEstimator(sd, { variant, onProgress: opts.onProgress }),
    );
    const { inputSize } = estimator;
    const { mean, std } = DEPTHART_CONFIGS[estimator.variant];
    // (byte / 255 − mean) / std, per channel.
    const preprocessor = createImagePreprocessor(
      {
        resizeMode,
        normalizeOpts: {
          alpha: std.map((s) => 1 / (255 * s)),
          beta: mean.map((m, c) => -m / std[c]!),
        },
      },
      { width: inputSize, height: inputSize },
    );
    // One scratch array, rewritten each call; the estimator copies it to the GPU.
    const pixels = new Float32Array(3 * inputSize * inputSize);

    let disposed = false;
    // Calls run one after another: the next preprocess overwrites the scratch
    // array only after the previous run took it.
    let queue: Promise<unknown> = Promise.resolve();
    return {
      inputSize,
      async estimateDepth(image) {
        if (disposed) throw new Error('depth estimator is disposed');
        const run = queue.then(() =>
          estimator.run(preprocessor.process(image, pixels), inputSize, inputSize),
        );
        queue = run.catch(() => undefined);
        return { width: inputSize, height: inputSize, data: await run };
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
