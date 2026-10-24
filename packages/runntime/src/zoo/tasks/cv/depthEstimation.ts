/** Depth estimation task: an image in, a relative depth map out. Runs
 *  DepthART on the initRunntime() device. */

import { d } from 'typegpu';
import {
  createImageToChw,
  createMapToTexture,
  createResourceScope,
  defaultRoot,
  RunntimeError,
  supportsF16,
} from '../../../core/index.ts';
import { openWeights, throwIfAborted, type LoadOptions, type ModelPath } from '../../load.ts';
import { asLoadError, rethrowRunError } from '../../errors.ts';
import {
  createEstimator,
  DEPTHART_CONFIGS,
  type DepthartVariant,
} from '../../depthart/estimator.ts';
import { models } from '../../models.ts';
import type { ImageBuffer, ResizeMode } from './image.ts';
import { createImagePreprocessor } from './utils/imagePreprocessor.ts';
import {
  encodeWith,
  unwrapSource,
  unwrapTarget,
  type EncodeOptions,
  type GpuImage,
  type GpuImageTarget,
  type GpuTensor,
} from '../gpu.ts';

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

/** The model input as a GPU buffer: normalized RGB, `[channel][row][column]`
 *  f32, `inputSize` wide and high. */
export type DepthInput = d.WgslArray<d.WgslArray<d.WgslArray<d.F32>>>;

/** A depth map as a GPU buffer: `[row][column]` f32, `inputSize` wide and
 *  high, the same values as DepthMap.data. */
export type DepthMapData = d.WgslArray<d.WgslArray<d.F32>>;

export interface EncodeEstimateDepthOptions extends EncodeOptions<
  GpuImage | GpuTensor<DepthInput>,
  GpuImageTarget | GpuTensor<DepthMapData>
> {
  /** Texture targets only. `minmax` maps each map to [0, 1], nearest = 0.
   *  Default `none`: the raw values, which an unorm format clamps. */
  readonly normalize?: 'none' | 'minmax';
}

export interface DepthEstimator {
  /** Width and height of every returned map. */
  readonly inputSize: number;
  /** Buffer layouts for encodeEstimateDepth, at this runner's size. Pass one
   *  to `root.createBuffer` to make a buffer of the right type and size. */
  readonly schemas: { readonly input: DepthInput; readonly output: DepthMapData };
  /** Image in, its depth map out. */
  estimateDepth(image: ImageBuffer): Promise<DepthMap>;
  /** Records a depth estimate from GPU memory into GPU memory, with no CPU
   *  copy. A texture source is fitted like an ImageBuffer; a buffer source
   *  is the model input as-is. A texture target gets the map stretched to
   *  its size in every channel; a buffer target gets the map's values.
   *  Work submitted after it sees the result.
   *  @throws RunntimeError `INVALID_ARGUMENT` for a resource of the wrong
   *  format, usage or size. */
  encodeEstimateDepth(options: EncodeEstimateDepthOptions): void;
  /** Frees the weights and every GPU buffer the runner holds. */
  dispose(): void;
}

/** Loads the checkpoint, warms the GPU path once, and returns the runner. */
export async function createDepthEstimator(
  config: DepthEstimatorModel = {},
  opts: LoadOptions = {},
): Promise<DepthEstimator> {
  if (!supportsF16()) {
    throw new RunntimeError(
      'UNSUPPORTED_DEVICE',
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
    const root = defaultRoot();
    const imageToChw = createImageToChw(root);
    const mapToTexture = scope.track(createMapToTexture(root));
    const schemas = {
      input: d.arrayOf(d.arrayOf(d.arrayOf(d.f32, inputSize), inputSize), 3),
      output: d.arrayOf(d.arrayOf(d.f32, inputSize), inputSize),
    };
    const inputBytes = d.sizeOf(schemas.input);
    const outputBytes = d.sizeOf(schemas.output);

    let disposed = false;
    // Calls run one after another: the next preprocess overwrites the scratch
    // array only after the previous run took it.
    let queue: Promise<unknown> = Promise.resolve();
    return {
      inputSize,
      schemas,
      async estimateDepth(image) {
        if (disposed) throw new RunntimeError('RESOURCE_DISPOSED', 'depth estimator is disposed');
        const run = queue.then(() =>
          estimator.run(preprocessor.process(image, pixels), inputSize, inputSize),
        );
        queue = run.catch(() => undefined);
        return { width: inputSize, height: inputSize, data: await run.catch(rethrowRunError) };
      },
      encodeEstimateDepth({ source, target, encoder, normalize = 'none' }) {
        if (disposed) throw new RunntimeError('RESOURCE_DISPOSED', 'depth estimator is disposed');
        const src = unwrapSource(source, inputBytes);
        const dst = unwrapTarget(target, outputBytes);
        if ('tensor' in dst && normalize !== 'none') {
          throw new RunntimeError(
            'INVALID_ARGUMENT',
            'normalize: only a texture target is rescaled',
          );
        }
        try {
          encodeWith(encoder, (enc) =>
            estimator.encode(
              enc,
              (input) => {
                if ('tensor' in src) {
                  enc.copyBufferToBuffer(src.tensor, 0, root.unwrap(input), 0, inputBytes);
                } else {
                  imageToChw.encode(enc, src.image, input, preprocessor.gpuArgs);
                }
              },
              (depth) => {
                if ('tensor' in dst) {
                  enc.copyBufferToBuffer(root.unwrap(depth), 0, dst.tensor, 0, outputBytes);
                } else {
                  mapToTexture.encode(enc, depth, dst.image, {
                    width: inputSize,
                    height: inputSize,
                    normalize,
                  });
                }
              },
            ),
          );
        } catch (err) {
          rethrowRunError(err);
        }
      },
      dispose() {
        disposed = true;
        scope.dispose();
      },
    };
  } catch (err) {
    scope.dispose();
    throw asLoadError(err, opts.signal);
  }
}
