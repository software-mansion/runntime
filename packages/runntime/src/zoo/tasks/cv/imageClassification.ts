/** Image classification task: an image in, the classes it shows out, best
 *  first. Runs MobileNetV4 on the initRunntime() device. */

import { createResourceScope, RunntimeError, supportsF16 } from '../../../core/index.ts';
import { openWeights, throwIfAborted, type LoadOptions, type ModelPath } from '../../load.ts';
import { asLoadError, rethrowRunError } from '../../errors.ts';
import { createClassifier } from '../../mobilenetv4/classifier.ts';
import {
  IMAGENET_MEAN,
  IMAGENET_STD,
  MOBILENETV4_CROP_FRACTION,
} from '../../mobilenetv4/config.ts';
import { IMAGENET1K_LABELS } from '../../mobilenetv4/labels.ts';
import { type Classification, decodeTopK } from '../../mobilenetv4/pipeline.ts';
import { models } from '../../models.ts';
import type { ImageBuffer, ResizeMode } from './image.ts';
import { createImagePreprocessor } from './utils/imagePreprocessor.ts';

/** What to load. With no config, MobileNetV4-Conv-S from the Hugging Face
 *  Hub with the ImageNet class names. */
export interface ImageClassifierModel {
  /** The weights: a MobileNetV4-Conv-S safetensors file, 1000 ImageNet classes. */
  readonly modelPath?: ModelPath;
  /** The class names, by class index; as many as the model has classes.
   *  Default: the 1000 ImageNet names. */
  readonly labels?: readonly string[];
  /** How an image is fitted into the model input. Default `crop`, the
   *  middle 87.5% of the centered square, the crop the model was evaluated
   *  with; it keeps the subject's proportions. */
  readonly resizeMode?: ResizeMode;
}

export interface ClassifyOptions {
  /** How many classes to return, best first. Default: all of them. */
  readonly topk?: number;
}

export interface ImageClassifier {
  /** The class names, by class index. */
  readonly labels: readonly string[];
  /** Width and height of the model input in pixels. */
  readonly inputSize: number;
  /** Image in, its most likely classes out, best first. */
  classify(
    image: ImageBuffer,
    options?: ClassifyOptions,
  ): Promise<(Classification & { label: string })[]>;
  /** Frees the weights and every GPU buffer the runner holds. */
  dispose(): void;
}

/** Loads the checkpoint, warms the GPU path once, and returns the runner.
 *  Needs a device with `shader-f16`. */
export async function createImageClassifier(
  config: ImageClassifierModel = {},
  opts: LoadOptions = {},
): Promise<ImageClassifier> {
  if (!supportsF16()) {
    throw new RunntimeError(
      'UNSUPPORTED_DEVICE',
      "createImageClassifier: this device has no shader-f16, which the conv kernels need. Request it with tgpu.init({ device: { optionalFeatures: ['shader-f16'] } })",
    );
  }
  const scope = createResourceScope();
  try {
    const {
      modelPath = models.imageClassification.MOBILENETV4.DEFAULT.modelPath,
      labels = IMAGENET1K_LABELS,
      resizeMode = 'crop',
    } = config;

    // Cache key. The whole path, so same-named files stay apart. Bump the
    // version to drop bytes cached under the old one.
    const path = typeof modelPath === 'string' ? modelPath.replace(/^https?:\/\//, '') : undefined;
    const sd = await openWeights(modelPath, {
      ...opts,
      cacheId: opts.cacheId ?? (path && `mobilenetv4-v1/${path}`),
    });

    throwIfAborted(opts.signal);
    const classifier = scope.track(await createClassifier(sd, { onProgress: opts.onProgress }));
    const { inputSize, numClasses } = classifier;
    if (labels.length !== numClasses) {
      throw new RunntimeError(
        'INVALID_ARGUMENT',
        `labels: ${labels.length} names for a ${numClasses}-class model`,
      );
    }
    // (byte / 255 − mean) / std, per channel. The crop fraction matches how
    // the model was evaluated: 224 out of a 256 short side.
    const preprocessor = createImagePreprocessor(
      {
        resizeMode,
        cropFraction: MOBILENETV4_CROP_FRACTION,
        normalizeOpts: {
          alpha: IMAGENET_STD.map((s) => 1 / (255 * s)),
          beta: IMAGENET_MEAN.map((m, c) => -m / IMAGENET_STD[c]!),
        },
      },
      { width: inputSize, height: inputSize },
    );
    // One scratch array, rewritten each call; the classifier copies it to the GPU.
    const pixels = new Float32Array(3 * inputSize * inputSize);

    let disposed = false;
    // Calls run one after another: the next preprocess overwrites the scratch
    // array only after the previous run took it.
    let queue: Promise<unknown> = Promise.resolve();
    return {
      labels,
      inputSize,
      async classify(image, options = {}) {
        if (disposed) throw new RunntimeError('RESOURCE_DISPOSED', 'image classifier is disposed');
        const { topk = numClasses } = options;
        if (!Number.isInteger(topk) || topk < 1) {
          throw new RunntimeError(
            'INVALID_ARGUMENT',
            `classify: topk must be a positive integer, got ${topk}`,
          );
        }
        const run = queue.then(() => classifier.run(preprocessor.process(image, pixels)));
        queue = run.catch(() => undefined);
        return decodeTopK(await run.catch(rethrowRunError), Math.min(topk, numClasses)).map(
          ({ classId, confidence }) => ({
            label: labels[classId]!,
            classId,
            confidence,
          }),
        );
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
