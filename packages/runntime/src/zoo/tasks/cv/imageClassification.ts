/** Image classification task: an image in, the classes it shows out, best
 *  first. Runs MobileNetV4 on the initRunntime() device. */

import { d } from 'typegpu';
import {
  classScore,
  classScoreList,
  createImageToChw,
  createResourceScope,
  createSoftmaxTopK,
  defaultRoot,
  RunntimeError,
  supportsF16,
} from '../../../core/index.ts';
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
import {
  encodeWith,
  unwrapSource,
  unwrapStorageTarget,
  type EncodeOptions,
  type GpuImage,
  type GpuTensor,
} from '../gpu.ts';

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

/** The model input as a GPU buffer: normalized RGB, `[channel][row][column]`
 *  f32, `inputSize` wide and high. */
export type ClassifyInput = d.WgslArray<d.WgslArray<d.WgslArray<d.F32>>>;

/** What encodeClassify writes: the best classes, best first, each a
 *  `classId` into `labels` and its `confidence`, 0 to 1, as classify()
 *  returns them. */
export type ClassificationList = d.WgslArray<typeof classScore>;

export interface EncodeClassifyOptions extends EncodeOptions<
  GpuImage | GpuTensor<ClassifyInput>,
  GpuTensor<ClassificationList>
> {
  /** How many classes to write, best first. Default: as many as the target
   *  holds, up to every class. */
  readonly topk?: number;
}

export interface ImageClassifier {
  /** The class names, by class index. */
  readonly labels: readonly string[];
  /** Width and height of the model input in pixels. */
  readonly inputSize: number;
  /** Buffer layouts for encodeClassify, at this runner's size: the model
   *  input, and the list with room for `topk` classes. Pass one to
   *  `root.createBuffer` to make a buffer of the right type and size. */
  readonly schemas: {
    readonly input: ClassifyInput;
    readonly output: (topk: number) => ClassificationList;
  };
  /** Image in, its most likely classes out, best first. */
  classify(
    image: ImageBuffer,
    options?: ClassifyOptions,
  ): Promise<(Classification & { label: string })[]>;
  /** Records a classification from GPU memory into GPU memory, with no CPU
   *  copy. A texture source is fitted like an ImageBuffer; a buffer source
   *  is the model input as-is. The target gets the best classes, ranked on
   *  the GPU. Work submitted after it sees the result.
   *  @throws RunntimeError `INVALID_ARGUMENT` for a resource of the wrong
   *  format, usage or size, or a `topk` out of range. */
  encodeClassify(options: EncodeClassifyOptions): void;
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
    const root = defaultRoot();
    const imageToChw = scope.track(createImageToChw(root));
    const softmaxTopK = createSoftmaxTopK(root);
    const schemas = {
      input: d.arrayOf(d.arrayOf(d.arrayOf(d.f32, inputSize), inputSize), 3),
      output: classScoreList,
    };
    const inputBytes = d.sizeOf(schemas.input);
    const scoreBytes = d.sizeOf(classScore);

    let disposed = false;
    // Calls run one after another: the next preprocess overwrites the scratch
    // array only after the previous run took it.
    let queue: Promise<unknown> = Promise.resolve();
    return {
      labels,
      inputSize,
      schemas,
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
      encodeClassify(options) {
        if (disposed) throw new RunntimeError('RESOURCE_DISPOSED', 'image classifier is disposed');
        const { source, target, encoder } = options;
        const src = unwrapSource(source, inputBytes);
        const out = unwrapStorageTarget(target, scoreBytes);
        const capacity = Math.min(Math.floor(out.size / scoreBytes), numClasses);
        const { topk = capacity } = options;
        if (!Number.isInteger(topk) || topk < 1 || topk > capacity) {
          throw new RunntimeError(
            'INVALID_ARGUMENT',
            `encodeClassify: topk must be an integer from 1 to ${capacity}, what the target holds, got ${topk}`,
          );
        }
        try {
          encodeWith(encoder, (enc) =>
            classifier.encode(
              enc,
              (input) => {
                if ('tensor' in src) {
                  enc.copyBufferToBuffer(src.tensor, 0, root.unwrap(input), 0, inputBytes);
                } else {
                  imageToChw.encode(enc, src.image, input, preprocessor.gpuArgs);
                }
              },
              (logits) => softmaxTopK.encode(enc, logits, out, { n: numClasses, k: topk }),
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
