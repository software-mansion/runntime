/** Embedding task: text, images, video and audio in, one unit-length vector
 *  out, all in one space. Runs EmbeddingGemma 2 or all-MiniLM-L6-v2 on the
 *  initRunntime() device. */

import {
  createResourceScope,
  defaultRoot,
  type LazyStateDict,
  RunntimeError,
  supportsF16,
  warmUp,
} from '../../../core/index.ts';
import {
  fetchJson,
  openWeights,
  throwIfAborted,
  type LoadOptions,
  type ModelPath,
} from '../../load.ts';
import { asLoadError, rethrowRunError } from '../../errors.ts';
import { audioTokens } from '../../embeddinggemma2/audioEncoder.ts';
import { EMBEDDINGGEMMA2_PROMPTS, IMAGE_TOKEN_BUDGETS } from '../../embeddinggemma2/config.ts';
import { createEmbeddingModel as createEmbeddingGemma2Model } from '../../embeddinggemma2/embeddingModel.ts';
import { createEmbeddingModel as createMinilmModel } from '../../minilm/embeddingModel.ts';
import { models } from '../../models.ts';
import { checkImageBuffer, type ImageBuffer } from '../cv/image.ts';

/** What a text is for. Each task adds the instruction the model was
 *  trained with. */
export type EmbeddingTask = keyof typeof EMBEDDINGGEMMA2_PROMPTS;

/** Most tokens one image takes: 70, 140, 280, 560 or 1120. */
export type ImageTokenBudget = (typeof IMAGE_TOKEN_BUDGETS)[number];

/** One piece of an input: text, an image, a video as its frames, or an
 *  audio clip as mono samples at 16 kHz. */
export type EmbedPart =
  | string
  | { readonly image: ImageBuffer }
  | { readonly video: readonly ImageBuffer[] }
  | { readonly audio: Float32Array };

/** What embed() takes: one part, or parts in reading order that together
 *  make one vector. */
export type EmbedInput = EmbedPart | readonly EmbedPart[];

/** What to load. Every field is optional; a missing one comes from the
 *  model's DEFAULT, EmbeddingGemma 2 without `modelPath`. */
export interface EmbedderModel {
  /** The safetensors weights of EmbeddingGemma 2 or all-MiniLM-L6-v2. The
   *  model is read from the file. */
  readonly modelPath?: ModelPath;
  /** The vision tower's weights, for images and video. EmbeddingGemma 2
   *  only. `null` skips it: a smaller download. */
  readonly visionPath?: ModelPath | null;
  /** The audio tower's weights, for audio. EmbeddingGemma 2 only. `null`
   *  skips it: a smaller download. */
  readonly audioPath?: ModelPath | null;
  /** The Hugging Face tokenizer.json. */
  readonly tokenizerPath?: string;
  /** Precision on the GPU. `f16` runs the whole model in half precision,
   *  weights and activations, for half the memory traffic at a small cost
   *  in precision. Needs a device with `shader-f16`. Default: `f16` where
   *  the device has it, `f32` elsewhere. Either loads any weight file. */
  readonly dtype?: 'f32' | 'f16';
  /** Longest input in tokens. EmbeddingGemma 2: up to 2048, default 2048.
   *  all-MiniLM-L6-v2: up to 512, default 256. Longer text is cut; a longer
   *  input with images, video or audio is an error. */
  readonly maxTokens?: number;
  /** Most tokens one image takes: 70, 140, 280, 560 or 1120. More tokens
   *  keep more detail and take longer. Default 280. */
  readonly imageTokens?: ImageTokenBudget;
}

/** Per-call options of embed() and embedBatch(). */
export interface EmbedOptions {
  /** What the text is for. A search embeds its queries with `query` and
   *  what it searches with `document`. all-MiniLM-L6-v2 ignores it.
   *  Default: none. */
  readonly task?: EmbeddingTask;
  /** Length of the returned vectors: 768, 512, 256 or 128 for
   *  EmbeddingGemma 2, 384 for all-MiniLM-L6-v2. A shorter vector keeps the
   *  first values, made unit length again. Compare only vectors of the same
   *  length. Default `dim`. */
  readonly dims?: number;
}

/** The runner createEmbedder() returns. */
export interface Embedder {
  /** Length of a vector without `dims`. */
  readonly dim: number;
  /** One input → unit-length vector. */
  embed(input: EmbedInput, options?: EmbedOptions): Promise<Float32Array>;
  /** Many inputs, one vector each. Much faster than embed() in a loop for
   *  text. */
  embedBatch(inputs: readonly EmbedInput[], options?: EmbedOptions): Promise<Float32Array[]>;
  /** Frees the weights and every GPU buffer the runner holds. */
  dispose(): void;
}

/** What a model's createEmbeddingModel() gets: the opened files and the
 *  options. */
export interface EmbeddingModelLoad {
  readonly text: LazyStateDict;
  readonly vision?: LazyStateDict;
  readonly audio?: LazyStateDict;
  /** The tokenizer JSON. */
  readonly tokenizer: unknown;
  readonly dtype: 'f32' | 'f16';
  readonly maxTokens?: number;
  readonly imageTokens?: ImageTokenBudget;
  readonly onProgress?: LoadOptions['onProgress'];
}

/** What a model's createEmbeddingModel() returns. */
export interface EmbeddingModel {
  /** Full length of a vector. */
  readonly dim: number;
  /** Lengths `dims` may take. */
  readonly dims: readonly number[];
  /** Whether it reads images, video and audio. */
  readonly media: boolean;
  /** Runs every GPU path once. */
  warm(): Promise<Float32Array>;
  /** The pooled rows of `inputs`, `dim` values each, not yet unit length.
   *  Errors start with `where`. */
  pool(where: string, inputs: readonly EmbedInput[], task?: EmbeddingTask): Promise<Float32Array>;
  dispose(): void;
}

/** Cosine similarity of two unit-length vectors: 1 for the same meaning,
 *  near 0 for unrelated inputs. */
export function similarity(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i]! * b[i]!;
  return dot;
}

/** Scales `v` to unit length in place and returns it. */
const normalize = (v: Float32Array): Float32Array => {
  let sumSq = 0;
  for (const x of v) sumSq += x * x;
  const inv = 1 / Math.sqrt(sumSq);
  for (let i = 0; i < v.length; i++) v[i]! *= inv;
  return v;
};

const isImagePart = (part: unknown): part is { image: ImageBuffer } =>
  typeof part === 'object' && part !== null && 'image' in part;
const isVideoPart = (part: unknown): part is { video: readonly ImageBuffer[] } =>
  typeof part === 'object' && part !== null && 'video' in part && Array.isArray(part.video);
const isAudioPart = (part: unknown): part is { audio: Float32Array } =>
  typeof part === 'object' &&
  part !== null &&
  'audio' in part &&
  part.audio instanceof Float32Array;

/** One cache key per file, using the whole path so same-named files of two
 *  precisions stay apart. Bump the version to invalidate. */
const cacheId = (path: ModelPath) =>
  typeof path === 'string' ? `embedding-v1/${path.replace(/^https?:\/\//, '')}` : undefined;

/** Loads the model in `modelPath` with its tokenizer and towers, warms the
 *  GPU, and returns the runner. With no config, loads EmbeddingGemma 2 from
 *  the Hugging Face Hub. */
export async function createEmbedder(
  config: EmbedderModel = {},
  opts: LoadOptions = {},
): Promise<Embedder> {
  const scope = createResourceScope();
  try {
    const device = defaultRoot().device;
    const modelPath = config.modelPath ?? models.embedding.EMBEDDINGGEMMA_2.DEFAULT.modelPath;
    const text = await openWeights(modelPath, {
      ...opts,
      cacheId: opts.cacheId ?? cacheId(modelPath),
    });
    // A file with BERT's word embeddings is all-MiniLM-L6-v2, any other is
    // EmbeddingGemma 2.
    const minilm = text.tensors.has('embeddings.word_embeddings.weight');
    const defaults = minilm
      ? {
          ...models.embedding.ALL_MINILM_L6_V2.DEFAULT,
          visionPath: null,
          audioPath: null,
        }
      : models.embedding.EMBEDDINGGEMMA_2.DEFAULT;
    const {
      visionPath = defaults.visionPath,
      audioPath = defaults.audioPath,
      tokenizerPath = defaults.tokenizerPath,
      dtype = supportsF16() ? 'f16' : 'f32',
      maxTokens,
      imageTokens,
    } = config;
    if (minilm && (visionPath !== null || audioPath !== null)) {
      throw new RunntimeError(
        'INVALID_ARGUMENT',
        'createEmbedder: all-MiniLM-L6-v2 has no vision or audio tower, set no visionPath or audioPath',
      );
    }
    const open = (path: ModelPath | null) =>
      path === null ? undefined : openWeights(path, { ...opts, cacheId: cacheId(path) });
    const vision = await open(visionPath);
    const audio = await open(audioPath);
    const tokenizer = await fetchJson(tokenizerPath, { ...opts, cacheId: cacheId(tokenizerPath) });

    throwIfAborted(opts.signal);
    const make = minilm ? createMinilmModel : createEmbeddingGemma2Model;
    const model = scope.track(
      await make({
        text,
        vision,
        audio,
        tokenizer,
        dtype,
        maxTokens,
        imageTokens,
        onProgress: opts.onProgress,
      }),
    );

    // Compiles the kernels, and turns a GPU error into one load error.
    throwIfAborted(opts.signal);
    await warmUp(device, () => model.warm());

    let disposed = false;
    const guard = () => {
      if (disposed) throw new RunntimeError('RESOURCE_DISPOSED', 'embedder is disposed');
    };

    /** Throws unless `input` is an EmbedInput this runner can embed. */
    const check = (where: string, input: unknown): void => {
      const parts: readonly unknown[] = Array.isArray(input) ? input : [input];
      if (parts.length === 0) {
        throw new RunntimeError('INVALID_ARGUMENT', `${where}: an input needs at least one part`);
      }
      for (const part of parts) {
        if (typeof part === 'string') continue;
        if (!model.media) {
          throw new RunntimeError(
            'INVALID_ARGUMENT',
            `${where}: this model reads only text, a part must be a string`,
          );
        }
        if (isAudioPart(part)) {
          audioTokens(part.audio.length);
          continue;
        }
        if (!isImagePart(part) && !(isVideoPart(part) && part.video.length > 0)) {
          throw new RunntimeError(
            'INVALID_ARGUMENT',
            `${where}: a part must be a string, { image }, { video } with at least one frame or { audio } as a Float32Array`,
          );
        }
        if (isImagePart(part)) checkImageBuffer(part.image);
        else for (const frame of part.video) checkImageBuffer(frame);
      }
    };

    const checkTask = (where: string, task: unknown): void => {
      if (task !== undefined && !Object.hasOwn(EMBEDDINGGEMMA2_PROMPTS, String(task))) {
        throw new RunntimeError(
          'INVALID_ARGUMENT',
          `${where}: task must be one of ${Object.keys(EMBEDDINGGEMMA2_PROMPTS).join(', ')}, got ${task}`,
        );
      }
    };

    const checkDims = (where: string, dims: unknown): void => {
      if (dims !== undefined && (typeof dims !== 'number' || !model.dims.includes(dims))) {
        throw new RunntimeError(
          'INVALID_ARGUMENT',
          `${where}: dims must be one of ${model.dims.join(', ')}, got ${dims}`,
        );
      }
    };

    /** Each input's vector: the first `dims` values of its pooled row, unit
     *  length. */
    const embedInputs = async (
      where: string,
      inputs: readonly EmbedInput[],
      { task, dims = model.dim }: EmbedOptions,
    ): Promise<Float32Array[]> => {
      if (inputs.length === 0) return [];
      const pooled = await model.pool(where, inputs, task);
      return inputs.map((_, b) => normalize(pooled.slice(b * model.dim, b * model.dim + dims)));
    };

    // Calls run one after another, on buffers the next call reuses.
    let queue: Promise<unknown> = Promise.resolve();
    const serial = <T>(work: () => Promise<T>): Promise<T> => {
      const run = queue.then(work);
      queue = run.catch(() => undefined);
      return run.catch(rethrowRunError);
    };

    return {
      dim: model.dim,
      async embed(input, embedOpts = {}) {
        guard();
        check('embed', input);
        checkTask('embed', embedOpts.task);
        checkDims('embed', embedOpts.dims);
        return serial(async () => (await embedInputs('embed', [input], embedOpts))[0]!);
      },
      async embedBatch(inputs, embedOpts = {}) {
        guard();
        if (!Array.isArray(inputs)) {
          throw new RunntimeError('INVALID_ARGUMENT', 'embedBatch: inputs must be an array');
        }
        for (const input of inputs) check('embedBatch', input);
        checkTask('embedBatch', embedOpts.task);
        checkDims('embedBatch', embedOpts.dims);
        return serial(() => embedInputs('embedBatch', inputs, embedOpts));
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
