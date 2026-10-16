/** Embedding task: text, images, video and audio in, one unit-length vector
 *  out, all in one space, so any vector compares with any other. Runs
 *  EmbeddingGemma 2 on the initRunntime() device. */

import {
  createResourceScope,
  defaultRoot,
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
import {
  EMBEDDING_DIM,
  EMBEDDING_DIMS,
  EMBEDDINGGEMMA2_MAX_TOKENS,
  EMBEDDINGGEMMA2_PROMPTS,
  EMBEDDINGGEMMA2_VISION,
  IMAGE_TOKEN_BUDGETS,
  IMAGE_TOKENS,
  VIDEO_FRAME_TOKENS,
} from '../../embeddinggemma2/config.ts';
import {
  createEmbeddingGemma2EmbedCore,
  type EmbeddingGemma2Input,
} from '../../embeddinggemma2/embedCore.ts';
import { EmbeddingGemma2Model } from '../../embeddinggemma2/model.ts';
import { EmbeddingGemma2AudioModel } from '../../embeddinggemma2/audioModel.ts';
import { audioTokens, encodeAudio } from '../../embeddinggemma2/audioEncoder.ts';
import { createEmbeddingGemma2Tokenizer } from '../../embeddinggemma2/tokenizer.ts';
import { encodeImage, patchGrid } from '../../embeddinggemma2/visionEncoder.ts';
import { EmbeddingGemma2VisionModel } from '../../embeddinggemma2/visionModel.ts';
import { models } from '../../models.ts';
import { checkImageBuffer, imageBuffer, type ImageBuffer } from '../cv/image.ts';
import { normalize } from '../nlp/textEmbedding.ts';

/** What a text is for. Each task puts its own instruction before the text,
 *  which the model was trained with. */
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

/** What to load. Every field is optional; a missing one comes from
 *  models.multimodalEmbedding.EMBEDDINGGEMMA_2.DEFAULT. */
export interface MultimodalEmbedderModel {
  /** The text model's safetensors checkpoint. */
  readonly modelPath?: ModelPath;
  /** The vision tower's safetensors checkpoint, needed for images and
   *  video. `null` loads no vision tower: a smaller download. */
  readonly visionPath?: ModelPath | null;
  /** The audio tower's safetensors checkpoint, needed for audio. `null`
   *  loads no audio tower: a smaller download. */
  readonly audioPath?: ModelPath | null;
  /** The Hugging Face tokenizer.json. */
  readonly tokenizerPath?: string;
  /** Precision on the GPU. `f16` runs the whole model in half precision,
   *  weights and activations, for half the memory traffic at a small cost
   *  in precision. Needs a device with `shader-f16`. Default: `f16` where
   *  the device has it, `f32` elsewhere. Either loads any weight file. */
  readonly dtype?: 'f32' | 'f16';
  /** Longest input in tokens, special tokens and media tokens included,
   *  at most 2048. Longer text is cut; a longer input with images, video or
   *  audio is an error. Default 2048. */
  readonly maxTokens?: number;
  /** Most tokens one image takes: 70, 140, 280, 560 or 1120. More tokens
   *  keep more detail and take longer. Default 280. */
  readonly imageTokens?: ImageTokenBudget;
}

/** Per-call options of embed() and embedBatch(). */
export interface EmbedOptions {
  /** What the text is for. A search embeds its queries with `query` and
   *  what it searches with `document`. The instruction goes before the
   *  whole input. Default: no instruction. */
  readonly task?: EmbeddingTask;
  /** Length of the returned vectors: 768, 512, 256 or 128. A shorter vector
   *  keeps the first values and is made unit length again, for less storage
   *  at a small cost in quality. Compare only vectors of the same length.
   *  Default 768. */
  readonly dims?: number;
}

/** The runner createMultimodalEmbedder() returns. */
export interface MultimodalEmbedder {
  /** Length of a returned vector when the call sets no `dims`. */
  readonly dim: number;
  /** One input → unit-length vector. */
  embed(input: EmbedInput, options?: EmbedOptions): Promise<Float32Array>;
  /** Many inputs, the text model in one GPU pass. Much faster than embed()
   *  in a loop for text. */
  embedBatch(inputs: readonly EmbedInput[], options?: EmbedOptions): Promise<Float32Array[]>;
  /** Frees the weights and every GPU buffer the runner holds. */
  dispose(): void;
}

const isImagePart = (part: unknown): part is { image: ImageBuffer } =>
  typeof part === 'object' && part !== null && 'image' in part;
const isVideoPart = (part: unknown): part is { video: readonly ImageBuffer[] } =>
  typeof part === 'object' && part !== null && 'video' in part && Array.isArray(part.video);
const isAudioPart = (part: unknown): part is { audio: Float32Array } =>
  typeof part === 'object' &&
  part !== null &&
  'audio' in part &&
  part.audio instanceof Float32Array;

/** Loads the text model, the vision and audio towers and the tokenizer,
 *  warms every GPU path once, and returns the runner. With no config, loads
 *  EmbeddingGemma 2 from the Hugging Face Hub. */
export async function createMultimodalEmbedder(
  config: MultimodalEmbedderModel = {},
  opts: LoadOptions = {},
): Promise<MultimodalEmbedder> {
  const scope = createResourceScope();
  try {
    const device = defaultRoot().device;
    const defaults = models.multimodalEmbedding.EMBEDDINGGEMMA_2.DEFAULT;
    const {
      modelPath = defaults.modelPath,
      visionPath = defaults.visionPath,
      audioPath = defaults.audioPath,
      tokenizerPath = defaults.tokenizerPath,
      dtype = supportsF16() ? 'f16' : 'f32',
      maxTokens = EMBEDDINGGEMMA2_MAX_TOKENS,
      imageTokens = IMAGE_TOKENS,
    } = config;
    if (!Number.isInteger(maxTokens) || maxTokens < 2 || maxTokens > EMBEDDINGGEMMA2_MAX_TOKENS) {
      throw new RunntimeError(
        'INVALID_ARGUMENT',
        `createMultimodalEmbedder: maxTokens must be an integer from 2 to ${EMBEDDINGGEMMA2_MAX_TOKENS}, got ${maxTokens}`,
      );
    }
    const budgets: readonly number[] = IMAGE_TOKEN_BUDGETS;
    if (!budgets.includes(imageTokens)) {
      throw new RunntimeError(
        'INVALID_ARGUMENT',
        `createMultimodalEmbedder: imageTokens must be one of ${budgets.join(', ')}, got ${imageTokens}`,
      );
    }

    // One cache key per file, using the whole path so same-named files of
    // two precisions stay apart. Bump the version to invalidate.
    const cacheId = (path: ModelPath) =>
      typeof path === 'string'
        ? `embeddinggemma-2-v1/${path.replace(/^https?:\/\//, '')}`
        : undefined;
    const sd = await openWeights(modelPath, {
      ...opts,
      cacheId: opts.cacheId ?? cacheId(modelPath),
    });
    const visionSd =
      visionPath === null
        ? undefined
        : await openWeights(visionPath, { ...opts, cacheId: cacheId(visionPath) });
    const audioSd =
      audioPath === null
        ? undefined
        : await openWeights(audioPath, { ...opts, cacheId: cacheId(audioPath) });
    const tokenizer = createEmbeddingGemma2Tokenizer(
      await fetchJson(tokenizerPath, { ...opts, cacheId: cacheId(tokenizerPath) }),
    );

    throwIfAborted(opts.signal);
    const model = scope.track(new EmbeddingGemma2Model());
    const vision = visionSd && scope.track(new EmbeddingGemma2VisionModel());
    const audio = audioSd && scope.track(new EmbeddingGemma2AudioModel());
    if (dtype === 'f16') {
      model.half();
      vision?.half();
      audio?.half();
    }
    // The files load at once; progress counts their bytes together.
    const progress = new Map<string, [number, number]>();
    const track = (file: string) => (name: string, done: number, total: number) => {
      progress.set(file, [done, total]);
      let d = 0;
      let t = 0;
      for (const [fd, ft] of progress.values()) {
        d += fd;
        t += ft;
      }
      opts.onProgress?.(name, d, t);
    };
    const loads = [model.loadStateDict(sd, { onProgress: track('text') })];
    if (vision && visionSd) {
      loads.push(vision.loadStateDict(visionSd, { onProgress: track('vision') }));
    }
    if (audio && audioSd) {
      loads.push(audio.loadStateDict(audioSd, { onProgress: track('audio') }));
    }
    await Promise.all(loads);
    const core = scope.track(createEmbeddingGemma2EmbedCore(model));
    const { bosId, eosId, padId, boiId, eoiId, boaId, eoaId } = tokenizer;
    const { patch, pool } = EMBEDDINGGEMMA2_VISION;

    let disposed = false;
    const guard = () => {
      if (disposed) throw new RunntimeError('RESOURCE_DISPOSED', 'multimodal embedder is disposed');
    };

    /** Throws unless `input` is an EmbedInput this runner can embed. */
    const check = (where: string, input: unknown): void => {
      const parts: readonly unknown[] = Array.isArray(input) ? input : [input];
      if (parts.length === 0) {
        throw new RunntimeError('INVALID_ARGUMENT', `${where}: an input needs at least one part`);
      }
      for (const part of parts) {
        if (typeof part === 'string') continue;
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

    /** An input as token ids and media rows: <bos>, the instruction and the
     *  parts in order, <eos>. Text next to other text joins it, each image
     *  or video frame is <|image> rows <image|>, and each clip <|audio> rows
     *  <audio|>. The length is checked before any tower runs. */
    const toInput = async (
      where: string,
      input: EmbedInput,
      task: EmbeddingTask | undefined,
    ): Promise<EmbeddingGemma2Input> => {
      // Array.isArray does not narrow a readonly array.
      const parts = (Array.isArray(input) ? input : [input]) as readonly EmbedPart[];
      const ids = [bosId];
      // Each media part's first row and how to compute its rows.
      const pending: { at: number; encode: () => Promise<Float32Array> }[] = [];
      let text = task === undefined ? '' : EMBEDDINGGEMMA2_PROMPTS[task];
      const flush = () => {
        ids.push(...tokenizer.encode(text));
        text = '';
      };
      const addMedia = (
        begin: number,
        tokens: number,
        end: number,
        encode: () => Promise<Float32Array>,
      ) => {
        flush();
        ids.push(begin);
        pending.push({ at: ids.length, encode });
        for (let i = 0; i < tokens; i++) ids.push(padId);
        ids.push(end);
      };
      const addImage = (image: ImageBuffer, budget: number) => {
        const [cols, rows] = patchGrid(image.width, image.height, budget, patch, pool);
        addMedia(boiId, (cols * rows) / (pool * pool), eoiId, () => {
          if (!vision) {
            throw new RunntimeError(
              'INVALID_ARGUMENT',
              `${where}: images and video need the vision tower, which this embedder did not load`,
            );
          }
          return encodeImage(vision, image, budget);
        });
      };
      for (const part of parts) {
        if (typeof part === 'string') text += part;
        else if ('image' in part) addImage(part.image, imageTokens);
        else if ('video' in part) {
          for (const frame of part.video) addImage(frame, VIDEO_FRAME_TOKENS);
        } else {
          const samples = part.audio;
          addMedia(boaId, audioTokens(samples.length), eoaId, () => {
            if (!audio) {
              throw new RunntimeError(
                'INVALID_ARGUMENT',
                `${where}: audio needs the audio tower, which this embedder did not load`,
              );
            }
            return encodeAudio(audio, samples);
          });
        }
      }
      flush();
      ids.push(eosId);
      if (ids.length > maxTokens) {
        if (pending.length === 0) {
          return { ids: [...ids.slice(0, maxTokens - 1), eosId], media: [] };
        }
        throw new RunntimeError(
          'INVALID_ARGUMENT',
          `${where}: the input takes ${ids.length} tokens, more than maxTokens ${maxTokens}`,
        );
      }
      const media: { at: number; rows: Float32Array }[] = [];
      for (const { at, encode } of pending) media.push({ at, rows: await encode() });
      return { ids, media };
    };

    const checkTask = (where: string, task: unknown): void => {
      if (task !== undefined && !Object.hasOwn(EMBEDDINGGEMMA2_PROMPTS, String(task))) {
        throw new RunntimeError(
          'INVALID_ARGUMENT',
          `${where}: task must be one of ${Object.keys(EMBEDDINGGEMMA2_PROMPTS).join(', ')}, got ${task}`,
        );
      }
    };

    const allowedDims: readonly number[] = EMBEDDING_DIMS;
    const checkDims = (where: string, dims: unknown): void => {
      if (dims !== undefined && (typeof dims !== 'number' || !allowedDims.includes(dims))) {
        throw new RunntimeError(
          'INVALID_ARGUMENT',
          `${where}: dims must be one of ${allowedDims.join(', ')}, got ${dims}`,
        );
      }
    };

    /** Pooled output to the returned vector: the first `dims` values, unit length. */
    const finish = (pooled: Float32Array, offset: number, dims: number): Float32Array =>
      normalize(pooled.slice(offset, offset + dims));

    const embedInput = async (
      input: EmbeddingGemma2Input,
      dims = EMBEDDING_DIM,
    ): Promise<Float32Array> => finish(await core.embedPooled(input), 0, dims);

    const embedInputs = async (
      inputs: EmbeddingGemma2Input[],
      dims = EMBEDDING_DIM,
    ): Promise<Float32Array[]> => {
      const [first, ...rest] = inputs;
      if (first === undefined) return [];
      // A batch of one is a single input: take the replay path.
      if (rest.length === 0) return [await embedInput(first, dims)];
      const pooled = await core.embedPooledBatch(inputs);
      return inputs.map((_, b) => finish(pooled, b * EMBEDDING_DIM, dims));
    };

    // Run every GPU path once. First use compiles kernels and records the
    // replays, and warmUp turns a GPU error into one load error. Later
    // calls skip both.
    throwIfAborted(opts.signal);
    await warmUp(device, async () => {
      // The empty text is just <bos> and <eos>.
      const probe = await toInput('createMultimodalEmbedder', '', undefined);
      await embedInputs([probe, probe]);
      await core.warmBuckets(probe);
      // A 1x1 image grows to the budget. The image and the video frame
      // budgets each run once, as their sizes may take different attention
      // paths.
      if (vision) {
        const dot = imageBuffer(new Uint8Array(4), 1, 1);
        await encodeImage(vision, dot, imageTokens);
        if (imageTokens !== VIDEO_FRAME_TOKENS) await encodeImage(vision, dot, VIDEO_FRAME_TOKENS);
      }
      // A tenth of a second of silence runs every audio kernel.
      if (audio) await encodeAudio(audio, new Float32Array(1600));
      return embedInput(probe);
    });

    // Calls run one after another: each runs the towers, then the text
    // model, on buffers the next call reuses.
    let queue: Promise<unknown> = Promise.resolve();
    const serial = <T>(work: () => Promise<T>): Promise<T> => {
      const run = queue.then(work);
      queue = run.catch(() => undefined);
      return run.catch(rethrowRunError);
    };

    return {
      dim: EMBEDDING_DIM,
      async embed(input, embedOpts = {}) {
        guard();
        check('embed', input);
        checkTask('embed', embedOpts.task);
        checkDims('embed', embedOpts.dims);
        return serial(async () =>
          embedInput(await toInput('embed', input, embedOpts.task), embedOpts.dims),
        );
      },
      async embedBatch(inputs, embedOpts = {}) {
        guard();
        if (!Array.isArray(inputs)) {
          throw new RunntimeError('INVALID_ARGUMENT', 'embedBatch: inputs must be an array');
        }
        for (const input of inputs) check('embedBatch', input);
        checkTask('embedBatch', embedOpts.task);
        checkDims('embedBatch', embedOpts.dims);
        return serial(async () => {
          const prepared: EmbeddingGemma2Input[] = [];
          for (const input of inputs) {
            prepared.push(await toInput('embedBatch', input, embedOpts.task));
          }
          return embedInputs(prepared, embedOpts.dims);
        });
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
