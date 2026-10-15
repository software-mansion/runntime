/** Embedding task: an input in, one unit-length vector out, in a space
 *  where any two vectors compare. Runs EmbeddingGemma 2 on the
 *  initRunntime() device. */

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
} from '../../embeddinggemma2/config.ts';
import { createEmbeddingGemma2EmbedCore } from '../../embeddinggemma2/embedCore.ts';
import { EmbeddingGemma2Model } from '../../embeddinggemma2/model.ts';
import { createEmbeddingGemma2Tokenizer } from '../../embeddinggemma2/tokenizer.ts';
import { models } from '../../models.ts';
import { normalize } from '../nlp/textEmbedding.ts';

/** What a text is for. Each task puts its own instruction before the text,
 *  which the model was trained with. */
export type EmbeddingTask = keyof typeof EMBEDDINGGEMMA2_PROMPTS;

/** What embed() takes: a text. */
export type EmbedInput = string;

/** What to load. Every field is optional; a missing one comes from
 *  models.multimodalEmbedding.EMBEDDINGGEMMA_2.DEFAULT. */
export interface MultimodalEmbedderModel {
  /** The safetensors checkpoint. */
  readonly modelPath?: ModelPath;
  /** The Hugging Face tokenizer.json. */
  readonly tokenizerPath?: string;
  /** Precision on the GPU. `f16` runs the whole model in half precision,
   *  weights and activations, for half the memory traffic; results move by
   *  about 1e-3. Needs a device with `shader-f16`. Default: `f16` where
   *  the device has it, `f32` elsewhere. Either loads any weight file. */
  readonly dtype?: 'f32' | 'f16';
  /** Longest input in tokens, special tokens included, at most 2048.
   *  Longer text is cut. Default 2048. */
  readonly maxTokens?: number;
  /** Length of the returned vectors: 768, 512, 256 or 128. A shorter vector
   *  keeps the first values and is made unit length again, for less storage
   *  at a small cost in quality. Default 768. */
  readonly dims?: number;
}

/** Per-call options of embed() and embedBatch(). */
export interface EmbedOptions {
  /** What the text is for. A search embeds its queries with `query` and
   *  what it searches with `document`. Default: no instruction, the text
   *  as given. */
  readonly task?: EmbeddingTask;
}

/** The runner createMultimodalEmbedder() returns. */
export interface MultimodalEmbedder {
  /** Length of every returned vector. */
  readonly dim: number;
  /** One input → unit-length vector. */
  embed(input: EmbedInput, options?: EmbedOptions): Promise<Float32Array>;
  /** Many inputs in one GPU pass. Much faster than embed() in a loop. */
  embedBatch(inputs: readonly EmbedInput[], options?: EmbedOptions): Promise<Float32Array[]>;
  /** Frees the weights and every GPU buffer the runner holds. */
  dispose(): void;
}

/** Loads the checkpoint and the tokenizer, warms every GPU path once, and
 *  returns the runner. With no config, loads EmbeddingGemma 2 from the
 *  Hugging Face Hub. */
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
      tokenizerPath = defaults.tokenizerPath,
      dtype = supportsF16() ? 'f16' : 'f32',
      maxTokens = EMBEDDINGGEMMA2_MAX_TOKENS,
      dims = EMBEDDING_DIM,
    } = config;
    const allowedDims: readonly number[] = EMBEDDING_DIMS;
    if (!allowedDims.includes(dims)) {
      throw new RunntimeError(
        'INVALID_ARGUMENT',
        `createMultimodalEmbedder: dims must be one of ${allowedDims.join(', ')}, got ${dims}`,
      );
    }
    if (!Number.isInteger(maxTokens) || maxTokens < 2 || maxTokens > EMBEDDINGGEMMA2_MAX_TOKENS) {
      throw new RunntimeError(
        'INVALID_ARGUMENT',
        `createMultimodalEmbedder: maxTokens must be an integer from 2 to ${EMBEDDINGGEMMA2_MAX_TOKENS}, got ${maxTokens}`,
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
    const tokenizer = createEmbeddingGemma2Tokenizer(
      await fetchJson(tokenizerPath, { ...opts, cacheId: cacheId(tokenizerPath) }),
      maxTokens,
    );

    throwIfAborted(opts.signal);
    const model = scope.track(new EmbeddingGemma2Model());
    if (dtype === 'f16') model.half();
    await model.loadStateDict(sd, { onProgress: opts.onProgress });
    const core = scope.track(createEmbeddingGemma2EmbedCore(model));

    let disposed = false;
    const guard = () => {
      if (disposed) throw new RunntimeError('RESOURCE_DISPOSED', 'multimodal embedder is disposed');
    };

    /** The text with its task's instruction in front. */
    const withTask = (where: string, text: string, task: EmbeddingTask | undefined): string => {
      if (task === undefined) return text;
      if (!Object.hasOwn(EMBEDDINGGEMMA2_PROMPTS, task)) {
        throw new RunntimeError(
          'INVALID_ARGUMENT',
          `${where}: task must be one of ${Object.keys(EMBEDDINGGEMMA2_PROMPTS).join(', ')}, got ${task}`,
        );
      }
      return EMBEDDINGGEMMA2_PROMPTS[task] + text;
    };

    /** Pooled output to the returned vector: the first `dims` values, unit length. */
    const finish = (pooled: Float32Array, offset: number): Float32Array =>
      normalize(pooled.slice(offset, offset + dims));

    const embedTokens = async (ids: readonly number[]): Promise<Float32Array> =>
      finish(await core.embedPooled(ids), 0);

    const embedTokensBatch = async (
      idsList: readonly (readonly number[])[],
    ): Promise<Float32Array[]> => {
      const [first, ...rest] = idsList;
      if (first === undefined) return [];
      // A batch of one is a single text: take the replay path.
      if (rest.length === 0) return [await embedTokens(first)];
      const pooled = await core.embedPooledBatch(idsList);
      return idsList.map((_, b) => finish(pooled, b * EMBEDDING_DIM));
    };

    // Run every GPU path once. First use compiles kernels and records the
    // replays, and warmUp turns a GPU error into one load error. Later
    // calls skip both.
    throwIfAborted(opts.signal);
    await warmUp(device, async () => {
      // The empty text is just <bos> and <eos>.
      const probe = tokenizer.encode('');
      await embedTokensBatch([probe, probe]);
      await core.warmBuckets(probe);
      return embedTokens(probe);
    });

    return {
      dim: dims,
      async embed(input, embedOpts = {}) {
        guard();
        if (typeof input !== 'string') {
          throw new RunntimeError('INVALID_ARGUMENT', 'embed: input must be a string');
        }
        const text = withTask('embed', input, embedOpts.task);
        return embedTokens(tokenizer.encode(text)).catch(rethrowRunError);
      },
      async embedBatch(inputs, embedOpts = {}) {
        guard();
        if (!Array.isArray(inputs) || !inputs.every((t) => typeof t === 'string')) {
          throw new RunntimeError(
            'INVALID_ARGUMENT',
            'embedBatch: inputs must be an array of strings',
          );
        }
        const texts = inputs.map((t) => withTask('embedBatch', t, embedOpts.task));
        return embedTokensBatch(texts.map((t) => tokenizer.encode(t))).catch(rethrowRunError);
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
