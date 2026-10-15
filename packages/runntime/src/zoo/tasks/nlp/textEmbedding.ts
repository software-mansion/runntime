/** Text embedding task: text in, one unit-length vector out. Runs
 *  all-MiniLM-L6-v2 or EmbeddingGemma 2 on the initRunntime() device. */

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
  EMBEDDING_DIM as GEMMA_DIM,
  EMBEDDING_DIMS as GEMMA_DIMS,
  EMBEDDINGGEMMA2_MAX_TOKENS,
  EMBEDDINGGEMMA2_PROMPTS,
} from '../../embeddinggemma2/config.ts';
import { createEmbeddingGemma2EmbedCore } from '../../embeddinggemma2/embedCore.ts';
import { EmbeddingGemma2Model } from '../../embeddinggemma2/model.ts';
import { createEmbeddingGemma2Tokenizer } from '../../embeddinggemma2/tokenizer.ts';
import { EMBEDDING_DIM as MINILM_DIM, MINILM_L6 } from '../../minilm/config.ts';
import { createMinilmEmbedCore } from '../../minilm/embedCore.ts';
import { MinilmModel } from '../../minilm/model.ts';
import { createMinilmTokenizer, minilmTokenizerAsset } from '../../minilm/tokenizer.ts';
import { models } from '../../models.ts';

/** The two text embedding families. Each has its own weights and tokenizer. */
export type TextEmbeddingArch = 'minilm' | 'embeddinggemma-2';

/** What an EmbeddingGemma 2 text is for. Each task puts its own instruction
 *  before the text, which the model was trained with. */
export type EmbeddingTask = keyof typeof EMBEDDINGGEMMA2_PROMPTS;

/** What to load. Every field is optional; a missing one comes from the
 *  family's default entry in models.textEmbedding. */
export interface TextEmbedderModel {
  /** The family the weights belong to. Default `minilm`. */
  readonly arch?: TextEmbeddingArch;
  /** The safetensors checkpoint. */
  readonly modelPath?: ModelPath;
  /** The tokenizer JSON: a Hugging Face tokenizer.json, or for MiniLM also
   *  our compact asset. */
  readonly tokenizerPath?: string;
  /** Precision on the GPU. `f16` runs the whole model in half precision,
   *  weights and activations, for half the memory traffic; results move by
   *  about 1e-3. Needs a device with `shader-f16`. Default: `f16` where
   *  the device has it, `f32` elsewhere. Either loads any weight file. */
  readonly dtype?: 'f32' | 'f16';
  /** Longest input in tokens, special tokens included. Longer text is cut.
   *  Default: 256 for MiniLM, its model card's limit, and 2048 for
   *  EmbeddingGemma 2, the most this runner takes. */
  readonly maxTokens?: number;
  /** Length of the returned vectors. EmbeddingGemma 2 takes 768, 512, 256 or
   *  128: a shorter vector keeps the first values and is made unit length
   *  again, for less storage at a small cost in quality. Default: the
   *  model's full size, 384 for MiniLM and 768 for EmbeddingGemma 2. */
  readonly dims?: number;
}

export interface EmbedOptions {
  /** EmbeddingGemma 2 only: what the text is for. A search embeds its
   *  queries with `query` and the texts it searches with `document`.
   *  Default: no instruction, the text as given. */
  readonly task?: EmbeddingTask;
}

export interface TextEmbedder {
  /** Length of every returned vector. */
  readonly dim: number;
  /** One text → unit-length vector. */
  embed(input: string, opts?: EmbedOptions): Promise<Float32Array>;
  /** Many texts in one GPU pass. Much faster than embed() in a loop. */
  embedBatch(inputs: readonly string[], opts?: EmbedOptions): Promise<Float32Array[]>;
  /** Frees the weights and every GPU buffer the runner holds. */
  dispose(): void;
}

/** Dot product of two unit-length embeddings, which is their cosine
 *  similarity: 1 for the same meaning, near 0 for unrelated text. */
export function similarity(a: Float32Array, b: Float32Array): number {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i]! * b[i]!;
  return dot;
}

const normalize = (v: Float32Array): Float32Array => {
  let sumSq = 0;
  for (const x of v) sumSq += x * x;
  const inv = 1 / Math.sqrt(sumSq);
  for (let i = 0; i < v.length; i++) v[i]! *= inv;
  return v;
};

/** Loads the checkpoint and the tokenizer, warms every GPU path once, and
 *  returns the runner. With no config, loads all-MiniLM-L6-v2 from the
 *  Hugging Face Hub. */
export async function createTextEmbedder(
  config: TextEmbedderModel = {},
  opts: LoadOptions = {},
): Promise<TextEmbedder> {
  const scope = createResourceScope();
  try {
    const device = defaultRoot().device;
    const arch = config.arch ?? 'minilm';
    const minilm = arch === 'minilm';
    const defaults = minilm
      ? models.textEmbedding.ALL_MINILM_L6_V2.DEFAULT
      : models.textEmbedding.EMBEDDINGGEMMA_2.DEFAULT;
    const fullDim = minilm ? MINILM_DIM : GEMMA_DIM;
    const {
      modelPath = defaults.modelPath,
      tokenizerPath = defaults.tokenizerPath,
      dtype = supportsF16() ? 'f16' : 'f32',
      maxTokens,
      dims = fullDim,
    } = config;
    const allowedDims: readonly number[] = minilm ? [MINILM_DIM] : GEMMA_DIMS;
    if (!allowedDims.includes(dims)) {
      throw new RunntimeError(
        'INVALID_ARGUMENT',
        `createTextEmbedder: dims must be one of ${allowedDims.join(', ')} for ${arch}, got ${dims}`,
      );
    }
    if (maxTokens !== undefined && (!Number.isInteger(maxTokens) || maxTokens < 2)) {
      throw new RunntimeError(
        'INVALID_ARGUMENT',
        `createTextEmbedder: maxTokens must be an integer of at least 2, got ${maxTokens}`,
      );
    }
    if (!minilm && maxTokens !== undefined && maxTokens > EMBEDDINGGEMMA2_MAX_TOKENS) {
      throw new RunntimeError(
        'INVALID_ARGUMENT',
        `createTextEmbedder: maxTokens ${maxTokens} exceeds EmbeddingGemma 2's ${EMBEDDINGGEMMA2_MAX_TOKENS}`,
      );
    }

    // One cache key per file, using the whole path so same-named files of
    // two precisions stay apart. Bump the version to invalidate.
    const cacheId = (path: ModelPath) =>
      typeof path === 'string' ? `${arch}-v1/${path.replace(/^https?:\/\//, '')}` : undefined;
    const sd = await openWeights(modelPath, {
      ...opts,
      cacheId: opts.cacheId ?? cacheId(modelPath),
    });
    const tokenizerJson = await fetchJson(tokenizerPath, {
      ...opts,
      cacheId: cacheId(tokenizerPath),
    });

    throwIfAborted(opts.signal);
    const load = { onProgress: opts.onProgress };
    let encode: (text: string) => number[];
    let core: ReturnType<typeof createMinilmEmbedCore | typeof createEmbeddingGemma2EmbedCore>;
    if (minilm) {
      const tokenizer = createMinilmTokenizer(minilmTokenizerAsset(tokenizerJson, maxTokens));
      encode = (text) => tokenizer.encode(text);
      const model = scope.track(new MinilmModel(MINILM_L6));
      if (dtype === 'f16') model.half();
      await model.loadStateDict(sd, load);
      core = scope.track(createMinilmEmbedCore(model));
    } else {
      const tokenizer = createEmbeddingGemma2Tokenizer(tokenizerJson, maxTokens);
      encode = (text) => tokenizer.encode(text);
      const model = scope.track(new EmbeddingGemma2Model());
      if (dtype === 'f16') model.half();
      await model.loadStateDict(sd, load);
      core = scope.track(createEmbeddingGemma2EmbedCore(model));
    }

    let disposed = false;
    const guard = () => {
      if (disposed) throw new RunntimeError('RESOURCE_DISPOSED', 'text embedder is disposed');
    };

    /** The text with its task's instruction in front. */
    const withTask = (text: string, task: EmbeddingTask | undefined): string => {
      if (task === undefined) return text;
      if (minilm) {
        throw new RunntimeError('INVALID_ARGUMENT', 'embed: MiniLM takes no task');
      }
      if (!Object.hasOwn(EMBEDDINGGEMMA2_PROMPTS, task)) {
        throw new RunntimeError(
          'INVALID_ARGUMENT',
          `embed: task must be one of ${Object.keys(EMBEDDINGGEMMA2_PROMPTS).join(', ')}, got ${task}`,
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
      if (idsList.length === 0) return [];
      // A batch of one is a single text: take the replay path.
      if (idsList.length === 1) return [await embedTokens(idsList[0]!)];
      const pooled = await core.embedPooledBatch(idsList);
      return idsList.map((_, b) => finish(pooled, b * fullDim));
    };

    // Run every GPU path once. First use compiles kernels and records the
    // replays, and warmUp turns a GPU error into one load error. Later
    // calls skip both.
    throwIfAborted(opts.signal);
    await warmUp(device, async () => {
      // The empty text is just the special tokens around it.
      const probe = encode('');
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
        const text = withTask(input, embedOpts.task);
        return embedTokens(encode(text)).catch(rethrowRunError);
      },
      async embedBatch(inputs, embedOpts = {}) {
        guard();
        if (!Array.isArray(inputs) || !inputs.every((t) => typeof t === 'string')) {
          throw new RunntimeError(
            'INVALID_ARGUMENT',
            'embedBatch: inputs must be an array of strings',
          );
        }
        const texts = inputs.map((t) => withTask(t, embedOpts.task));
        return embedTokensBatch(texts.map((t) => encode(t))).catch(rethrowRunError);
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
