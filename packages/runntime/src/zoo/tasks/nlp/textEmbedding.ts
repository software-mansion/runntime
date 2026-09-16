/** Text embedding task: text in, one unit-length vector out. Runs
 *  all-MiniLM-L6-v2 on the initRunntime() device. */

import { createResourceScope, defaultRoot, supportsF16, warmUp } from '../../../core/index.ts';
import {
  fetchJson,
  openWeights,
  throwIfAborted,
  type LoadOptions,
  type ModelPath,
} from '../../load.ts';
import { EMBEDDING_DIM, MINILM_L6 } from '../../minilm/config.ts';
import { createMinilmEmbedCore } from '../../minilm/embedCore.ts';
import { MinilmModel } from '../../minilm/model.ts';
import { createMinilmTokenizer, minilmTokenizerAsset } from '../../minilm/tokenizer.ts';
import { models } from '../../models.ts';

/** What to load. Every field is optional; a missing one comes from
 *  models.textEmbedding.ALL_MINILM_L6_V2.default. */
export interface TextEmbedderModel {
  /** The safetensors checkpoint. */
  readonly modelPath?: ModelPath;
  /** The tokenizer JSON: a Hugging Face tokenizer.json or our compact asset. */
  readonly tokenizerPath?: string;
  /** Precision on the GPU. `f16` runs the whole model in half precision,
   *  weights and activations, for half the memory traffic; results move by
   *  about 1e-3. Needs a device with `shader-f16`. Default: `f16` where
   *  the device has it, `f32` elsewhere. Either loads any weight file. */
  readonly dtype?: 'f32' | 'f16';
  /** Longest input in tokens, [CLS] and [SEP] included. Longer text is cut.
   *  Default 256, the model card's limit. */
  readonly maxTokens?: number;
}

export interface TextEmbedder {
  /** Length of every returned vector. */
  readonly dim: number;
  /** One text → unit-length vector. */
  embed(input: string): Promise<Float32Array>;
  /** Many texts in one GPU pass. Much faster than embed() in a loop. */
  embedBatch(inputs: readonly string[]): Promise<Float32Array[]>;
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
    const {
      modelPath = models.textEmbedding.ALL_MINILM_L6_V2.DEFAULT.modelPath,
      tokenizerPath = models.textEmbedding.ALL_MINILM_L6_V2.DEFAULT.tokenizerPath,
      dtype = supportsF16() ? 'f16' : 'f32',
      maxTokens,
    } = config;

    // One cache key per file, using the whole path so same-named files of
    // two precisions stay apart. Bump the version to invalidate.
    const cacheId = (path: ModelPath) =>
      typeof path === 'string' ? `minilm-v1/${path.replace(/^https?:\/\//, '')}` : undefined;
    const sd = await openWeights(modelPath, {
      ...opts,
      cacheId: opts.cacheId ?? cacheId(modelPath),
    });
    const tokenizer = createMinilmTokenizer(
      minilmTokenizerAsset(
        await fetchJson(tokenizerPath, { ...opts, cacheId: cacheId(tokenizerPath) }),
        maxTokens,
      ),
    );

    throwIfAborted(opts.signal);
    const model = scope.track(new MinilmModel(MINILM_L6));
    if (dtype === 'f16') model.half();
    await model.loadStateDict(sd, { onProgress: opts.onProgress });
    const core = scope.track(createMinilmEmbedCore(model));

    let disposed = false;
    const guard = () => {
      if (disposed) throw new Error('text embedder is disposed');
    };

    const embedTokens = async (ids: readonly number[]): Promise<Float32Array> =>
      normalize(await core.embedPooled(ids));

    const embedTokensBatch = async (
      idsList: readonly (readonly number[])[],
    ): Promise<Float32Array[]> => {
      if (idsList.length === 0) return [];
      // A batch of one is a single sentence: take the replay path.
      if (idsList.length === 1) return [await embedTokens(idsList[0]!)];
      const pooled = await core.embedPooledBatch(idsList);
      return idsList.map((_, b) =>
        normalize(pooled.slice(b * EMBEDDING_DIM, (b + 1) * EMBEDDING_DIM)),
      );
    };

    // Run every GPU path once. First use compiles kernels and records the
    // replays, and warmUp turns a GPU error into one load error. Later
    // calls skip both.
    throwIfAborted(opts.signal);
    await warmUp(device, async () => {
      const pair = [tokenizer.clsId, tokenizer.sepId];
      await embedTokensBatch([pair, pair]);
      await core.warmBuckets(pair);
      return embedTokens(pair);
    });

    return {
      dim: EMBEDDING_DIM,
      async embed(input) {
        guard();
        return embedTokens(tokenizer.encode(input));
      },
      async embedBatch(inputs) {
        guard();
        return embedTokensBatch(inputs.map((t) => tokenizer.encode(t)));
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
