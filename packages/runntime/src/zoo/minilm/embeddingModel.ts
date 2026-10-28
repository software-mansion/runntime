/** all-MiniLM-L6-v2 for createEmbedder: English text only, no instructions,
 *  384 values. */

import { createResourceScope, RunntimeError } from '../../core/index.ts';
import type { EmbeddingModel, EmbeddingModelLoad } from '../tasks/multimodal/embedding.ts';
import { EMBEDDING_DIM, MINILM_L6 } from './config.ts';
import { createMinilmEmbedCore } from './embedCore.ts';
import { MinilmModel } from './model.ts';
import { createMinilmTokenizer } from './tokenizer.ts';

export async function createEmbeddingModel(load: EmbeddingModelLoad): Promise<EmbeddingModel> {
  const { maxTokens } = load;
  const most = MINILM_L6.maxPositions;
  if (
    maxTokens !== undefined &&
    (!Number.isInteger(maxTokens) || maxTokens < 2 || maxTokens > most)
  ) {
    throw new RunntimeError(
      'INVALID_ARGUMENT',
      `createEmbedder: maxTokens must be an integer from 2 to ${most}, got ${maxTokens}`,
    );
  }
  const tokenizer = createMinilmTokenizer(load.tokenizer, maxTokens);

  const scope = createResourceScope();
  try {
    const model = scope.track(new MinilmModel(MINILM_L6));
    if (load.dtype === 'f16') model.half();
    await model.loadStateDict(load.text, { onProgress: load.onProgress });
    const core = scope.track(createMinilmEmbedCore(model));
    const pair = [tokenizer.clsId, tokenizer.sepId];

    return {
      dim: EMBEDDING_DIM,
      dims: [EMBEDDING_DIM],
      media: false,
      async warm() {
        await core.embedPooledBatch([pair, pair]);
        await core.warmBuckets(pair);
        return core.embedPooled(pair);
      },
      pool(_where, inputs) {
        // The runner lets only text through.
        const ids = inputs.map((input) =>
          tokenizer.encode(
            typeof input === 'string' ? input : (input as readonly string[]).join(''),
          ),
        );
        // A batch of one is a single text: take the replay path.
        return ids.length === 1 ? core.embedPooled(ids[0]!) : core.embedPooledBatch(ids);
      },
      dispose: () => scope.dispose(),
    };
  } catch (err) {
    scope.dispose();
    throw err;
  }
}
