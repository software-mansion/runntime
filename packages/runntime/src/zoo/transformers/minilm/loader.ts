import { defaultRoot, inGpuErrorScopes, supportsF16 } from '../../../core/index.ts';
import { MINILM_L6 } from '../../minilm/config.ts';
import { MinilmModel } from '../../minilm/model.ts';
import { models } from '../../models.ts';
import { RunntimeBertModel } from './model.ts';
import { cacheKey, loadCachedWeights } from '../weights.ts';
import type { RunntimeModelLoader } from '../registry.ts';

/** The hosted f16 export; loadStateDict fuses qkv and folds the token-type
 *  embedding while loading it. */
const MINILM_WEIGHTS_URL = models.textEmbedding.ALL_MINILM_L6_V2.DEFAULT.modelPath;

export const MINILM_MODEL_IDS = [
  'Xenova/all-MiniLM-L6-v2',
  'sentence-transformers/all-MiniLM-L6-v2',
];

export function minilmLoader(weightsUrl: string = MINILM_WEIGHTS_URL): RunntimeModelLoader {
  return {
    async load(_modelId, config) {
      defaultRoot(); // fail fast (→ ONNX fallback) before fetching weights
      const sd = await loadCachedWeights(weightsUrl, `whole/minilm-v1/${cacheKey(weightsUrl)}`);
      const model = new MinilmModel(MINILM_L6);
      if (supportsF16()) model.half();
      await model.loadStateDict(sd);
      const wrapped = new RunntimeBertModel(config, model);
      await inGpuErrorScopes(defaultRoot().device, 'plugin warm-up', () => wrapped.warm());
      return wrapped;
    },
  };
}
