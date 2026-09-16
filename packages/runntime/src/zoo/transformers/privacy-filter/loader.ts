import { defaultRoot } from '../../../core/index.ts';
import type { PretrainedConfig } from '@huggingface/transformers';
import { V2_NER_CLASS_NAMES } from '../../privacy-filter/config.ts';
import { loadPrivacyFilterModel } from '../../privacy-filter/detector.ts';
import { models } from '../../models.ts';
import { RunntimePrivacyFilterForTokenClassification } from './model.ts';
import { cacheKey, loadStreamedWeights } from '../weights.ts';
import type { RunntimeModelLoader } from '../registry.ts';

/** The hosted int8 export, 1.55 GB. */
const PRIVACY_FILTER_WEIGHTS_URL = models.privacyFilter.PRIVACY_FILTER.DEFAULT.modelPath;

export const PRIVACY_FILTER_MODEL_IDS = ['openai/privacy-filter'];

const EXPECTED_CONFIG = {
  hidden_size: 640,
  num_hidden_layers: 8,
  num_local_experts: 128,
  num_experts_per_tok: 4,
} as const;

export function assertPrivacyFilterConfig(config: PretrainedConfig): void {
  const cfg = config as unknown as Record<string, unknown>;
  for (const [key, expected] of Object.entries(EXPECTED_CONFIG)) {
    if (cfg[key] !== expected) {
      throw new Error(
        `runntime backend: privacy-filter config ${key} is ${String(cfg[key])}, expected ${expected}`,
      );
    }
  }
  const id2label = cfg['id2label'] as Record<string, string> | undefined;
  if (!id2label) throw new Error('runntime backend: privacy-filter config is missing id2label');
  V2_NER_CLASS_NAMES.forEach((name, i) => {
    if (id2label[String(i)] !== name) {
      throw new Error(
        `runntime backend: privacy-filter label ${i} is '${id2label[String(i)]}', expected '${name}'`,
      );
    }
  });
}

export function privacyFilterLoader(
  weightsUrl: string = PRIVACY_FILTER_WEIGHTS_URL,
  opts: {
    onProgress?: (name: string, doneBytes: number, totalBytes: number) => void;
  } = {},
): RunntimeModelLoader {
  return {
    async load(_modelId, config) {
      defaultRoot(); // fail fast (→ ONNX fallback) before fetching weights
      assertPrivacyFilterConfig(config);
      // The file is gigabytes: streamed by byte range, never whole in memory.
      const sd = await loadStreamedWeights(
        weightsUrl,
        `range/privacy-filter-v1/${cacheKey(weightsUrl)}`,
      );
      const { model } = await loadPrivacyFilterModel(sd, { onProgress: opts.onProgress });
      return new RunntimePrivacyFilterForTokenClassification(config, model);
    },
  };
}
