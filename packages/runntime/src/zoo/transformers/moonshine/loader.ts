import {
  defaultRoot,
  fromSafetensors,
  inGpuErrorScopes,
  supportsF16,
} from '../../../core/index.ts';
import type { PretrainedConfig } from '@huggingface/transformers';
import {
  MOONSHINE_BASE,
  MOONSHINE_TINY,
  assertConfigMatches,
  configFromCheckpoint,
  type MoonshineConfig,
} from '../../moonshine/config.ts';
import type { MoonshineTokenizer } from '../../moonshine/tokenizer.ts';
import { models } from '../../models.ts';
import { createTranscriber } from '../../moonshine/transcriber.ts';
import { RunntimeMoonshineForConditionalGeneration } from './model.ts';
import { cacheKey, cachedWeightsSource } from '../weights.ts';
import type { RunntimeModelLoader } from '../registry.ts';

/** The hosted f16 exports, picked by the hub config's size. */
const MOONSHINE_WEIGHTS_URLS = {
  tiny: models.speechToText.MOONSHINE.TINY.DEFAULT.modelPath,
  base: models.speechToText.MOONSHINE.BASE.DEFAULT.modelPath,
} as const;

export const MOONSHINE_MODEL_IDS = [
  'onnx-community/moonshine-tiny-ONNX',
  'onnx-community/moonshine-base-ONNX',
];

export function moonshinePreset(config: PretrainedConfig): {
  size: 'tiny' | 'base';
  cfg: MoonshineConfig;
} {
  const parsed = configFromCheckpoint(config as unknown as Record<string, unknown>);
  const size =
    parsed.dModel === MOONSHINE_TINY.dModel
      ? 'tiny'
      : parsed.dModel === MOONSHINE_BASE.dModel
        ? 'base'
        : undefined;
  if (!size) {
    throw new Error(
      `runntime backend: moonshine hidden_size ${parsed.dModel} is neither tiny nor base`,
    );
  }
  const cfg = size === 'tiny' ? MOONSHINE_TINY : MOONSHINE_BASE;
  assertConfigMatches({ ...parsed, maxPositions: cfg.maxPositions }, cfg);
  return { size, cfg };
}

function idsOnlyTokenizer(cfg: MoonshineConfig): MoonshineTokenizer {
  return {
    version: 0,
    bosId: cfg.bos,
    eosId: cfg.eos,
    specialIds: [cfg.bos, cfg.eos],
    vocab: new Array<string>(cfg.vocab).fill(''),
  };
}

export function moonshineLoader(
  weightsUrl?: string,
  opts: {
    weightDtype?: 'f16' | 'f32';
    onProgress?: (name: string, doneBytes: number, totalBytes: number) => void;
  } = {},
): RunntimeModelLoader {
  return {
    async load(_modelId, config) {
      defaultRoot(); // fail fast (→ ONNX fallback) before fetching weights
      const { size, cfg } = moonshinePreset(config);
      const url = weightsUrl ?? MOONSHINE_WEIGHTS_URLS[size];
      const source = await cachedWeightsSource(url, `whole/moonshine-v1/${cacheKey(url)}`);
      const sd = await fromSafetensors(source);
      const transcriber = await createTranscriber(sd, idsOnlyTokenizer(cfg), {
        cfg,
        weightDtype: opts.weightDtype ?? (supportsF16() ? 'f16' : 'f32'),
        onProgress: opts.onProgress,
      });
      // One second of silence compiles every pipeline before the first call.
      await inGpuErrorScopes(defaultRoot().device, 'plugin warm-up', () =>
        transcriber.transcribe(new Float32Array(16_000)),
      );
      return new RunntimeMoonshineForConditionalGeneration(config, transcriber, cfg);
    },
  };
}
