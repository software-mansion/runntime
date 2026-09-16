import { defaultRoot, fromSafetensors } from '../../../core/index.ts';
import type { PretrainedConfig } from '@huggingface/transformers';
import { createClassifier } from '../../mobilenetv4/classifier.ts';
import { MOBILENETV4_CONV_S } from '../../mobilenetv4/config.ts';
import { models } from '../../models.ts';
import { RunntimeMobileNetV4ForImageClassification } from './model.ts';
import { cacheKey, cachedWeightsSource } from '../weights.ts';
import type { RunntimeModelLoader } from '../registry.ts';

/** The timm checkpoint (f32, 15 MB), loaded as f16 with the batch norms
 *  folded into their convs. */
const MOBILENETV4_WEIGHTS_URL = models.imageClassification.MOBILENETV4.DEFAULT.modelPath;

export const MOBILENETV4_MODEL_IDS = ['onnx-community/mobilenetv4_conv_small.e2400_r224_in1k'];

export function assertMobilenetv4Config(config: PretrainedConfig): void {
  const cfg = config as unknown as Record<string, unknown>;
  const id2label = cfg['id2label'] as Record<string, string> | undefined;
  const numClasses = id2label ? Object.keys(id2label).length : cfg['num_classes'];
  if (numClasses !== undefined && numClasses !== MOBILENETV4_CONV_S.numClasses) {
    throw new Error(
      `runntime backend: mobilenetv4 config has ${String(numClasses)} classes, expected ${MOBILENETV4_CONV_S.numClasses}`,
    );
  }
  const inputSize = cfg['input_size'] as number[] | undefined;
  const side = MOBILENETV4_CONV_S.inputSize;
  if (inputSize && (inputSize[1] !== side || inputSize[2] !== side)) {
    throw new Error(
      `runntime backend: mobilenetv4 config input_size is [${inputSize}], expected [3, ${side}, ${side}]`,
    );
  }
}

export function mobilenetv4Loader(
  weightsUrl: string = MOBILENETV4_WEIGHTS_URL,
  opts: {
    onProgress?: (name: string, doneBytes: number, totalBytes: number) => void;
  } = {},
): RunntimeModelLoader {
  return {
    async load(_modelId, config) {
      defaultRoot(); // fail fast (→ ONNX fallback) before fetching weights
      assertMobilenetv4Config(config);
      const source = await cachedWeightsSource(
        weightsUrl,
        `whole/mobilenetv4-v1/${cacheKey(weightsUrl)}`,
      );
      const classifier = await createClassifier(await fromSafetensors(source), {
        onProgress: opts.onProgress,
      });
      return new RunntimeMobileNetV4ForImageClassification(config, classifier);
    },
  };
}
