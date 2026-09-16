import { defaultRoot, fromSafetensors } from '../../../core/index.ts';
import type { Yolo26Variant } from '../../yolo26/config.ts';
import { createDetector } from '../../yolo26/detector.ts';
import { models } from '../../models.ts';
import { RunntimeYolosForObjectDetection } from './model.ts';
import { cacheKey, cachedWeightsSource } from '../weights.ts';
import type { RunntimeModelLoader } from '../registry.ts';

/** The variant letter in a hub id such as `onnx-community/yolo26s-ONNX`. */
export function yolo26Variant(modelId: string): Yolo26Variant {
  const m = /yolo26([nsmlx])(?![a-z])/i.exec(modelId);
  if (!m) throw new Error(`runntime backend: cannot tell the yolo26 variant from '${modelId}'`);
  return m[1]!.toLowerCase() as Yolo26Variant;
}

/** The hosted f16 exports, by variant. */
const YOLO26_WEIGHTS_URLS: Partial<Record<Yolo26Variant, string>> = {
  n: models.objectDetection.YOLO26.N.modelPath,
  m: models.objectDetection.YOLO26.M.modelPath,
};

/** Every hub id with a hosted export. */
export const YOLO26_MODEL_IDS = ['onnx-community/yolo26n-ONNX', 'onnx-community/yolo26m-ONNX'];

/** `weightsUrl` defaults to the hosted export of the variant in the hub
 *  id; other variants need one. */
export function yolo26Loader(
  weightsUrl?: string,
  opts: {
    variant?: Yolo26Variant;
    onProgress?: (name: string, doneBytes: number, totalBytes: number) => void;
  } = {},
): RunntimeModelLoader {
  return {
    async load(modelId, config) {
      defaultRoot(); // no device: fail before fetching weights, the pipeline falls back to ONNX
      const variant = opts.variant ?? yolo26Variant(modelId);
      const url = weightsUrl ?? YOLO26_WEIGHTS_URLS[variant];
      if (!url) {
        throw new Error(`runntime backend: no hosted yolo26${variant} weights, pass a weightsUrl`);
      }
      const source = await cachedWeightsSource(url, `whole/yolo26-v1/${cacheKey(url)}`);
      const detector = await createDetector(await fromSafetensors(source), {
        task: 'detect',
        variant,
        onProgress: opts.onProgress,
      });
      return new RunntimeYolosForObjectDetection(config, { ...detector, maxDet: 300 });
    },
  };
}
