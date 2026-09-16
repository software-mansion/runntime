import { defaultRoot, fromSafetensors, inGpuErrorScopes } from '../../../core/index.ts';
import { createEstimator } from '../../depthart/estimator.ts';
import { models } from '../../models.ts';
import { RunntimeDepthartForDepthEstimation } from './model.ts';
import { cacheKey, cachedWeightsSource } from '../weights.ts';
import type { RunntimeModelLoader } from '../registry.ts';

/** The hosted f16 exports, keyed by the depth-estimation hub id they stand
 *  in for: one whose image processor normalizes with ImageNet statistics,
 *  since the pipeline keeps using that id's config and processor. */
const DEPTHART_WEIGHTS_URLS: Record<string, string> = {
  'onnx-community/depth-anything-v2-small': models.depthEstimation.DEPTHART.S.modelPath,
  'onnx-community/depth-anything-v2-base': models.depthEstimation.DEPTHART.B.modelPath,
};

export const DEPTHART_MODEL_IDS = Object.keys(DEPTHART_WEIGHTS_URLS);

/** `weightsUrl`: a safetensors export of a Fengxue93/DepthART relative
 *  checkpoint (tools/export_weights_depthart.py); defaults to the hosted
 *  export for the hub id. The variant is read from the file. */
export function depthartLoader(
  weightsUrl?: string,
  opts: {
    onProgress?: (name: string, doneBytes: number, totalBytes: number) => void;
  } = {},
): RunntimeModelLoader {
  return {
    async load(modelId, config) {
      const root = defaultRoot(); // fail fast (→ ONNX fallback) before fetching weights
      const url = weightsUrl ?? DEPTHART_WEIGHTS_URLS[modelId];
      if (!url) throw new Error(`runntime backend: no hosted DepthART weights for '${modelId}'`);
      const source = await cachedWeightsSource(url, `whole/depthart-v1/${cacheKey(url)}`);
      const sd = await fromSafetensors(source);
      const estimator = await inGpuErrorScopes(root.device, 'plugin warm-up', () =>
        createEstimator(sd, { onProgress: opts.onProgress }),
      );
      return new RunntimeDepthartForDepthEstimation(config, estimator);
    },
  };
}
