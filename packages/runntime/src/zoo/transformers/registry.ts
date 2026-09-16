/** registerRunntimeBackend() patches PreTrainedModel.from_pretrained, the one
 *  loader that pipeline() and the AutoModel classes call, so registered hub
 *  ids load as TypeGPU zoo models. Tokenizers, processors and pipeline
 *  pre/post-processing stay stock transformers.js. Unregistered ids and
 *  loader failures throw; fallbackToOnnx warns and uses ONNX instead. */

import { AutoConfig, PreTrainedModel, type PretrainedConfig } from '@huggingface/transformers';
import { MINILM_MODEL_IDS, minilmLoader } from './minilm/loader.ts';
import { PRIVACY_FILTER_MODEL_IDS, privacyFilterLoader } from './privacy-filter/loader.ts';
import { MOONSHINE_MODEL_IDS, moonshineLoader } from './moonshine/loader.ts';
import { MOBILENETV4_MODEL_IDS, mobilenetv4Loader } from './mobilenetv4/loader.ts';
import { YOLO26_MODEL_IDS, yolo26Loader } from './yolo26/loader.ts';
import { DEPTHART_MODEL_IDS, depthartLoader } from './depthart/loader.ts';

export interface RunntimeModelLoader {
  load(modelId: string, config: PretrainedConfig): Promise<PreTrainedModel>;
}

export interface RegisterRunntimeBackendOpts {
  models?: Record<string, RunntimeModelLoader>;
  fallbackToOnnx?: boolean;
}

const registry = new Map<string, RunntimeModelLoader>();
let fallbackToOnnx = false;
let original: typeof PreTrainedModel.from_pretrained | undefined;

type FromPretrainedArgs = Parameters<typeof PreTrainedModel.from_pretrained>;

export function registerRunntimeBackend(opts: RegisterRunntimeBackendOpts = {}): void {
  registry.clear();
  for (const id of MINILM_MODEL_IDS) registry.set(id, minilmLoader());
  for (const id of PRIVACY_FILTER_MODEL_IDS) registry.set(id, privacyFilterLoader());
  for (const id of MOONSHINE_MODEL_IDS) registry.set(id, moonshineLoader());
  for (const id of MOBILENETV4_MODEL_IDS) registry.set(id, mobilenetv4Loader());
  for (const id of YOLO26_MODEL_IDS) registry.set(id, yolo26Loader());
  for (const id of DEPTHART_MODEL_IDS) registry.set(id, depthartLoader());
  if (opts.models) for (const [id, loader] of Object.entries(opts.models)) registry.set(id, loader);
  fallbackToOnnx = opts.fallbackToOnnx ?? false;

  if (original) return;
  original = PreTrainedModel.from_pretrained;
  PreTrainedModel.from_pretrained = async function (
    this: typeof PreTrainedModel,
    ...args: FromPretrainedArgs
  ) {
    const [modelId, options = {}] = args;
    const loader = typeof modelId === 'string' ? registry.get(modelId) : undefined;
    if (loader) {
      try {
        const config = await AutoConfig.from_pretrained(modelId, options);
        return await loader.load(modelId, config);
      } catch (e) {
        if (!fallbackToOnnx) {
          throw new Error(
            `runntime backend: loading '${modelId}' failed. ` +
              `Pass fallbackToOnnx: true to load it on ONNX instead.`,
            { cause: e },
          );
        }
        console.warn(`runntime backend: loading '${modelId}' failed, falling back to ONNX`, e);
      }
    } else if (!fallbackToOnnx) {
      const ids = [...registry.keys()].join(', ');
      throw new Error(
        `runntime backend: '${String(modelId)}' is not supported (registered ids: ${ids}). ` +
          `Pass fallbackToOnnx: true to load unsupported models on ONNX.`,
      );
    }
    return original!.call(this, ...args);
  } as typeof PreTrainedModel.from_pretrained;
}

export function unregisterRunntimeBackend(): void {
  if (original) {
    PreTrainedModel.from_pretrained = original;
    original = undefined;
  }
  registry.clear();
  fallbackToOnnx = false;
}
