/** transformers.js plugin: hub ids we implement load as TypeGPU zoo models
 *  through the stock pipeline() call.
 *  - init.ts       one-call setup: device + initRunntime + registerRunntimeBackend
 *  - registry.ts   patches PreTrainedModel.from_pretrained and holds the
 *                  id → loader registry
 *  - weights.ts    OPFS weight caches for the loaders: whole-file for small
 *                  models, byte ranges for big ones
 *  - runntimeModel.ts   isRunntimeModel, knows every wrapper class
 *  One folder per zoo model, each with:
 *  - model.ts      the transformers.js model stand-in running the zoo model
 *  - loader.ts     fetches the weights and builds the stand-in for a hub id */

export { initRunntimeBackend } from './init.ts';
export {
  registerRunntimeBackend,
  unregisterRunntimeBackend,
  type RegisterRunntimeBackendOpts,
  type RunntimeModelLoader,
} from './registry.ts';
export { isRunntimeModel } from './runntimeModel.ts';
export { minilmLoader } from './minilm/loader.ts';
export { privacyFilterLoader } from './privacy-filter/loader.ts';
export { moonshineLoader } from './moonshine/loader.ts';
export { yolo26Loader } from './yolo26/loader.ts';
export { depthartLoader } from './depthart/loader.ts';
export { mobilenetv4Loader } from './mobilenetv4/loader.ts';
