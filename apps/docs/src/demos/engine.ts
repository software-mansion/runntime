import { initRunntime } from 'runntime/zoo';
import tgpu, { type TgpuRoot } from 'typegpu';

let started: Promise<TgpuRoot> | undefined;

/** Creates the WebGPU device and points the engine at it. Every example
 *  awaits this before it loads a model, and the setup runs only once.
 *  `subgroups` and `shader-f16` are optional: kernels take a faster path
 *  where the device has them. Returns the root, for examples that hand the
 *  model GPU resources. */
export function initEngine(): Promise<TgpuRoot> {
  started ??= (async () => {
    if (!navigator.gpu) {
      throw new Error('This browser has no WebGPU. Try Chrome, Edge, or Safari 26 and newer.');
    }
    const root = await tgpu.init({ device: { optionalFeatures: ['subgroups', 'shader-f16'] } });
    initRunntime(root);
    return root;
  })();
  return started;
}
