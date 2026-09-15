import { initRunntime } from 'runntime/zoo';
import tgpu from 'typegpu';

let started: Promise<void> | undefined;

/** Creates the WebGPU device and points the engine at it. Every example
 *  awaits this before it loads a model, and the setup runs only once.
 *  `subgroups` and `shader-f16` are optional: kernels take a faster path
 *  where the device has them. */
export function initEngine(): Promise<void> {
  started ??= (async () => {
    if (!navigator.gpu) {
      throw new Error('This browser has no WebGPU. Try Chrome, Edge, or Safari 26 and newer.');
    }
    initRunntime(await tgpu.init({ device: { optionalFeatures: ['subgroups', 'shader-f16'] } }));
  })();
  return started;
}
