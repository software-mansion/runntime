import tgpu from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import { initRunntime } from '../../core/index.ts';
import { registerRunntimeBackend, type RegisterRunntimeBackendOpts } from './registry.ts';

/** One-call plugin setup: creates the WebGPU device, points runntime/core at it
 *  and patches transformers.js. Requests subgroups, which the fast matmul
 *  and attention kernels need, and shader-f16, which yolo26 and depthart
 *  require and the other models use when present. Both are optional: a
 *  device lacking either still loads the f32-capable models. Options go
 *  straight to registerRunntimeBackend(), so `models` and `fallbackToOnnx` work
 *  here too. Returns the root for apps that also use TypeGPU directly. */
export async function initRunntimeBackend(
  opts: RegisterRunntimeBackendOpts = {},
): Promise<TgpuRoot> {
  const root = await tgpu.init({ device: { optionalFeatures: ['subgroups', 'shader-f16'] } });
  initRunntime(root);
  registerRunntimeBackend(opts);
  return root;
}
