/** Shared pieces of the encode<Verb> tests: freeing the buffers a test
 *  makes, and the encoders a caller can hand a method, each with how the
 *  caller then submits it. */
import type { TgpuRoot } from 'typegpu';
import { onTestFinished } from 'vitest';
import type { GpuEncoder } from '../../../src/zoo/tasks/gpu.ts';

/** `resource`, destroyed when the test ends. */
export function freed<Resource extends { destroy(): void }>(resource: Resource): Resource {
  onTestFinished(() => resource.destroy());
  return resource;
}

export const ENCODERS = [
  {
    name: 'a TypeGPU encoder',
    record(root: TgpuRoot, encode: (encoder: GpuEncoder) => void): void {
      const encoder = root['~unstable'].createCommandEncoder();
      encode(encoder);
      encoder.submit();
    },
  },
  {
    name: 'a GPUCommandEncoder',
    record(root: TgpuRoot, encode: (encoder: GpuEncoder) => void): void {
      const encoder = root.device.createCommandEncoder();
      encode(encoder);
      root.device.queue.submit([encoder.finish()]);
    },
  },
];
