/** The encoders a caller can hand an encode<Verb> method, each with how the
 *  caller then submits it. */
import type { TgpuRoot } from 'typegpu';
import type { GpuEncoder } from '../../../src/zoo/tasks/gpu.ts';

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
