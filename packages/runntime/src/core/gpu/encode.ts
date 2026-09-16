/** Encoding KernelHandles into a compute pass. */

import { type KernelHandle, MAX_WORKGROUPS_PER_DIM } from '../kernels/common.ts';

export function workgroupGrid(h: KernelHandle): readonly [number, number] {
  if (typeof h.workgroups !== 'number') return h.workgroups;
  return h.workgroups > MAX_WORKGROUPS_PER_DIM
    ? [MAX_WORKGROUPS_PER_DIM, Math.ceil(h.workgroups / MAX_WORKGROUPS_PER_DIM)]
    : [h.workgroups, 1];
}

export function encodeHandle(pass: GPUComputePassEncoder, h: KernelHandle): void {
  const [x, y] = workgroupGrid(h);
  h.pipeline.with(pass).with(h.bindGroup).dispatchWorkgroups(x, y);
}

export function encodePass(
  encoder: GPUCommandEncoder,
  handles: readonly KernelHandle[],
  descriptor: GPUComputePassDescriptor = {},
): void {
  const pass = encoder.beginComputePass(descriptor);
  for (const h of handles) encodeHandle(pass, h);
  pass.end();
}
