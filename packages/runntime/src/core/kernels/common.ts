import { d } from 'typegpu';
import type { Elem } from './elem.ts';
import type { StorageFlag, TgpuBindGroup, TgpuBuffer, TgpuComputePipeline } from 'typegpu';

export const WORKGROUP_SIZE = 64;

export const MAX_WORKGROUPS_PER_DIM = 65535;

export const gridYStride = (workgroupSize: number) => MAX_WORKGROUPS_PER_DIM * workgroupSize;

export const GRID_Y_STRIDE = gridYStride(WORKGROUP_SIZE); // 4_194_240

export const makeFlatIndex = (workgroupSize: number) => {
  const stride = gridYStride(workgroupSize);
  return (gid: d.v3u): number => {
    'use gpu';
    return gid.x + gid.y * stride;
  };
};
export const flatIndex = makeFlatIndex(WORKGROUP_SIZE);

/** Row index for a one-workgroup-per-row kernel: folds the executor's 2-D
 *  split back in. Not for kernels whose handle passes an explicit [x, y]. */
export const flatWorkgroupId = (wid: d.v3u): number => {
  'use gpu';
  return wid.x + wid.y * MAX_WORKGROUPS_PER_DIM;
};

export interface KernelHandle {
  pipeline: TgpuComputePipeline;
  bindGroup: TgpuBindGroup;
  workgroups: number | readonly [number, number];
  name: string;
}

export type V4Buffer = TgpuBuffer<d.WgslArray<d.Vec4f>> & StorageFlag;
export type U32Buffer = TgpuBuffer<d.WgslArray<d.U32>> & StorageFlag;
export type F32Buffer = TgpuBuffer<d.WgslArray<d.F32>> & StorageFlag;
export type F16Buffer = TgpuBuffer<d.WgslArray<d.F16>> & StorageFlag;
export type FloatBuffer = F32Buffer | F16Buffer;

export function f32Filler(
  buffers: { q: FloatBuffer; dummyF32?: F32Buffer },
  elem: Elem,
): F32Buffer {
  if (buffers.dummyF32 !== undefined) return buffers.dummyF32;
  if (elem.key !== 'f32') throw new Error('f16 attention needs an f32 dummyF32 filler');
  return buffers.q as F32Buffer;
}

export function makeHandle(
  pipeline: TgpuComputePipeline,
  name: string,
  bindGroup: TgpuBindGroup,
  workgroups: number | readonly [number, number],
  detail?: string,
): KernelHandle {
  return { pipeline: pipeline.$name(name), bindGroup, workgroups, name: detail ?? name };
}
