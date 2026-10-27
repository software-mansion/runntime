/** Snake over time-major [T, C]: x + inv·sin²(αx) per channel, with `ab`
 *  f32 [2, C] holding α and then inv = 1 / (α + 1e-9). One thread per 4
 *  channels of one frame; C is a multiple of 4. */

import tgpu, { d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import {
  type F32Buffer,
  type FloatBuffer,
  flatIndex,
  type KernelHandle,
  makeHandle,
  WORKGROUP_SIZE,
} from '../common.ts';
import { cachedBindGroup, cachedUniform } from '../../gpu/dispatchCache.ts';
import { type Elem, F32_ELEM } from '../elem.ts';

const Dims = d.struct({ total4: d.u32, c4: d.u32 });

function makeSnakeVariant(elem: Elem) {
  const layout = tgpu.bindGroupLayout({
    x: { storage: d.arrayOf(elem.vec4), access: 'readonly' },
    ab: { storage: d.arrayOf(d.vec4f), access: 'readonly' },
    out: { storage: d.arrayOf(elem.vec4), access: 'mutable' },
    dims: { uniform: Dims },
  });
  const store = elem.vec4;
  const kernel = tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const i = flatIndex(gid);
    if (i >= layout.$.dims.total4) return;
    const c = i % layout.$.dims.c4;
    const x = d.vec4f(layout.$.x[i]!);
    const s = std.sin(layout.$.ab[c]! * x);
    layout.$.out[i] = store(x + layout.$.ab[layout.$.dims.c4 + c]! * s * s);
  });
  return { layout, kernel };
}

const variants = new Map<string, ReturnType<typeof makeSnakeVariant>>();
function snakeVariant(elem: Elem): ReturnType<typeof makeSnakeVariant> {
  let v = variants.get(elem.key);
  if (!v) {
    v = makeSnakeVariant(elem);
    variants.set(elem.key, v);
  }
  return v;
}

export function createSnakePipeline(root: TgpuRoot, elem: Elem = F32_ELEM) {
  return root.createComputePipeline({ compute: snakeVariant(elem).kernel });
}

export function snakeHandle(
  root: TgpuRoot,
  pipeline: ReturnType<typeof createSnakePipeline>,
  rows: number,
  cols: number,
  buffers: { x: FloatBuffer; ab: F32Buffer; out: FloatBuffer },
  elem: Elem = F32_ELEM,
): KernelHandle {
  const total4 = (rows * cols) / 4;
  const dims = cachedUniform(root, Dims, { total4, c4: cols / 4 });
  const bindGroup = cachedBindGroup(root, snakeVariant(elem).layout, { ...buffers, dims });
  return makeHandle(pipeline, 'snake', bindGroup, Math.ceil(total4 / WORKGROUP_SIZE));
}
