/** GPU map output: a single-channel f32 map drawn into any renderable
 *  texture, resampled to its size, optionally rescaled to [0, 1]. */

import tgpu, { common, d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import { WORKGROUP_SIZE, type F32Buffer } from '../common.ts';
import { cachedBindGroup, cachedUniform } from '../../gpu/dispatchCache.ts';

/** `minmax` 1 maps the map's own [min, max] to [0, 1]. */
const Params = d.struct({ width: d.u32, height: d.u32, minmax: d.u32 });

const rangeLayout = tgpu.bindGroupLayout({
  map: { storage: d.arrayOf(d.f32), access: 'readonly' },
  range: { storage: d.arrayOf(d.f32), access: 'mutable' },
  params: { uniform: Params },
});

const drawLayout = tgpu.bindGroupLayout({
  map: { storage: d.arrayOf(d.f32), access: 'readonly' },
  range: { storage: d.arrayOf(d.f32), access: 'readonly' },
  params: { uniform: Params },
});

const TREE_STRIDES = Array.from(
  { length: Math.log2(WORKGROUP_SIZE) },
  (_, level) => WORKGROUP_SIZE >> (level + 1),
);

const partialLo = tgpu.workgroupVar(d.arrayOf(d.f32, WORKGROUP_SIZE));
const partialHi = tgpu.workgroupVar(d.arrayOf(d.f32, WORKGROUP_SIZE));

/** Largest finite f32, the seed of the min/max fold. */
const F32_MAX = 3.4028234663852886e38;

/** One workgroup: threads stride the map, then a pairwise tree. */
const rangeKernel = tgpu.computeFn({
  in: { lid: d.builtin.localInvocationId },
  workgroupSize: [WORKGROUP_SIZE],
})(({ lid }) => {
  'use gpu';
  const P = rangeLayout.$.params;
  const tid = lid.x;
  let lo = d.f32(F32_MAX);
  let hi = d.f32(-F32_MAX);
  for (let i = tid; i < P.width * P.height; i += WORKGROUP_SIZE) {
    const v = rangeLayout.$.map[i]!;
    lo = std.min(lo, v);
    hi = std.max(hi, v);
  }
  partialLo.$[tid] = lo;
  partialHi.$[tid] = hi;
  std.workgroupBarrier();
  for (const stride of tgpu.unroll(TREE_STRIDES)) {
    if (tid < stride) {
      partialLo.$[tid] = std.min(partialLo.$[tid]!, partialLo.$[tid + stride]!);
      partialHi.$[tid] = std.max(partialHi.$[tid]!, partialHi.$[tid + stride]!);
    }
    std.workgroupBarrier();
  }
  if (tid === 0) {
    rangeLayout.$.range[0] = partialLo.$[0]!;
    rangeLayout.$.range[1] = partialHi.$[0]!;
  }
});

const drawMap = tgpu.fragmentFn({ in: { uv: d.vec2f }, out: d.vec4f })(({ uv }) => {
  'use gpu';
  const P = drawLayout.$.params;
  // Half-pixel bilinear with clamped edges, in map pixels.
  const s = uv * d.vec2f(d.f32(P.width), d.f32(P.height)) - 0.5;
  const a = std.floor(s);
  const w = s - a;
  const last = d.vec2i(d.i32(P.width) - 1, d.i32(P.height) - 1);
  const i0 = std.clamp(d.vec2i(a), d.vec2i(), last);
  const i1 = std.clamp(d.vec2i(a) + 1, d.vec2i(), last);
  const W = d.i32(P.width);
  const map = drawLayout.$.map;
  const top = std.mix(map[i0.y * W + i0.x]!, map[i0.y * W + i1.x]!, w.x);
  const bot = std.mix(map[i1.y * W + i0.x]!, map[i1.y * W + i1.x]!, w.x);
  let v = std.mix(top, bot, w.y);
  if (P.minmax === 1) {
    const lo = drawLayout.$.range[0]!;
    // A flat map has no range; it draws as 0 instead of dividing by 0.
    v = (v - lo) / std.max(drawLayout.$.range[1]! - lo, 1e-30);
  }
  return d.vec4f(v, v, v, 1);
});

/** Records map-to-texture passes. Create once per model: it holds one
 *  render pipeline per target format and the [min, max] scratch buffer. */
export function createMapToTexture(root: TgpuRoot) {
  const rangePipeline = root.createComputePipeline({ compute: rangeKernel });
  const range = root.createBuffer(d.arrayOf(d.f32, 2)).$usage('storage');
  const drawPipelineFor = (format: GPUTextureFormat) =>
    root.createRenderPipeline({
      vertex: common.fullScreenTriangle,
      fragment: drawMap,
      targets: { format },
    });
  const drawPipelines = new Map<GPUTextureFormat, ReturnType<typeof drawPipelineFor>>();
  return {
    /** Draws `map`, `[height, width]` f32, over all of `target` mip 0. */
    encode(
      encoder: GPUCommandEncoder,
      map: F32Buffer,
      target: GPUTexture,
      args: { width: number; height: number; normalize: 'none' | 'minmax' },
    ): void {
      const { width, height } = args;
      const minmax = args.normalize === 'minmax' ? 1 : 0;
      const params = cachedUniform(root, Params, { width, height, minmax });
      if (minmax) {
        const bindGroup = cachedBindGroup(root, rangeLayout, { map, range, params });
        const pass = encoder.beginComputePass();
        rangePipeline.with(pass).with(bindGroup).dispatchWorkgroups(1);
        pass.end();
      }
      let pipeline = drawPipelines.get(target.format);
      if (!pipeline) {
        pipeline = drawPipelineFor(target.format);
        drawPipelines.set(target.format, pipeline);
      }
      const view = target.createView({
        dimension: '2d',
        baseMipLevel: 0,
        mipLevelCount: 1,
        baseArrayLayer: 0,
        arrayLayerCount: 1,
      });
      pipeline
        .with(encoder)
        .with(cachedBindGroup(root, drawLayout, { map, range, params }))
        .withColorAttachment({ view, loadOp: 'clear', storeOp: 'store' })
        .draw(3);
    },
    dispose(): void {
      range.destroy();
    },
  };
}
