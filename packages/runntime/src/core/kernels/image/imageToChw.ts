/** GPU image input: a texture fitted into the model input and written as
 *  normalized RGB CHW f32, with the same math as the CPU image preprocessor. */

import tgpu, { d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import { flatIndex, WORKGROUP_SIZE, type F32Buffer } from '../common.ts';
import { cachedBindGroup, cachedUniform } from '../../gpu/dispatchCache.ts';

/** How the image is fitted into the input, as the CPU preprocessor's
 *  ResizeMode. */
export type FitMode = 'stretch' | 'letterbox' | 'crop';

const FIT_MODES: Record<FitMode, number> = { stretch: 0, letterbox: 1, crop: 2 };

/** A texel's [0, 1] value stands for a byte 0–255, and value = byte * alpha
 *  + beta per channel. `pad` is the normalized color outside the image. */
const Params = d.struct({
  mode: d.u32,
  cropFraction: d.f32,
  outW: d.u32,
  outH: d.u32,
  alphaR: d.f32,
  alphaG: d.f32,
  alphaB: d.f32,
  betaR: d.f32,
  betaG: d.f32,
  betaB: d.f32,
  padR: d.f32,
  padG: d.f32,
  padB: d.f32,
});

const ioLayout = tgpu.bindGroupLayout({
  out: { storage: d.arrayOf(d.f32), access: 'mutable' },
  params: { uniform: Params },
});

const textureLayout = tgpu.bindGroupLayout({
  src: { texture: d.texture2d(d.f32), sampleType: 'unfilterable-float' },
});

const externalLayout = tgpu.bindGroupLayout({
  src: { externalTexture: d.textureExternal() },
});

const textureSource = {
  size: (): d.v2u => {
    'use gpu';
    return std.textureDimensions(textureLayout.$.src);
  },
  load: (p: d.v2i): d.v4f => {
    'use gpu';
    return std.textureLoad(textureLayout.$.src, p, 0);
  },
};

const externalSource = {
  size: (): d.v2u => {
    'use gpu';
    return std.textureDimensions(externalLayout.$.src);
  },
  load: (p: d.v2i): d.v4f => {
    'use gpu';
    return std.textureLoad(externalLayout.$.src, p);
  },
};

// The image size is only known on the GPU (an external texture has none on
// the CPU), so the fit is computed here. Keep it in step with
// resizeTransform in zoo/tasks/cv/ops/point.ts.
const makeImageToChwKernel = (source: typeof textureSource) =>
  tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid }) => {
    'use gpu';
    const P = ioLayout.$.params;
    const idx = flatIndex(gid);
    const hw = P.outW * P.outH;
    if (idx >= hw) {
      return;
    }
    const srcSize = source.size();
    const img = d.vec2f(srcSize);
    const out = d.vec2f(d.f32(P.outW), d.f32(P.outH));
    // Input pixel = image pixel * scale + offset.
    let scale = out / img;
    let offset = d.vec2f();
    if (P.mode === 1) {
      scale = d.vec2f(std.min(scale.x, scale.y));
      offset = (out - img * scale) / 2;
    }
    if (P.mode === 2) {
      const side = std.min(img.x, img.y) * P.cropFraction;
      scale = out / side;
      offset = ((side - img) / 2) * scale;
    }

    const o = d.vec2f(d.f32(idx % P.outW), d.f32(d.u32(idx / P.outW)));
    // Image coordinate of this input pixel's center, in image pixels.
    const s = (o + 0.5 - offset) / scale;
    let v = d.vec3f(P.padR, P.padG, P.padB);
    if (s.x >= 0 && s.y >= 0 && s.x < img.x && s.y < img.y) {
      // Half-pixel bilinear with clamped edges, as buildAxis does on the CPU.
      const f = s - 0.5;
      const a = std.floor(f);
      const w = std.select(f - a, d.vec2f(), std.lt(a, d.vec2f()));
      const last = d.vec2i(srcSize) - 1;
      const i0 = std.max(d.vec2i(a), d.vec2i());
      const i1 = std.min(d.vec2i(a) + 1, last);
      const top = std.mix(source.load(i0).xyz, source.load(d.vec2i(i1.x, i0.y)).xyz, w.x);
      const bot = std.mix(source.load(d.vec2i(i0.x, i1.y)).xyz, source.load(i1).xyz, w.x);
      const alpha = d.vec3f(P.alphaR, P.alphaG, P.alphaB);
      v = std.mix(top, bot, w.y) * 255 * alpha + d.vec3f(P.betaR, P.betaG, P.betaB);
    }
    ioLayout.$.out[idx] = v.x;
    ioLayout.$.out[hw + idx] = v.y;
    ioLayout.$.out[2 * hw + idx] = v.z;
  });

/** How the image is fitted and how its bytes become model values: the CPU
 *  preprocessor's options, resolved per channel. */
export interface ImageToChwArgs {
  readonly mode: FitMode;
  /** `crop` only, in (0, 1]. */
  readonly cropFraction: number;
  readonly outW: number;
  readonly outH: number;
  readonly alpha: readonly [number, number, number];
  readonly beta: readonly [number, number, number];
  /** Normalized RGB written where the image does not cover the input. */
  readonly pad: readonly [number, number, number];
}

/** Records texture-to-input passes. Create once per model: it holds the two
 *  pipelines, one for `texture_2d<f32>` and one for external textures. */
export function createImageToChw(root: TgpuRoot) {
  const pipelines = {
    texture: root.createComputePipeline({ compute: makeImageToChwKernel(textureSource) }),
    external: root.createComputePipeline({ compute: makeImageToChwKernel(externalSource) }),
  };
  return {
    /** Fits `src` into `out`, `[3, outH, outW]` f32. */
    encode(
      encoder: GPUCommandEncoder,
      src: GPUTexture | GPUExternalTexture,
      out: F32Buffer,
      args: ImageToChwArgs,
    ): void {
      const { mode, cropFraction, outW, outH, alpha, beta, pad } = args;
      // Nothing here depends on the image, so the uniform is never rewritten
      // and two calls in one encoder cannot see each other's values.
      const params = cachedUniform(root, Params, {
        mode: FIT_MODES[mode],
        cropFraction,
        outW,
        outH,
        alphaR: alpha[0],
        alphaG: alpha[1],
        alphaB: alpha[2],
        betaR: beta[0],
        betaG: beta[1],
        betaB: beta[2],
        padR: pad[0],
        padG: pad[1],
        padB: pad[2],
      });
      const io = cachedBindGroup(root, ioLayout, { out, params });
      const workgroups = Math.ceil((outW * outH) / WORKGROUP_SIZE);
      const pass = encoder.beginComputePass();
      // A bind group per call: external textures expire every frame, and a
      // cache would keep every texture the caller ever passed alive.
      if ('format' in src) {
        const tex = root.createBindGroup(textureLayout, { src });
        pipelines.texture.with(pass).with(io).with(tex).dispatchWorkgroups(workgroups);
      } else {
        const ext = root.createBindGroup(externalLayout, { src });
        pipelines.external.with(pass).with(io).with(ext).dispatchWorkgroups(workgroups);
      }
      pass.end();
    },
  };
}
