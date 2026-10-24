/** GPU in, GPU out: the types every runner's encode<Verb> method takes, and
 *  how they are unwrapped from TypeGPU and checked. */

import {
  isBuffer,
  isTexture,
  type d,
  type RenderFlag,
  type SampledFlag,
  type TgpuBuffer,
  type TgpuCommandEncoder,
  type TgpuTexture,
} from 'typegpu';
import { defaultRoot, RunntimeError } from '../../core/index.ts';

/** An image already on the GPU. A texel's [0, 1] value stands for a byte
 *  0–255, as in an ImageBuffer. */
export type GpuImage = GPUTexture | GPUExternalTexture | (TgpuTexture & SampledFlag);

/** A texture the runner draws into. */
export type GpuImageTarget = GPUTexture | (TgpuTexture & RenderFlag);

/** A buffer laid out as `Schema`. A raw GPUBuffer must hold the same bytes. */
export type GpuTensor<Schema extends d.BaseData> = GPUBuffer | TgpuBuffer<Schema>;

/** A command encoder to record into. A TgpuCommandEncoder comes from
 *  `root['~unstable'].createCommandEncoder()`. */
export type GpuEncoder = GPUCommandEncoder | TgpuCommandEncoder;

/** The options every encode<Verb> method takes. Every resource must come from
 *  the device passed to initRunntime(). */
export interface EncodeOptions<Source, Target> {
  readonly source: Source;
  readonly target: Target;
  /** Records into this encoder, and the caller finishes and submits it,
   *  before disposing the runner. It must not have an open pass. Default: a
   *  new encoder, submitted before the method returns. */
  readonly encoder?: GpuEncoder;
}

/** Texture formats a GpuImage can have. sRGB formats are left out: the GPU
 *  decodes them to linear values, which are not the bytes the model expects. */
const IMAGE_FORMATS: ReadonlySet<GPUTextureFormat> = new Set<GPUTextureFormat>([
  'rgba8unorm',
  'bgra8unorm',
  'rgb10a2unorm',
  'rgba16float',
  'rgba32float',
]);

/** Texture formats a GpuImageTarget can have: the float-renderable ones. */
const TARGET_FORMATS: ReadonlySet<GPUTextureFormat> = new Set<GPUTextureFormat>([
  'r8unorm',
  'r16float',
  'r32float',
  'rg8unorm',
  'rg16float',
  'rg32float',
  'rgba8unorm',
  'rgba8unorm-srgb',
  'bgra8unorm',
  'bgra8unorm-srgb',
  'rgb10a2unorm',
  'rgba16float',
  'rgba32float',
]);

/** A source or target, unwrapped: an image or a tensor buffer. */
type Unwrapped<Image> = { readonly image: Image } | { readonly tensor: GPUBuffer };

/** Unwraps a source and checks it. A tensor must hold at least `bytes`. */
export function unwrapSource(
  source: GpuImage | GpuTensor<d.BaseData>,
  bytes: number,
): Unwrapped<GPUTexture | GPUExternalTexture> {
  const raw = unwrap(source);
  if ('mapState' in raw) {
    checkBuffer(raw, bytes, GPUBufferUsage.COPY_SRC, 'COPY_SRC', 'source');
    return { tensor: raw };
  }
  if ('format' in raw) checkImage(raw);
  return { image: raw };
}

/** Unwraps a target and checks it. A tensor must hold at least `bytes`. */
export function unwrapTarget(
  target: GpuImageTarget | GpuTensor<d.BaseData>,
  bytes: number,
): Unwrapped<GPUTexture> {
  const raw = unwrap(target);
  if ('mapState' in raw) {
    checkBuffer(raw, bytes, GPUBufferUsage.COPY_DST, 'COPY_DST', 'target');
    return { tensor: raw };
  }
  if (!('format' in raw)) {
    throw new RunntimeError('INVALID_ARGUMENT', 'target: an external texture cannot be drawn into');
  }
  checkTexture(raw, 'target', GPUTextureUsage.RENDER_ATTACHMENT, 'RENDER_ATTACHMENT');
  if (!TARGET_FORMATS.has(raw.format)) {
    throw new RunntimeError(
      'INVALID_ARGUMENT',
      `target: format ${raw.format} is not supported, use one of ${[...TARGET_FORMATS].join(', ')}`,
    );
  }
  return { image: raw };
}

/** Runs `record` on the caller's encoder, or on a new one it then submits. */
export function encodeWith(
  encoder: GpuEncoder | undefined,
  record: (encoder: GPUCommandEncoder) => void,
): void {
  const root = defaultRoot();
  if (encoder !== undefined) {
    record('resourceType' in encoder ? root.unwrap(encoder) : encoder);
    return;
  }
  const own = root.device.createCommandEncoder();
  record(own);
  root.device.queue.submit([own.finish()]);
}

function unwrap(
  resource: GpuImage | GpuImageTarget | GpuTensor<d.BaseData>,
): GPUTexture | GPUExternalTexture | GPUBuffer {
  if (isTexture(resource)) return defaultRoot().unwrap(resource);
  if (isBuffer(resource)) return defaultRoot().unwrap(resource);
  return resource;
}

function checkImage(texture: GPUTexture): void {
  checkTexture(texture, 'source', GPUTextureUsage.TEXTURE_BINDING, 'TEXTURE_BINDING');
  if (!IMAGE_FORMATS.has(texture.format)) {
    throw new RunntimeError(
      'INVALID_ARGUMENT',
      `source: format ${texture.format} is not supported, use one of ${[...IMAGE_FORMATS].join(', ')}`,
    );
  }
  if (texture.depthOrArrayLayers !== 1) {
    throw new RunntimeError(
      'INVALID_ARGUMENT',
      `source: ${texture.depthOrArrayLayers} array layers, expected 1`,
    );
  }
}

function checkTexture(texture: GPUTexture, what: string, usage: number, usageName: string): void {
  if (texture.dimension !== '2d' || texture.sampleCount !== 1) {
    throw new RunntimeError(
      'INVALID_ARGUMENT',
      `${what}: expected a 2d texture with one sample, got ${texture.dimension} with ${texture.sampleCount}`,
    );
  }
  if (!(texture.usage & usage)) {
    throw new RunntimeError('INVALID_ARGUMENT', `${what}: the texture needs ${usageName} usage`);
  }
}

function checkBuffer(
  buffer: GPUBuffer,
  bytes: number,
  usage: number,
  usageName: string,
  what: string,
): void {
  if (buffer.size < bytes) {
    throw new RunntimeError(
      'INVALID_ARGUMENT',
      `${what}: buffer holds ${buffer.size} bytes, expected ${bytes}`,
    );
  }
  if (!(buffer.usage & usage)) {
    throw new RunntimeError('INVALID_ARGUMENT', `${what}: the buffer needs ${usageName} usage`);
  }
}
