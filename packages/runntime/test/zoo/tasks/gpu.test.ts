import { beforeAll, describe, expect, it, vi } from 'vitest';
import { unwrapSource, unwrapStorageTarget, unwrapTarget } from '../../../src/zoo/tasks/gpu.ts';

// The WebGPU usage bits, as the spec numbers them. Node has no WebGPU.
const BUFFER = { COPY_SRC: 0x4, COPY_DST: 0x8, STORAGE: 0x80 };
const TEXTURE = { COPY_DST: 0x2, TEXTURE_BINDING: 0x4, RENDER_ATTACHMENT: 0x10 };

beforeAll(() => {
  vi.stubGlobal('GPUBufferUsage', BUFFER);
  vi.stubGlobal('GPUTextureUsage', TEXTURE);
});

const buffer = (size: number, usage: number) =>
  ({ size, usage, mapState: 'unmapped' }) as GPUBuffer;

const texture = (props: Partial<GPUTexture> = {}) =>
  ({
    format: 'rgba8unorm',
    dimension: '2d',
    sampleCount: 1,
    depthOrArrayLayers: 1,
    usage: TEXTURE.TEXTURE_BINDING | TEXTURE.RENDER_ATTACHMENT,
    ...props,
  }) as GPUTexture;

const external = {} as GPUExternalTexture;

describe('unwrapSource', () => {
  it.each([
    { name: 'a buffer of the input size', source: buffer(64, BUFFER.COPY_SRC), kind: 'tensor' },
    { name: 'a larger buffer', source: buffer(128, BUFFER.COPY_SRC), kind: 'tensor' },
    { name: 'an rgba8unorm texture', source: texture(), kind: 'image' },
    { name: 'a bgra8unorm texture', source: texture({ format: 'bgra8unorm' }), kind: 'image' },
    { name: 'an external texture', source: external, kind: 'image' },
  ])('accepts $name', ({ source, kind }) => {
    expect(Object.keys(unwrapSource(source, 64))).toEqual([kind]);
  });

  it.each([
    { name: 'a buffer smaller than the input', source: buffer(60, BUFFER.COPY_SRC) },
    { name: 'a buffer without COPY_SRC', source: buffer(64, BUFFER.STORAGE) },
    { name: 'an sRGB texture', source: texture({ format: 'rgba8unorm-srgb' }) },
    { name: 'an integer texture', source: texture({ format: 'rgba8uint' }) },
    { name: 'a texture without TEXTURE_BINDING', source: texture({ usage: TEXTURE.COPY_DST }) },
    { name: 'a 3d texture', source: texture({ dimension: '3d' }) },
    { name: 'a texture array', source: texture({ depthOrArrayLayers: 2 }) },
    { name: 'a multisampled texture', source: texture({ sampleCount: 4 }) },
  ])('rejects $name', ({ source }) => {
    expect(() => unwrapSource(source, 64)).toThrow(
      expect.objectContaining({ code: 'INVALID_ARGUMENT' }),
    );
  });
});

describe('unwrapTarget', () => {
  it.each([
    { name: 'a buffer of the map size', target: buffer(64, BUFFER.COPY_DST), kind: 'tensor' },
    { name: 'an rgba8unorm texture', target: texture(), kind: 'image' },
    { name: 'an r32float texture', target: texture({ format: 'r32float' }), kind: 'image' },
    { name: 'an sRGB texture', target: texture({ format: 'bgra8unorm-srgb' }), kind: 'image' },
    { name: 'a texture array', target: texture({ depthOrArrayLayers: 2 }), kind: 'image' },
  ])('accepts $name', ({ target, kind }) => {
    expect(Object.keys(unwrapTarget(target, 64))).toEqual([kind]);
  });

  it.each([
    { name: 'a buffer smaller than the map', target: buffer(60, BUFFER.COPY_DST) },
    { name: 'a buffer without COPY_DST', target: buffer(64, BUFFER.COPY_SRC) },
    { name: 'an external texture', target: external },
    { name: 'an integer texture', target: texture({ format: 'r32uint' }) },
    { name: 'a depth texture', target: texture({ format: 'depth32float' }) },
    {
      name: 'a texture without RENDER_ATTACHMENT',
      target: texture({ usage: TEXTURE.TEXTURE_BINDING }),
    },
    { name: 'a multisampled texture', target: texture({ sampleCount: 4 }) },
  ])('rejects $name', ({ target }) => {
    expect(() => unwrapTarget(target as GPUTexture, 64)).toThrow(
      expect.objectContaining({ code: 'INVALID_ARGUMENT' }),
    );
  });
});

describe('unwrapStorageTarget', () => {
  it.each([
    { name: 'a buffer of the list size', target: buffer(64, BUFFER.STORAGE) },
    { name: 'a larger buffer', target: buffer(128, BUFFER.STORAGE | BUFFER.COPY_SRC) },
  ])('accepts $name', ({ target }) => {
    expect(unwrapStorageTarget(target, 64)).toBe(target);
  });

  it.each([
    { name: 'a buffer smaller than the list', target: buffer(60, BUFFER.STORAGE) },
    { name: 'a buffer without STORAGE', target: buffer(64, BUFFER.COPY_DST) },
    { name: 'a texture', target: texture() },
  ])('rejects $name', ({ target }) => {
    expect(() => unwrapStorageTarget(target as GPUBuffer, 64)).toThrow(
      expect.objectContaining({ code: 'INVALID_ARGUMENT' }),
    );
  });
});
