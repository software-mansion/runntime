/** Test images for the cv runners: seeded RGBA bytes, and the same bytes as
 *  a texture. */
import type { TgpuRoot } from 'typegpu';
import { onTestFinished } from 'vitest';

/** Deterministic RGBA bytes, alpha 255, so a failure reproduces. */
export function randomRgba(w: number, h: number, seed: number): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(w * h * 4);
  let s = seed >>> 0;
  for (let i = 0; i < out.length; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    out[i] = i % 4 === 3 ? 255 : s >>> 24;
  }
  return out;
}

/** `bytes` as an rgba8unorm texture, destroyed when the test ends. */
export function rgbaTexture(
  root: TgpuRoot,
  bytes: Uint8Array<ArrayBuffer>,
  w: number,
  h: number,
): GPUTexture {
  const texture = root.device.createTexture({
    size: [w, h],
    format: 'rgba8unorm',
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  root.device.queue.writeTexture({ texture }, bytes, { bytesPerRow: w * 4 }, [w, h]);
  onTestFinished(() => texture.destroy());
  return texture;
}
