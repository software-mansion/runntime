/** Test images for the cv runners: seeded RGBA bytes, the same bytes as a
 *  texture, and a seeded video frame with the pixels a canvas reads from it. */
import type { TgpuRoot } from 'typegpu';
import { onTestFinished } from 'vitest';
import { imageBufferFromImageData, type ImageBuffer } from '../../../../src/zoo/tasks/cv/image.ts';

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

/** Limited-range luma stays in 16–235 and chroma in 16–240. These narrower
 *  bands keep every pixel's RGB inside 0–255 after the BT.709 conversion:
 *  Chrome clamps what falls outside, and the two paths may not clamp alike. */
const LUMA_RANGE = [48, 200] as const;
const CHROMA_RANGE = [112, 144] as const;

/** A seeded `w` × `h` I420 VideoFrame, tagged BT.709 limited range, closed
 *  when the test ends. `w` and `h` are even. BT.709 because Chrome applies a
 *  colour conversion to SMPTE 170M frames on the GPU import but not on the
 *  canvas, while BT.709 gives the same pixels on both. */
export function videoFrame(w: number, h: number, seed: number): VideoFrame {
  const luma = w * h;
  const chroma = (w / 2) * (h / 2);
  const planes = new Uint8Array(luma + 2 * chroma);
  let s = seed >>> 0;
  for (let i = 0; i < planes.length; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    const [lo, hi] = i < luma ? LUMA_RANGE : CHROMA_RANGE;
    planes[i] = lo + Math.floor(((s >>> 24) / 256) * (hi - lo + 1));
  }
  const frame = new VideoFrame(planes, {
    format: 'I420',
    codedWidth: w,
    codedHeight: h,
    timestamp: 0,
    colorSpace: { primaries: 'bt709', transfer: 'bt709', matrix: 'bt709', fullRange: false },
  });
  onTestFinished(() => frame.close());
  return frame;
}

/** The frame's pixels as a canvas reads them, for the CPU method. */
export function frameImage(frame: VideoFrame): ImageBuffer {
  const { displayWidth: w, displayHeight: h } = frame;
  const context = new OffscreenCanvas(w, h).getContext('2d')!;
  context.drawImage(frame, 0, 0);
  return imageBufferFromImageData(context.getImageData(0, 0, w, h));
}
