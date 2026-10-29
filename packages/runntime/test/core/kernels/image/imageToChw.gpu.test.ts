import { d } from 'typegpu';
import { describe, expect, it, onTestFinished } from 'vitest';
import {
  createImageToChw,
  type FitMode,
  type ImageToChwArgs,
} from '../../../../src/core/kernels/image/imageToChw.ts';
import { imageBuffer } from '../../../../src/zoo/tasks/cv/image.ts';
import { createImagePreprocessor } from '../../../../src/zoo/tasks/cv/utils/imagePreprocessor.ts';
import { expectClose, gpu } from '../../../gpu.ts';

const { root } = await gpu();
const imageToChw = createImageToChw(root);

/** Deterministic RGBA bytes, alpha 255, so a failure reproduces. */
function randomRgba(w: number, h: number, seed: number): Uint8Array<ArrayBuffer> {
  const out = new Uint8Array(w * h * 4);
  let s = seed >>> 0;
  for (let i = 0; i < out.length; i++) {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    out[i] = i % 4 === 3 ? 255 : s >>> 24;
  }
  return out;
}

/** The fit and normalization from the definition: the image is scaled to
 *  fill the input (stretch), fit inside it centered (letterbox) or a
 *  centered square of side min(w, h) * cropFraction fills it (crop). Each
 *  input pixel center samples the image with half-pixel bilinear, edges
 *  clamped; outside the image it is `pad`. Output is CHW, value = byte *
 *  alpha + beta. */
function fitRef(rgba: Uint8Array, w: number, h: number, args: ImageToChwArgs): Float64Array {
  const { mode, cropFraction, outW, outH, alpha, beta, pad } = args;
  let scaleX = outW / w;
  let scaleY = outH / h;
  if (mode === 'letterbox') scaleX = scaleY = Math.min(scaleX, scaleY);
  if (mode === 'crop') {
    const side = Math.min(w, h) * cropFraction;
    scaleX = outW / side;
    scaleY = outH / side;
  }
  // The scaled image is centered on the input in every mode.
  const offsetX = (outW - w * scaleX) / 2;
  const offsetY = (outH - h * scaleY) / 2;
  const at = (x: number, y: number, c: number) =>
    rgba[(Math.min(Math.max(y, 0), h - 1) * w + Math.min(Math.max(x, 0), w - 1)) * 4 + c]!;
  const hw = outW * outH;
  const out = new Float64Array(3 * hw);
  for (let y = 0; y < outH; y++) {
    for (let x = 0; x < outW; x++) {
      const sx = (x + 0.5 - offsetX) / scaleX;
      const sy = (y + 0.5 - offsetY) / scaleY;
      const inside = sx >= 0 && sy >= 0 && sx < w && sy < h;
      const x0 = Math.floor(sx - 0.5);
      const y0 = Math.floor(sy - 0.5);
      const fx = sx - 0.5 - x0;
      const fy = sy - 0.5 - y0;
      for (let c = 0; c < 3; c++) {
        const top = at(x0, y0, c) * (1 - fx) + at(x0 + 1, y0, c) * fx;
        const bot = at(x0, y0 + 1, c) * (1 - fx) + at(x0 + 1, y0 + 1, c) * fx;
        const byte = top * (1 - fy) + bot * fy;
        out[c * hw + y * outW + x] = inside ? byte * alpha[c]! + beta[c]! : pad[c]!;
      }
    }
  }
  return out;
}

/** Uploads RGBA `bytes` as a `format` texture: rgba32float gets byte / 255,
 *  bgra8unorm gets the bytes reordered so the texel still reads as RGBA. */
function sourceTexture(
  bytes: Uint8Array<ArrayBuffer>,
  w: number,
  h: number,
  format: 'rgba8unorm' | 'bgra8unorm' | 'rgba32float',
): GPUTexture {
  const texture = root.device.createTexture({
    size: [w, h],
    format,
    usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST,
  });
  onTestFinished(() => texture.destroy());
  if (format === 'rgba32float') {
    const texels = Float32Array.from(bytes, (b) => b / 255);
    root.device.queue.writeTexture({ texture }, texels, { bytesPerRow: w * 16 }, [w, h]);
    return texture;
  }
  const texels =
    format === 'bgra8unorm'
      ? bytes.map((_, i) => bytes[i - (i % 4) + [2, 1, 0, 3][i % 4]!]!)
      : bytes;
  root.device.queue.writeTexture({ texture }, texels, { bytesPerRow: w * 4 }, [w, h]);
  return texture;
}

async function run(src: GPUTexture | GPUExternalTexture, args: ImageToChwArgs) {
  const out = root.createBuffer(d.arrayOf(d.f32, 3 * args.outW * args.outH)).$usage('storage');
  onTestFinished(() => out.destroy());
  const enc = root.device.createCommandEncoder();
  imageToChw.encode(enc, src, out, args);
  root.device.queue.submit([enc.finish()]);
  return out.read();
}

/** Bytes to [0, 1], no padding color: the plain case most rows use. */
const PLAIN = {
  cropFraction: 1,
  alpha: [1 / 255, 1 / 255, 1 / 255],
  beta: [0, 0, 0],
  pad: [0, 0, 0],
} as const;

describe('imageToChw kernel', () => {
  it.each<{ w: number; h: number; mode: FitMode; outW: number; outH: number; crop?: number }>([
    // Same size: each input pixel center lands on one image pixel, so this
    // pins the orientation and channel order as well as the values.
    { w: 8, h: 6, mode: 'stretch', outW: 8, outH: 6 },
    // Upsampling and downsampling, by a different factor per axis.
    { w: 5, h: 3, mode: 'stretch', outW: 16, outH: 12 },
    { w: 37, h: 29, mode: 'stretch', outW: 16, outH: 8 },
    { w: 1, h: 1, mode: 'stretch', outW: 4, outH: 4 },
    // 17×13 = 221 pixels: not a multiple of the workgroup, so the tail
    // threads must write nothing.
    { w: 10, h: 10, mode: 'stretch', outW: 17, outH: 13 },
    // Letterbox pads above and below a wide image, left and right of a tall one.
    { w: 20, h: 10, mode: 'letterbox', outW: 16, outH: 16 },
    { w: 9, h: 17, mode: 'letterbox', outW: 16, outH: 16 },
    { w: 12, h: 12, mode: 'letterbox', outW: 16, outH: 10 },
    // Crop cuts the sides of a wide image, the top and bottom of a tall one.
    { w: 24, h: 12, mode: 'crop', outW: 16, outH: 16 },
    { w: 12, h: 20, mode: 'crop', outW: 16, outH: 16 },
    { w: 20, h: 20, mode: 'crop', outW: 8, outH: 8, crop: 0.5 },
    // A non-square input stretches the square crop.
    { w: 30, h: 18, mode: 'crop', outW: 16, outH: 8, crop: 0.75 },
  ])('$mode $w×$h to $outW×$outH', async ({ w, h, mode, outW, outH, crop = 1 }) => {
    const bytes = randomRgba(w, h, w * 31 + h);
    const args: ImageToChwArgs = { ...PLAIN, mode, cropFraction: crop, outW, outH };
    const got = await run(sourceTexture(bytes, w, h, 'rgba8unorm'), args);
    expectClose(got, fitRef(bytes, w, h, args), { atol: 1e-5, rtol: 1e-5 });
  });

  it('applies alpha, beta and pad per channel', async () => {
    const bytes = randomRgba(20, 10, 3);
    const args: ImageToChwArgs = {
      mode: 'letterbox',
      cropFraction: 1,
      outW: 16,
      outH: 16,
      alpha: [0.01, -0.02, 0.03],
      beta: [-1, 0.5, 2],
      pad: [7, -8, 9],
    };
    const got = await run(sourceTexture(bytes, 20, 10, 'rgba8unorm'), args);
    expectClose(got, fitRef(bytes, 20, 10, args), { atol: 1e-4, rtol: 1e-5 });
  });

  it.each(['bgra8unorm', 'rgba32float'] as const)(
    'reads a %s texture as RGB bytes',
    async (format) => {
      const bytes = randomRgba(7, 5, 11);
      const args: ImageToChwArgs = { ...PLAIN, mode: 'stretch', outW: 16, outH: 12 };
      const got = await run(sourceTexture(bytes, 7, 5, format), args);
      expectClose(got, fitRef(bytes, 7, 5, args), { atol: 1e-5, rtol: 1e-5 });
    },
  );

  it('reads an external texture', async () => {
    const [w, h] = [12, 8];
    const bytes = randomRgba(w, h, 5);
    const canvas = new OffscreenCanvas(w, h);
    canvas.getContext('2d')!.putImageData(new ImageData(new Uint8ClampedArray(bytes), w, h), 0, 0);
    const frame = new VideoFrame(canvas, { timestamp: 0 });
    onTestFinished(() => frame.close());
    const args: ImageToChwArgs = { ...PLAIN, mode: 'letterbox', outW: 16, outH: 16 };
    const got = await run(root.device.importExternalTexture({ source: frame }), args);
    // The external texture path may go through a YUV or color conversion,
    // so allow one byte of error.
    expectClose(got, fitRef(bytes, w, h, args), { atol: 1.01 / 255, rtol: 0 });
  });

  // Work after the fit reads the image size from here: an external texture
  // has none on the CPU.
  it.each([
    { name: 'a texture', external: false },
    { name: 'an external texture', external: true },
  ])('writes the size of $name', async ({ external }) => {
    const [w, h] = [13, 7];
    const bytes = randomRgba(w, h, 2);
    let src: GPUTexture | GPUExternalTexture = sourceTexture(bytes, w, h, 'rgba8unorm');
    if (external) {
      const canvas = new OffscreenCanvas(w, h);
      canvas
        .getContext('2d')!
        .putImageData(new ImageData(new Uint8ClampedArray(bytes), w, h), 0, 0);
      const frame = new VideoFrame(canvas, { timestamp: 0 });
      onTestFinished(() => frame.close());
      src = root.device.importExternalTexture({ source: frame });
    }
    await run(src, { ...PLAIN, mode: 'stretch', outW: 4, outH: 4 });
    expect(await imageToChw.size.read()).toEqual([w, h]);
  });

  // The kernel is the GPU twin of the CPU preprocessor: both must give the
  // same model input for the same options.
  it.each<{ mode: FitMode; crop: number }>([
    { mode: 'stretch', crop: 1 },
    { mode: 'letterbox', crop: 1 },
    { mode: 'crop', crop: 0.8 },
  ])('matches the CPU preprocessor, $mode', async ({ mode, crop }) => {
    const [w, h] = [23, 14];
    const bytes = randomRgba(w, h, 9);
    const preprocessor = createImagePreprocessor(
      {
        resizeMode: mode,
        cropFraction: crop,
        normalizeOpts: { alpha: [0.017, 0.018, 0.019], beta: [-2.1, -2, -1.8] },
      },
      { width: 16, height: 12 },
    );
    const got = await run(sourceTexture(bytes, w, h, 'rgba8unorm'), preprocessor.gpuArgs);
    const want = preprocessor.process(imageBuffer(bytes, w, h, 'rgba'));
    expectClose(got, want, { atol: 1e-4, rtol: 1e-5 });
  });
});
