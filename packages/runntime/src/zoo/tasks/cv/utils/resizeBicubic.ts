/** Antialiased bicubic resize on the CPU. Downscaling widens the kernel by
 *  the scale, so every source pixel counts and fine detail does not
 *  alias. */

import { checkImageBuffer, checkImageSize, FORMAT_CHANNELS, type ImageBuffer } from '../image.ts';
import { RGB_INDEX } from './imagePreprocessor.ts';

/** The a of Keys' cubic convolution kernel. */
const CUBIC_A = -0.5;

function cubic(x: number): number {
  const t = Math.abs(x);
  if (t < 1) return ((CUBIC_A + 2) * t - (CUBIC_A + 3)) * t * t + 1;
  if (t < 2) return (((t - 5) * t + 8) * t - 4) * CUBIC_A;
  return 0;
}

/** For each output index, the first source index and the normalized
 *  weights of the source pixels it blends, `taps` slots per output. */
interface Axis {
  start: Int32Array;
  count: Int32Array;
  weights: Float32Array;
  taps: number;
}

function buildAxis(outLen: number, srcLen: number): Axis {
  const scale = srcLen / outLen;
  // The kernel spans 2 pixels each side, widened by the scale when shrinking.
  const support = 2 * Math.max(scale, 1);
  const inv = 1 / Math.max(scale, 1);
  const taps = Math.ceil(support) * 2 + 1;
  const start = new Int32Array(outLen);
  const count = new Int32Array(outLen);
  const weights = new Float32Array(outLen * taps);
  for (let i = 0; i < outLen; i++) {
    const center = scale * (i + 0.5);
    const lo = Math.max(Math.trunc(center - support + 0.5), 0);
    const n = Math.min(Math.trunc(center + support + 0.5), srcLen) - lo;
    let total = 0;
    for (let j = 0; j < n; j++) {
      const w = cubic((j + lo - center + 0.5) * inv);
      weights[i * taps + j] = w;
      total += w;
    }
    if (total !== 0) for (let j = 0; j < n; j++) weights[i * taps + j]! /= total;
    start[i] = lo;
    count[i] = n;
  }
  return { start, count, weights, taps };
}

/** `image` resized to width × height: RGB [3, height, width] of whole
 *  values in 0..255. Width resizes first, then height, rounded to bytes in
 *  between. Throws INVALID_ARGUMENT for a bad size or byte count. */
export function resizeBicubic(image: ImageBuffer, width: number, height: number): Float32Array {
  checkImageSize(width, height, 'resize');
  checkImageBuffer(image);
  const ch = FORMAT_CHANNELS[image.format];
  const { data } = image;
  const rgb = RGB_INDEX[image.format];
  const srcW = image.width;
  const srcH = image.height;
  const xs = buildAxis(width, srcW);
  const ys = buildAxis(height, srcH);

  // Horizontal pass: [3, srcH, width], the three channels of a pixel at
  // once. The clamped array rounds and clamps.
  const plane = srcH * width;
  const tmp = new Uint8ClampedArray(3 * plane);
  const [ri, gi, bi] = rgb;
  for (let y = 0; y < srcH; y++) {
    for (let x = 0; x < width; x++) {
      const n = xs.count[x]!;
      const w0 = x * xs.taps;
      let p = (y * srcW + xs.start[x]!) * ch;
      let r = 0;
      let g = 0;
      let b = 0;
      for (let j = 0; j < n; j++, p += ch) {
        const w = xs.weights[w0 + j]!;
        r += data[p + ri]! * w;
        g += data[p + gi]! * w;
        b += data[p + bi]! * w;
      }
      const o = y * width + x;
      tmp[o] = r;
      tmp[plane + o] = g;
      tmp[2 * plane + o] = b;
    }
  }
  // Vertical pass, a whole output row at once, rounded and clamped like a
  // uint8 image.
  const out = new Float32Array(3 * height * width);
  const acc = new Float64Array(width);
  for (let c = 0; c < 3; c++) {
    for (let y = 0; y < height; y++) {
      acc.fill(0);
      const w0 = y * ys.taps;
      for (let j = 0; j < ys.count[y]!; j++) {
        const w = ys.weights[w0 + j]!;
        const src = (c * srcH + ys.start[y]! + j) * width;
        for (let x = 0; x < width; x++) acc[x]! += tmp[src + x]! * w;
      }
      const dst = (c * height + y) * width;
      for (let x = 0; x < width; x++) {
        out[dst + x] = Math.min(255, Math.max(0, Math.round(acc[x]!)));
      }
    }
  }
  return out;
}
