/** CPU image preprocessor: any ImageBuffer in, the model's RGB CHW f32 input
 *  out. Resizes (stretch / letterbox / crop), converts the pixel format,
 *  moves channels first and normalizes, all in one pass over the output. */

import { checkImageSize, FORMAT_CHANNELS, type ImageBuffer, type ResizeMode } from '../image.ts';
import type { ScaleBoxOptions } from '../ops/box.ts';
import { resizeTransform, type Size } from '../ops/point.ts';

/** value = byte * alpha + beta, per RGB channel. */
export interface NormalizeOptions {
  /** One number for all channels, or one per RGB channel. Default 1/255. */
  readonly alpha?: number | readonly number[];
  /** One number for all channels, or one per RGB channel. Default 0. */
  readonly beta?: number | readonly number[];
}

/** How pixels are sampled when the image is scaled. */
export type Interpolation = 'nearest' | 'linear';

export interface ImagePreprocessorOptions {
  /** How the image is fitted into the model input. */
  readonly resizeMode: ResizeMode;
  /** `crop` only: how much of the centered square fills the input, measured
   *  from the middle, in (0, 1]. Default 1, the whole square. */
  readonly cropFraction?: number;
  /** Default `linear`. */
  readonly interpolation?: Interpolation;
  readonly normalizeOpts?: NormalizeOptions;
  /** Byte value of letterbox padding, before normalization. Default 114. */
  readonly padValue?: number;
}

export interface ImagePreprocessor {
  readonly width: number;
  readonly height: number;
  /** Image in, `[3, height, width]` RGB f32 out. Pass `out` to reuse one
   *  array across frames. */
  process(input: ImageBuffer, out?: Float32Array): Float32Array;
  /** How to map results computed on the model input back onto `input`. */
  scaleOptions(input: Size): ScaleBoxOptions;
}

/** Index of the R, G and B byte inside one pixel of each format. */
const RGB_INDEX = {
  rgb: [0, 1, 2],
  rgba: [0, 1, 2],
  bgr: [2, 1, 0],
  bgra: [2, 1, 0],
  gray: [0, 0, 0],
} as const;

function perChannel(v: number | readonly number[], name: string): [number, number, number] {
  if (typeof v === 'number') return [v, v, v];
  if (v.length !== 3) {
    throw new Error(`normalize.${name}: expected 3 values, got ${v.length}`);
  }
  return [v[0]!, v[1]!, v[2]!];
}

/** Per-axis sample table: for each output index, the two source indices to
 *  blend, the blend weight, and whether the pixel is inside the image. */
interface Axis {
  i0: Int32Array;
  i1: Int32Array;
  w: Float32Array;
  inside: Uint8Array;
}

function buildAxis(
  outLen: number,
  srcLen: number,
  scale: number,
  offset: number,
  interpolation: Interpolation,
): Axis {
  const i0 = new Int32Array(outLen);
  const i1 = new Int32Array(outLen);
  const w = new Float32Array(outLen);
  const inside = new Uint8Array(outLen);
  for (let d = 0; d < outLen; d++) {
    // Source coordinate of this output pixel's center, in source pixels.
    const s = (d + 0.5 - offset) / scale;
    if (s < 0 || s >= srcLen) continue;
    inside[d] = 1;
    if (interpolation === 'nearest') {
      i0[d] = i1[d] = Math.min(srcLen - 1, Math.floor(s));
      continue;
    }
    const f = s - 0.5;
    const a = Math.floor(f);
    i0[d] = Math.max(0, a);
    i1[d] = Math.min(srcLen - 1, a + 1);
    w[d] = a < 0 ? 0 : f - a;
  }
  return { i0, i1, w, inside };
}

export function createImagePreprocessor(
  options: ImagePreprocessorOptions,
  size: Size,
): ImagePreprocessor {
  const { width, height, resizeMode } = { ...size, resizeMode: options.resizeMode };
  checkImageSize(width, height, 'input size');
  const interpolation = options.interpolation ?? 'linear';
  const cropFraction = options.cropFraction ?? 1;
  if (!(cropFraction > 0 && cropFraction <= 1)) {
    throw new Error(`cropFraction: expected a number in (0, 1], got ${cropFraction}`);
  }
  const padValue = options.padValue ?? 114;
  const alpha = perChannel(options.normalizeOpts?.alpha ?? 1 / 255, 'alpha');
  const beta = perChannel(options.normalizeOpts?.beta ?? 0, 'beta');
  const hw = width * height;

  const scaleOptions = (input: Size): ScaleBoxOptions => ({
    from: { width, height },
    to: { width: input.width, height: input.height },
    resizeMode,
    cropFraction,
  });

  const process = (input: ImageBuffer, out?: Float32Array): Float32Array => {
    out ??= new Float32Array(3 * hw);
    if (out.length !== 3 * hw) {
      throw new Error(`out: ${out.length} floats, expected ${3 * hw}`);
    }
    checkImageSize(input.width, input.height);
    const ch = FORMAT_CHANNELS[input.format];
    const { data } = input;
    if (data.length !== input.width * input.height * ch) {
      throw new Error(
        `image: ${data.length} bytes for ${input.width}x${input.height} ${input.format}`,
      );
    }
    const [ri, gi, bi] = RGB_INDEX[input.format];
    const [ar, ag, ab] = alpha;
    const [br, bg, bb] = beta;
    const t = resizeTransform(scaleOptions(input));

    // Same size, nothing moved: plain byte to float conversion.
    if (t.scaleX === 1 && t.scaleY === 1 && t.offsetX === 0 && t.offsetY === 0) {
      for (let i = 0; i < hw; i++) {
        const p = i * ch;
        out[i] = data[p + ri]! * ar + br;
        out[hw + i] = data[p + gi]! * ag + bg;
        out[2 * hw + i] = data[p + bi]! * ab + bb;
      }
      return out;
    }

    const xs = buildAxis(width, input.width, t.scaleX, t.offsetX, interpolation);
    const ys = buildAxis(height, input.height, t.scaleY, t.offsetY, interpolation);
    const padR = padValue * ar + br;
    const padG = padValue * ag + bg;
    const padB = padValue * ab + bb;
    const rowStride = input.width * ch;
    for (let y = 0; y < height; y++) {
      const rowIn = ys.inside[y]!;
      const r0 = ys.i0[y]! * rowStride;
      const r1 = ys.i1[y]! * rowStride;
      const wy = ys.w[y]!;
      for (let x = 0; x < width; x++) {
        const o = y * width + x;
        if (!rowIn || !xs.inside[x]) {
          out[o] = padR;
          out[hw + o] = padG;
          out[2 * hw + o] = padB;
          continue;
        }
        const c0 = xs.i0[x]! * ch;
        const c1 = xs.i1[x]! * ch;
        const wx = xs.w[x]!;
        const w00 = (1 - wy) * (1 - wx);
        const w01 = (1 - wy) * wx;
        const w10 = wy * (1 - wx);
        const w11 = wy * wx;
        const r =
          data[r0 + c0 + ri]! * w00 +
          data[r0 + c1 + ri]! * w01 +
          data[r1 + c0 + ri]! * w10 +
          data[r1 + c1 + ri]! * w11;
        const g =
          data[r0 + c0 + gi]! * w00 +
          data[r0 + c1 + gi]! * w01 +
          data[r1 + c0 + gi]! * w10 +
          data[r1 + c1 + gi]! * w11;
        const b =
          data[r0 + c0 + bi]! * w00 +
          data[r0 + c1 + bi]! * w01 +
          data[r1 + c0 + bi]! * w10 +
          data[r1 + c1 + bi]! * w11;
        out[o] = r * ar + br;
        out[hw + o] = g * ag + bg;
        out[2 * hw + o] = b * ab + bb;
      }
    }
    return out;
  };

  return { width, height, process, scaleOptions };
}
