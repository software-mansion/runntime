/** Raw CPU image types shared by every vision task. */

import { RunntimeError } from '../../../core/index.ts';

/** Pixel formats an ImageBuffer can carry: which channels, in what order. */
export type ImageFormat = 'rgb' | 'rgba' | 'bgr' | 'bgra' | 'gray';

/** Bytes per pixel for each format. */
export const FORMAT_CHANNELS: Record<ImageFormat, number> = {
  rgb: 3,
  rgba: 4,
  bgr: 3,
  bgra: 4,
  gray: 1,
};

/** A raw image in memory: one row after another, all channels of one pixel
 *  together (the way a canvas or a camera gives it). */
export interface ImageBuffer {
  readonly data: Uint8Array | Uint8ClampedArray;
  readonly width: number;
  readonly height: number;
  readonly format: ImageFormat;
  readonly layout: 'hwc';
}

/** How an image is fitted into the model's input size.
 *  - `stretch`: scale width and height separately; nothing is cut or padded.
 *  - `letterbox`: keep the aspect ratio, pad the rest with a flat color.
 *  - `crop`: keep the aspect ratio, cut the centered square, then scale. */
export type ResizeMode = 'stretch' | 'letterbox' | 'crop';

/** Throws unless `width` and `height` are positive integers. */
export function checkImageSize(width: number, height: number, what = 'image'): void {
  if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
    throw new RunntimeError(
      'INVALID_ARGUMENT',
      `${what}: ${width}x${height} is not a positive size`,
    );
  }
}

/** Wraps raw bytes as an ImageBuffer. Checks the size and the byte count. */
export function imageBuffer(
  data: Uint8Array | Uint8ClampedArray,
  width: number,
  height: number,
  format: ImageFormat = 'rgba',
): ImageBuffer {
  checkImageSize(width, height);
  const expected = width * height * FORMAT_CHANNELS[format];
  if (data.length !== expected) {
    throw new RunntimeError(
      'INVALID_ARGUMENT',
      `image: ${data.length} bytes for ${width}x${height} ${format}, expected ${expected}`,
    );
  }
  return { data, width, height, format, layout: 'hwc' };
}

/** Views canvas ImageData as an ImageBuffer. No copy. */
export function imageBufferFromImageData(img: ImageData): ImageBuffer {
  return { data: img.data, width: img.width, height: img.height, format: 'rgba', layout: 'hwc' };
}
