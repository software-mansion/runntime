/** Weightless spatial ops on HWC4-stored image tensors: max pooling and
 *  nearest-neighbor upsampling. Row-major inputs must enter via toHwc4 like
 *  the convs they surround. */

import { hwc4Meta, pending, Value } from '../../value.ts';
import { wantHwc4 } from '../shared.ts';

/** 2D max pooling: x=[C,H,W] → [C, H_out, W_out] (torch: max_pool2d).
 *  attrs: [kernelSize, stride, padding] */
export function maxPool2d(
  x: Value,
  opts: { kernelSize: number; stride: number; padding: number },
): Value {
  const [c, h, w] = wantHwc4('maxPool2d', x);
  const { kernelSize, stride, padding } = opts;
  if (kernelSize < 1 || stride < 1 || padding < 0 || padding > kernelSize / 2) {
    throw new Error(
      `maxPool2d: bad geometry k=${kernelSize} stride=${stride} padding=${padding} (torch requires padding <= k/2)`,
    );
  }
  const hOut = Math.floor((h + 2 * padding - kernelSize) / stride) + 1;
  const wOut = Math.floor((w + 2 * padding - kernelSize) / stride) + 1;
  if (hOut < 1 || wOut < 1) {
    throw new Error(`maxPool2d: window k=${kernelSize} larger than padded input ${h}×${w}`);
  }
  return pending(hwc4Meta(c, hOut, wOut), 'maxPool2dHwc4', [x], undefined, [
    kernelSize,
    stride,
    padding,
  ]);
}

/** Nearest-neighbor upsample: x=[C,H,W] → [C, H·scale, W·scale]. Integer scale
 *  only (torch: interpolate(mode='nearest')). attrs: [scale] */
export function upsample2d(x: Value, opts: { scale: number }): Value {
  const [c, h, w] = wantHwc4('upsample2d', x);
  const { scale } = opts;
  if (!Number.isInteger(scale) || scale < 1) {
    throw new Error(`upsample2d: scale must be a positive integer, got ${scale}`);
  }
  return pending(hwc4Meta(c, h * scale, w * scale), 'upsample2dHwc4', [x], undefined, [scale]);
}

/** Average pool, no-padding form: x=[C,H,W] → [C,H_out,W_out] with the k²
 *  divisor (torch AvgPool2d(k, s), padding 0). attrs: [kernelSize, stride] */
export function avgPool2d(x: Value, opts: { kernelSize: number; stride?: number }): Value {
  const [c, h, w] = wantHwc4('avgPool2d', x);
  const { kernelSize, stride = opts.kernelSize } = opts;
  if (kernelSize < 1 || stride < 1) {
    throw new Error(`avgPool2d: bad geometry k=${kernelSize} stride=${stride}`);
  }
  const hOut = Math.floor((h - kernelSize) / stride) + 1;
  const wOut = Math.floor((w - kernelSize) / stride) + 1;
  if (hOut < 1 || wOut < 1) {
    throw new Error(`avgPool2d: window k=${kernelSize} larger than input ${h}×${w}`);
  }
  return pending(hwc4Meta(c, hOut, wOut), 'avgPool2dHwc4', [x], undefined, [kernelSize, stride]);
}

/** Zero-pad each side of the map: [C,H,W] → [C, H+2·padH, W+2·padW] — the
 *  companion of padding-0 convs when the two axes need different padding.
 *  attrs: [padH, padW] */
export function pad2d(x: Value, opts: { padH: number; padW: number }): Value {
  const [c, h, w] = wantHwc4('pad2d', x);
  const { padH, padW } = opts;
  if (!Number.isInteger(padH) || !Number.isInteger(padW) || padH < 0 || padW < 0) {
    throw new Error(`pad2d: pads must be non-negative integers, got (${padH}, ${padW})`);
  }
  return pending(hwc4Meta(c, h + 2 * padH, w + 2 * padW), 'pad2dHwc4', [x], undefined, [
    padH,
    padW,
  ]);
}

/** Bilinear resize: x=[C,H,W] → [C,outH,outW]. mode 'alignCorners' (default —
 *  torch interpolate(align_corners=True), the DPT-head form) or 'halfPixel'
 *  (ONNX/torch default coordinates — DepthART's upsampling form). Arbitrary
 *  output sizes, up or down. attrs: [outH, outW, alignCorners] */
export function resizeBilinear2d(
  x: Value,
  opts: { outH: number; outW: number; mode?: 'alignCorners' | 'halfPixel' },
): Value {
  const [c] = wantHwc4('resizeBilinear2d', x);
  const { outH, outW, mode = 'alignCorners' } = opts;
  if (!Number.isInteger(outH) || !Number.isInteger(outW) || outH < 1 || outW < 1) {
    throw new Error(`resizeBilinear2d: bad output size ${outH}×${outW}`);
  }
  return pending(hwc4Meta(c, outH, outW), 'resizeBilinearHwc4', [x], undefined, [
    outH,
    outW,
    mode === 'alignCorners' ? 1 : 0,
  ]);
}
