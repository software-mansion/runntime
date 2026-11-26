/** HWC4 boundary conversions. HWC4 is internal storage: models enter it once
 *  after upload, run the conv-shaped graph inside it, and leave before anything
 *  row-major. The conversions fold the dtype change into the same dispatch. */

import { RunntimeError } from '../../error.ts';
import { hwc4Meta, pending, Value } from '../value.ts';
import { hwc4Dtype, wantHwc4, wantHwc4F16 } from './shared.ts';

/** A row-major [C, H, W] float tensor as an hwc4 image stored as `dtype`,
 *  f16 by default. The channels pad with zeros to a multiple of 4. */
export function toHwc4(x: Value, dtype: 'f32' | 'f16' = 'f16'): Value {
  const dims = x.shape.dims;
  const from = x.shape.dtype;
  if (x.shape.layout !== undefined) {
    throw new Error(`toHwc4: input is already stored '${x.shape.layout}'`);
  }
  if (!dims || dims.length !== 3 || (from !== 'f32' && from !== 'f16')) {
    throw new Error(
      `toHwc4: expected a 3D float CHW tensor, got dims=${JSON.stringify(dims)} dtype=${from}`,
    );
  }
  const [c, h, w] = dims as [number, number, number];
  return pending(hwc4Meta(c, h, w, dtype), 'toHwc4', [x]);
}

/** Per-channel affine in hwc4 storage: out = x·scale (+ shift), with f32 [C]
 *  tables. Running it in-layout replaces a reshape and column-broadcast round
 *  trip. Throws INVALID_ARGUMENT for an f32 image. */
export function channelAffine(x: Value, scale: Value, shift?: Value): Value {
  const [c, h, w] = wantHwc4F16('channelAffine', x);
  for (const [name, t] of [
    ['scale', scale],
    ['shift', shift],
  ] as const) {
    if (t && (t.shape.dtype !== 'f32' || t.shape.elems !== c)) {
      throw new Error(
        `channelAffine: ${name} must be f32 with ${c} elems, got ${t.shape.elems} ${t.shape.dtype}`,
      );
    }
  }
  const inputs = shift ? [x, scale, shift] : [x, scale];
  return pending(hwc4Meta(c, h, w), 'channelAffineHwc4', inputs, undefined, [shift ? 1 : 0]);
}

/** Padding lanes are dropped. */
export function toChw(x: Value, dtype: 'f32' | 'f16'): Value {
  const [c, h, w] = wantHwc4('toChw', x);
  return pending({ elems: c * h * w, dtype, dims: [c, h, w] }, 'toChw', [x]);
}

/** An hwc4 image as row-major [H, W, C], in its own dtype: torch's
 *  permute(1, 2, 0) of the [C, H, W] image. Throws INVALID_ARGUMENT unless C
 *  is a multiple of 4, since the padding lanes are not dropped. */
export function toHwc(x: Value): Value {
  const [c, h, w] = wantHwc4('toHwc', x);
  if (c % 4 !== 0) {
    throw new RunntimeError(
      'INVALID_ARGUMENT',
      `toHwc: channels must be a multiple of 4, got ${c}; toChw() takes any count`,
    );
  }
  return pending({ elems: h * w * c, dtype: hwc4Dtype(x), dims: [h, w, c] }, 'reshape', [x]);
}
