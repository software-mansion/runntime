/** HWC4 boundary conversions. HWC4 is internal storage: models enter it once
 *  after upload, run the conv-shaped graph inside it, and leave before anything
 *  row-major. Both ops fold the dtype change into the same dispatch. */

import { hwc4Meta, pending, Value } from '../value.ts';
import { wantHwc4 } from './shared.ts';

export function toHwc4(x: Value): Value {
  const dims = x.shape.dims;
  const dtype = x.shape.dtype;
  if (x.shape.layout !== undefined) {
    throw new Error(`toHwc4: input is already stored '${x.shape.layout}'`);
  }
  if (!dims || dims.length !== 3 || (dtype !== 'f32' && dtype !== 'f16')) {
    throw new Error(
      `toHwc4: expected a 3D float CHW tensor, got dims=${JSON.stringify(dims)} dtype=${dtype}`,
    );
  }
  const [c, h, w] = dims as [number, number, number];
  return pending(hwc4Meta(c, h, w), 'toHwc4', [x]);
}

/** Per-channel affine in hwc4 storage: out = x·scale (+ shift), with f32 [C]
 *  tables. Running it in-layout replaces a reshape and column-broadcast round
 *  trip. */
export function channelAffine(x: Value, scale: Value, shift?: Value): Value {
  const [c, h, w] = wantHwc4('channelAffine', x);
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
