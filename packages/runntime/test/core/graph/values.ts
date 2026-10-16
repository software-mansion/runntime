/** Stand-in inputs for graph tests, with no GPU. Op tests use pending Values
 *  with no op behind them; eval tests use materialized ones on stub buffers. */
import {
  materialized,
  pending,
  type EagerDtype,
  type GpuBufferRef,
  type Value,
  type ValueMeta,
} from '../../../src/core/graph/value.ts';

/** The meta an op should produce for a row-major tensor of `dims`. */
export function meta(dims: number[], dtype: EagerDtype = 'f32'): ValueMeta {
  return { elems: dims.reduce((a, b) => a * b, 1), dtype, dims };
}

/** A row-major input of the given shape. */
export function input(dims: number[], dtype: EagerDtype = 'f32'): Value {
  return pending(meta(dims, dtype), 'reshape', []);
}

/** The meta of an hwc4-stored f16 image, [C, H, W]: channels padded up to a
 *  multiple of 4. Written out rather than taken from hwc4Meta, which the ops
 *  use, so a bug there fails here. */
function hwc4ImageMeta(c: number, h: number, w: number): ValueMeta {
  const elems = h * w * Math.ceil(c / 4) * 4;
  return { elems, dtype: 'f16', dims: [c, h, w], layout: 'hwc4' };
}

/** An hwc4-stored f16 image input, [C, H, W]. */
export function hwc4Input(c: number, h: number, w: number): Value {
  return pending(hwc4ImageMeta(c, h, w), 'toHwc4', []);
}

/** An hwc4-stored f16 image, [C, H, W], already on the GPU in a stub buffer. */
export function materializedHwc4Input(c: number, h: number, w: number): Value {
  return materialized(hwc4ImageMeta(c, h, w), {} as GpuBufferRef);
}

/** A quantized weight of logical [rows, cols]: `bits`-wide values packed into
 *  u32 words. */
export function quantWeight(rows: number, cols: number, bits: 8 | 4): Value {
  const elems = (rows * cols) / (32 / bits);
  return pending({ elems, dtype: 'quantW', dims: [rows, cols] }, 'reshape', []);
}

/** A caller-owned input that is already on the GPU. Nothing reads the buffer,
 *  only its identity matters, so a stub stands in for it. */
export function materializedInput(dims: number[], dtype: EagerDtype = 'f32'): Value {
  return materialized(meta(dims, dtype), {} as GpuBufferRef);
}
