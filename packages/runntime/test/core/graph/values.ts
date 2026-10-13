/** Stand-in inputs for op tests. Ops only read their inputs' shapes, so an
 *  input is a pending Value with no op behind it, and nothing is ever
 *  evaluated. */
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

/** An hwc4-stored f16 image input, [C, H, W]: channels padded up to a
 *  multiple of 4. Written out rather than taken from hwc4Meta, which the ops
 *  use, so a bug there fails here. */
export function hwc4Input(c: number, h: number, w: number): Value {
  const elems = h * w * Math.ceil(c / 4) * 4;
  return pending({ elems, dtype: 'f16', dims: [c, h, w], layout: 'hwc4' }, 'toHwc4', []);
}

/** A quantized weight of logical [rows, cols]: `bits`-wide values packed into
 *  u32 words. */
export function quantWeight(rows: number, cols: number, bits: 8 | 4): Value {
  const elems = (rows * cols) / (32 / bits);
  return pending({ elems, dtype: 'quantW', dims: [rows, cols] }, 'reshape', []);
}

/** A caller-owned input that is already on the GPU. Only writeRows checks for
 *  one, and it never touches the buffer, so a stub stands in for it. */
export function materializedInput(dims: number[], dtype: EagerDtype = 'f32'): Value {
  return materialized(meta(dims, dtype), {} as GpuBufferRef);
}
