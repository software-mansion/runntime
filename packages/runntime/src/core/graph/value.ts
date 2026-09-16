/** The eager core's array type: either a materialized GPU buffer or a pending
 *  op describing how to compute one. */

import type { F16Buffer, F32Buffer, U32Buffer } from '../kernels/common.ts';
import type { MatmulOpts } from './ops.ts';

export type EagerDtype = 'f32' | 'f16' | 'quantW';

/** Bytes per element of `dtype`; quantW counts u32 words, not values. The one
 *  place buffer sizes and readback counts are derived. */
export function elemBytes(dtype: EagerDtype): number {
  return dtype === 'f16' ? 2 : 4;
}

/** Everything known about a Value without computing it. quantW is the
 *  exception: `dims` stay logical while `elems` is a u32 word count. */
export interface ValueMeta {
  /** Element count in units of `dtype`. */
  readonly elems: number;
  readonly dtype: EagerDtype;
  /** Logical shape for non-flat ops, e.g. [rows, cols]. */
  readonly dims?: readonly number[];
  /** Physical storage order, when it differs from row-major. Under 'hwc4'
   *  `elems` includes the channel padding while `dims` stay logical, and only
   *  hwc4-aware ops accept the Value — everything else throws. */
  readonly layout?: 'hwc4';
}

/** Channel blocks of an hwc4 tensor: ceil(c / 4). */
export function c4of(c: number): number {
  return Math.ceil(c / 4);
}

/** Meta for an hwc4-stored [C, H, W] f16 image tensor. */
export function hwc4Meta(c: number, h: number, w: number): ValueMeta {
  return { elems: h * w * c4of(c) * 4, dtype: 'f16', dims: [c, h, w], layout: 'hwc4' };
}

/** Build a row-major f32 matrix shape. */
export function matrix(rows: number, cols: number): ValueMeta {
  return { elems: rows * cols, dtype: 'f32', dims: [rows, cols] };
}

export function tensor3d(c: number, h: number, w: number): ValueMeta {
  const chw = c * h * w;
  return { elems: chw, dtype: 'f32', dims: [c, h, w] };
}

/** The storage buffer behind a materialized Value. */
export type GpuBufferRef = F32Buffer | F16Buffer | U32Buffer;

/** Every op a pending node can name. */
export type OpName =
  // elementwise
  | 'add'
  | 'mul'
  | 'sub'
  | 'swiglu'
  | 'swigluChunk'
  | 'rsqrt'
  | 'addScalar'
  | 'sigmoid'
  | 'mulScalar'
  | 'clampScalar'
  | 'tanh'
  | 'gelu'
  | 'silu'
  | 'asinh'
  // matmul
  | 'matmul'
  | 'matmulQuantW'
  | 'matmulGatherQuantW'
  | 'matmulGather'
  | 'argmaxDot'
  // attention
  | 'attn'
  | 'rope'
  // conv
  | 'conv1d'
  | 'convTranspose2d'
  | 'conv2dHwc4'
  | 'maxPool2dHwc4'
  | 'upsample2dHwc4'
  | 'avgPool2dHwc4'
  | 'pad2dHwc4'
  | 'resizeBilinearHwc4'
  | 'channelAffineHwc4'
  | 'concatChannelsHwc4'
  | 'sliceChannelsHwc4'
  | 'toHwc4'
  | 'toChw'
  // ssm
  | 'ssmScanProject'
  | 'ssmSelectiveScan'
  | 'ssmScanMerge'
  // reduce
  | 'softmax'
  | 'mean'
  | 'meanSquare'
  | 'layerNorm'
  | 'topk'
  // shape
  | 'astype'
  | 'transpose'
  | 'sliceCols'
  | 'concatCols'
  | 'sliceRows'
  | 'concatRows'
  | 'reshape'
  | 'sliceChannels'
  | 'concatChannels'
  | 'writeRows'
  | 'gatherRowsFrom'
  | 'gatherRows';

/** The recipe for a not-yet-computed Value. Inputs may themselves be pending,
 *  so chained ops form the DAG eval() walks. */
export interface PendingOp {
  readonly op: OpName;
  readonly inputs: readonly Value[];
  /** Scalar operand for ops like addScalar. */
  readonly scalar?: number;
  /** Numeric attributes for layout and scalar ops, e.g. [start, end]. */
  readonly attrs?: readonly number[];
}

/** A tensor as the eager API sees it, and one node of a lazy DAG. Calling ops
 *  only links nodes; reading a result evaluates everything it depends on in one
 *  submit. */
export class Value {
  private constructor(
    readonly shape: ValueMeta,
    /** Undefined until eval() runs the op, which is what pending means. */
    private _buffer: GpuBufferRef | undefined,
    /** Set when this Value came from an op. Stays set after eval but is inert:
     *  `state` depends on `_buffer` alone. */
    readonly pending: PendingOp | undefined,
  ) {}

  /** Why the buffer is gone. `_buffer` keeps the stale reference on purpose —
   *  clearing it would flip state back to pending and rerun the recipe. */
  private _released: string | undefined;

  get state(): 'materialized' | 'pending' | 'released' {
    if (this._released !== undefined) return 'released';
    return this._buffer !== undefined ? 'materialized' : 'pending';
  }

  /** Why the buffer is gone; undefined while it is still there. */
  get releasedReason(): string | undefined {
    return this._released;
  }

  /** Throws if not yet materialized, or already gone. */
  get buffer(): GpuBufferRef {
    if (this._released !== undefined) {
      throw new Error(`Value.buffer: ${this._released}`);
    }
    if (this._buffer === undefined) {
      throw new Error('Value.buffer: not materialized yet — call eval() first');
    }
    return this._buffer;
  }

  /** @internal */
  setMaterialized(buffer: GpuBufferRef): void {
    this._buffer = buffer;
  }

  /** @internal Every later use of the Value throws `reason`. */
  markReleased(
    reason = 'buffer was recycled by eval — it must have been a TARGET of the eval that materialized it (evalValues([primary, v])) to stay alive across evals',
  ): void {
    this._released = reason;
  }

  static materialized(shape: ValueMeta, buffer: GpuBufferRef): Value {
    return new Value(shape, buffer, undefined);
  }

  static pending(
    shape: ValueMeta,
    op: OpName,
    inputs: readonly Value[],
    scalar?: number,
    attrs?: readonly number[],
  ): Value {
    return new Value(shape, undefined, { op, inputs, scalar, attrs });
  }
}

/** Method forms of the ops, implemented in methods.ts. Declared here so the
 *  published .d.ts carries them on Value itself rather than as an
 *  augmentation of a relative path that no longer exists once bundled.
 *  The import is type-only, so value.ts still never imports ops.ts at
 *  runtime. */
export interface Value {
  add(b: Value | number): Value;
  sub(b: Value | number): Value;
  mul(b: Value | number): Value;
  matmul(b: Value, opts?: MatmulOpts): Value;
  reshape(newDims: number[]): Value;
  transpose(): Value;
  clamp(lo: number, hi: number): Value;
  slice(dim: number, start: number, end: number): Value;
  chunk(n: number, dim: number): Value[];
  split(sizes: readonly number[], dim: number): Value[];
  softmax(): Value;
  sigmoid(): Value;
  tanh(): Value;
  asinh(): Value;
  mean(): Value;
  rsqrt(): Value;
  topk(k: number): Value;
}

export const materialized = Value.materialized;
export const pending = Value.pending;
