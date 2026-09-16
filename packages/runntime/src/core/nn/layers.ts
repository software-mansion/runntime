/** Stock nn layers: each holds Parameters and calls the matching op. */

import { groupNorm, linear, rmsNorm } from '../graph/compose.ts';
import {
  type ConvAct,
  type SlotAct,
  conv1d,
  conv2d,
  convTranspose2d,
  gatherRows,
  gatherRowsFrom,
  layerNorm,
  sdpa,
  sdpaPacked,
} from '../graph/ops.ts';
import { matrix, type Value } from '../graph/value.ts';
import {
  convToTapMajor,
  packConvHwc4F16,
  packDwHwc4F16,
  permuteConvTransposeF16,
  transposeF32,
} from '../weights/convert.ts';
import { foldBnIntoConv } from '../weights/foldBn.ts';
import { derivedTensor, type LazyStateDict } from '../weights/safetensors.ts';
import { tensor } from '../gpu/buffers.ts';
import { CpuParameter, Module, Parameter, type AnyModule } from './module.ts';

/** Token-id lookup table: ids to rows of a [numEmbeddings, dims] weight.
 *
 *  On 'gpu' the weight doubles as a tied LM head, read through `weight.value`
 *  in its raw orientation. On 'cpu' it never reaches the GPU — forward()
 *  gathers the requested rows and uploads those — and tying is unavailable. */
export class Embedding extends Module<[readonly number[] | Value], Value> {
  readonly weight: Parameter;
  constructor(
    readonly numEmbeddings: number,
    readonly embeddingDim: number,
    opts: { device?: 'gpu' | 'cpu' } = {},
  ) {
    super();
    this.weight =
      opts.device === 'cpu'
        ? new CpuParameter({
            elems: numEmbeddings * embeddingDim,
            dtype: 'f32', // FIXME: this is heresy, the underlying data can be other type and this is just so that the loader counts bytes properly
            dims: [numEmbeddings, embeddingDim],
          })
        : new Parameter(matrix(numEmbeddings, embeddingDim));
  }
  /** CPU gather of the requested rows. cpu tables only. */
  gather(tokenIds: readonly number[]): Float32Array {
    if (!(this.weight instanceof CpuParameter)) {
      throw new Error('Embedding.gather: cpu-resident tables only');
    }
    const { numEmbeddings: vocab, embeddingDim: dim } = this;
    const table = this.weight.data;
    const out = new Float32Array(tokenIds.length * dim);
    const u32 = new Uint32Array(out.buffer);
    tokenIds.forEach((id, t) => {
      if (!Number.isInteger(id) || id < 0 || id >= vocab) {
        throw new Error(`Embedding: token id ${id} out of range (vocab ${vocab})`);
      }
      const base = id * dim;
      if (table instanceof Float32Array) {
        out.set(table.subarray(base, base + dim), t * dim);
      } else {
        for (let i = 0; i < dim; i++) u32[t * dim + i] = table[base + i]! << 16;
      }
    });
    return out;
  }
  /** Token ids to embedded rows [t, dims], always f32. A GPU-resident id
   *  vector lets a captured graph re-run on new tokens; cpu tables take the
   *  array form only. */
  forward(ids: readonly number[] | Value): Value {
    if (Array.isArray(ids)) {
      return this.weight instanceof CpuParameter
        ? tensor(this.gather(ids), {
            elems: ids.length * this.embeddingDim,
            dtype: this.weight.shape.dtype,
            dims: [ids.length, this.embeddingDim],
          })
        : gatherRows(this.weight.value, ids as readonly number[]);
    }
    if (this.weight instanceof CpuParameter) {
      throw new Error('Embedding: GPU-resident ids need a gpu table');
    }
    return gatherRowsFrom(this.weight.value, ids as Value);
  }
}

/** y = x @ weight + bias. weight is stored [in, out]; torch checkpoints
 *  transpose once at load.
 *
 *  `addend` and `activation` fold a residual add and a following tanh or GELU
 *  into the matmul epilogue, as act(x·W + bias) + addend. */
export class Linear extends Module {
  readonly weight: Parameter;
  readonly bias?: Parameter;
  readonly activation?: SlotAct;
  constructor(
    readonly inFeatures: number,
    readonly outFeatures: number,
    opts: { bias?: boolean; activation?: SlotAct } = {},
  ) {
    super();
    this.weight = new Parameter(matrix(inFeatures, outFeatures));
    this.activation = opts.activation;
    this.bias =
      opts.bias === false
        ? undefined
        : new Parameter({ elems: outFeatures, dtype: 'f32', dims: [outFeatures] });
  }
  override transformStateDict(sd: LazyStateDict, prefix: string): void {
    const name = prefix ? `${prefix}.weight` : 'weight';
    const tensor = sd.tensors.get(name);
    if (!tensor) return; // missing keys are the strict check's job, not ours
    // Without this a square wrong-layout weight loads silently transposed.
    if (
      tensor.shape.length !== 2 ||
      tensor.shape[0] !== this.outFeatures ||
      tensor.shape[1] !== this.inFeatures
    ) {
      throw new Error(
        `Linear: ${name} has shape ${JSON.stringify(tensor.shape)}, ` +
          `expected torch layout [${this.outFeatures}, ${this.inFeatures}]`,
      );
    }
    sd.tensors.set(
      name,
      derivedTensor(
        [this.inFeatures, this.outFeatures],
        async () => transposeF32(await tensor.f32(), this.outFeatures, this.inFeatures),
        tensor.byteLength,
      ),
    );
  }
  forward(x: Value, call: { addend?: Value } = {}): Value {
    return linear(x, this.weight.value, this.bias?.value, call.addend, this.activation);
  }
}

/** Strided 1D convolution over time-major [T, inChannels]. No batch dim, no
 *  dilation, no groups.
 *
 *  `padding` takes torch's number, 'valid' or 'same'; padLeft and padRight pad
 *  each side independently instead, with kernelSize−1 on the left being causal.
 *  weight is stored tap-major and permutes from torch layout at load. bias and
 *  `activation` fold into the conv epilogue. */
export class Conv1d extends Module {
  readonly weight: Parameter;
  readonly bias?: Parameter;
  readonly stride: number;
  readonly padding?: number | 'valid' | 'same';
  readonly padLeft?: number;
  readonly padRight?: number;
  readonly activation?: SlotAct;
  constructor(
    readonly inChannels: number,
    readonly outChannels: number,
    readonly kernelSize: number,
    opts: {
      stride?: number;
      bias?: boolean;
      padding?: number | 'valid' | 'same';
      padLeft?: number;
      padRight?: number;
      activation?: SlotAct;
    } = {},
  ) {
    super();
    this.stride = opts.stride ?? 1;
    this.padding = opts.padding;
    this.padLeft = opts.padLeft;
    this.padRight = opts.padRight;
    this.activation = opts.activation;
    const rows = kernelSize * inChannels;
    this.weight = new Parameter(matrix(rows, outChannels));
    this.bias =
      opts.bias === false
        ? undefined
        : new Parameter({ elems: outChannels, dtype: 'f32', dims: [outChannels] });
  }
  override transformStateDict(sd: LazyStateDict, prefix: string): void {
    const name = prefix ? `${prefix}.weight` : 'weight';
    const tensor = sd.tensors.get(name);
    if (!tensor) return; // a missing key is reported by the strict key check
    // Without this a weight with equal in/out channels loads scrambled.
    if (
      tensor.shape.length !== 3 ||
      tensor.shape[0] !== this.outChannels ||
      tensor.shape[1] !== this.inChannels ||
      tensor.shape[2] !== this.kernelSize
    ) {
      throw new Error(
        `Conv1d: ${name} has shape ${JSON.stringify(tensor.shape)}, ` +
          `expected torch layout [${this.outChannels}, ${this.inChannels}, ${this.kernelSize}]`,
      );
    }
    sd.tensors.set(
      name,
      derivedTensor(
        [this.kernelSize * this.inChannels, this.outChannels],
        async () =>
          convToTapMajor(await tensor.f32(), this.outChannels, this.inChannels, this.kernelSize),
        tensor.byteLength,
      ),
    );
  }
  forward(x: Value): Value {
    return conv1d(x, this.weight.value, {
      kernelSize: this.kernelSize,
      stride: this.stride,
      padding: this.padding,
      padLeft: this.padLeft,
      padRight: this.padRight,
      bias: this.bias?.value,
      activation: this.activation,
    });
  }
}

/** LayerNorm over the last dim of width `n`, eps 1e-5. Weight-only unless
 *  `bias: true`. */
export class LayerNorm extends Module {
  readonly weight: Parameter;
  readonly bias?: Parameter;
  constructor(
    readonly n: number,
    readonly eps = 1e-5,
    opts: { bias?: boolean } = {},
  ) {
    super();
    this.weight = new Parameter({ elems: n, dtype: 'f32', dims: [n] });
    if (opts.bias) {
      this.bias = new Parameter({ elems: n, dtype: 'f32', dims: [n] });
    }
  }
  forward(x: Value): Value {
    return layerNorm(x, this.weight.value, this.eps, this.bias?.value);
  }
}

/** GroupNorm over [M, numChannels], global statistics, per-channel affine.
 *  numGroups must be 1. */
export class GroupNorm extends Module {
  readonly weight: Parameter;
  readonly bias: Parameter;
  constructor(
    readonly numGroups: number,
    readonly numChannels: number,
    readonly eps = 1e-5,
  ) {
    super();
    if (numGroups !== 1) {
      throw new Error(`GroupNorm: only numGroups = 1 is supported, got ${numGroups}`);
    }
    this.weight = new Parameter({ elems: numChannels, dtype: 'f32', dims: [numChannels] });
    this.bias = new Parameter({ elems: numChannels, dtype: 'f32', dims: [numChannels] });
  }
  forward(x: Value): Value {
    return groupNorm(x, this.numGroups, this.weight.value, this.bias.value, this.eps);
  }
}

/** 2D convolution over [C_in, H, W], matching torch.nn.Conv2d.
 *
 *  Takes and returns hwc4-stored activations, so models enter the layout once
 *  with toHwc4 and leave with toChw. Requires shader-f16. Groups are 1 or fully
 *  depthwise. BatchNorm folds in at load rather than running, so `bias: true`
 *  is normal for BN-fused checkpoints. */
export class Conv2d extends Module {
  readonly weight: Parameter;
  readonly bias?: Parameter;
  constructor(
    readonly inChannels: number,
    readonly outChannels: number,
    readonly opts: {
      kernelSize: number | [number, number];
      stride?: number;
      padding?: number;
      groups?: number;
      bias?: boolean;
      /** Fuses silu, gelu or relu into the conv kernel. */
      activation?: ConvAct;
    },
  ) {
    super();
    const { kernelSize, groups = 1, bias = true } = opts;
    const [kH, kW] = typeof kernelSize === 'number' ? [kernelSize, kernelSize] : kernelSize;
    const dw = groups === inChannels && inChannels === outChannels;
    if (groups !== 1 && !dw) {
      throw new Error(
        `Conv2d: groups=${groups} is neither 1 nor depthwise (cIn=${inChannels}, cOut=${outChannels})`,
      );
    }
    const kk = (inChannels / groups) * kH * kW;
    this.weight = new Parameter(
      // Raw checkpoint size, which is what the loader pre-checks; the padded
      // blob from preprocess sizes the buffer instead.
      { elems: outChannels * kk, dtype: 'f16', dims: [outChannels, kk], layout: 'hwc4' },
      (data) => {
        if (!(data instanceof Uint16Array)) {
          throw new Error('Conv2d: weight must arrive as f16 bits (Uint16Array)');
        }
        return dw
          ? packDwHwc4F16(data, outChannels, kH, kW)
          : packConvHwc4F16(data, outChannels, inChannels, kH, kW);
      },
      true, // pinned: the conv kernels only ever read f16 weights
    );
    // The hwc4 kernels bind nothing else, so conv-only models need no half().
    this.bias = bias
      ? new Parameter({ elems: outChannels, dtype: 'f16', dims: [outChannels] })
      : undefined;
  }
  forward(x: Value): Value {
    const { kernelSize, stride = 1, padding = 0, groups = 1, activation } = this.opts;
    return conv2d(x, this.weight.value, this.bias?.value, {
      kernelSize,
      stride,
      padding,
      groups,
      activation,
    });
  }
}

/** Transposed 2D conv, no-overlap form only (k equals stride). The weight
 *  permutes from torch layout at load. */
export class ConvTranspose2d extends Module {
  readonly weight: Parameter;
  readonly bias?: Parameter;
  constructor(
    readonly inChannels: number,
    readonly outChannels: number,
    readonly opts: { kernelSize: number; stride: number; bias?: boolean },
  ) {
    super();
    const { kernelSize: k, bias = true } = opts;
    this.weight = new Parameter(
      {
        elems: outChannels * k * k * inChannels,
        dtype: 'f16',
        dims: [outChannels * k * k, inChannels],
      },
      (data) =>
        data instanceof Uint16Array
          ? permuteConvTransposeF16(data, inChannels, outChannels, k)
          : data,
      true, // pinned: the conv kernels only ever read f16 weights
    );
    this.bias = bias
      ? new Parameter({ elems: outChannels, dtype: 'f32', dims: [outChannels] })
      : undefined;
  }
  forward(x: Value): Value {
    return convTranspose2d(x, this.weight.value, this.bias?.value, {
      kernelSize: this.opts.kernelSize,
      stride: this.opts.stride,
    });
  }
}

/** torch.nn.BatchNorm2d. Free at runtime: its stats fold into the conv
 *  declared immediately before it while weights load. Pass `foldInto` when
 *  that conv is not the preceding field; loading throws if there is none. */
export class BatchNorm2d extends Module {
  constructor(
    readonly numFeatures: number,
    readonly opts: { eps?: number; foldInto?: string } = {},
  ) {
    super();
  }
  // No Parameters: the fold consumes every bn.* key, leaving nothing to bind.
  override transformStateDict(sd: LazyStateDict, prefix: string, parent?: AnyModule): void {
    const cut = prefix.lastIndexOf('.');
    const sibling = (name: string) => (cut < 0 ? name : `${prefix.slice(0, cut + 1)}${name}`);
    foldBnIntoConv(
      sd,
      sibling(this.opts.foldInto ?? this.precedingConvName(prefix, parent)),
      prefix,
      this.opts.eps ?? 1e-5,
    );
  }
  /** The Conv2d registered immediately before this bn in the parent. */
  private precedingConvName(prefix: string, parent?: AnyModule): string {
    let prev: readonly [string, AnyModule] | undefined;
    for (const entry of parent?.namedChildren() ?? []) {
      if (entry[1] === this) {
        if (prev && prev[1] instanceof Conv2d && prev[1].outChannels === this.numFeatures) {
          return prev[0];
        }
        break;
      }
      prev = entry;
    }
    throw new Error(
      `BatchNorm2d at '${prefix}': cannot infer its conv — needs a Conv2d with ` +
        `${this.numFeatures} out-channels registered immediately before it, ` +
        `or pass { foldInto: '<convField>' }`,
    );
  }
  /** Identity: the fold already moved this bn's math into its conv. */
  forward(x: Value): Value {
    return x;
  }
}

/** RMSNorm over the last dim of width `n`. */
export class RMSNorm extends Module {
  readonly scale: Parameter;
  constructor(
    readonly n: number,
    readonly eps = 1e-6,
  ) {
    super();
    this.scale = new Parameter({ elems: n, dtype: 'f32', dims: [n] });
  }
  forward(x: Value): Value {
    return rmsNorm(x, this.scale.value, this.eps);
  }
}

export interface MultiHeadAttentionOpts {
  qHeads: number;
  kvHeads: number;
  headDim: number;
  /** Key j is visible to a query at qPos when
   *  qPos − windowLeft ≤ j ≤ qPos + windowRight. Infinity is unbounded. */
  windowLeft: number;
  windowRight: number;
  /** Registers a per-q-head sink-logit Parameter named `sinks`. */
  sinks?: boolean;
}

/** Banded, causal or cross multi-head attention over packed
 *  [T, heads·headDim] slabs.
 *
 *  Takes already-projected q/k/v — models own the projections — and applies
 *  sm_scale = 1, so fold any score scale upstream. */
export class MultiHeadAttention extends Module<
  [Value, Value, Value, { qPosOffset?: number; segments?: Value; maxSegment?: number }?],
  Value
> {
  readonly sinks?: Parameter;
  constructor(private readonly opts: MultiHeadAttentionOpts) {
    super();
    this.sinks = opts.sinks
      ? new Parameter(
          { elems: opts.qHeads, dtype: 'f32', dims: [opts.qHeads] },
          undefined,
          true, // pinned: attention sinks are f32 in every kernel variant
        )
      : undefined;
  }
  forward(
    q: Value,
    k: Value,
    v: Value,
    call: { qPosOffset?: number; segments?: Value; maxSegment?: number } = {},
  ): Value {
    const { qHeads, kvHeads, headDim, windowLeft, windowRight } = this.opts;
    return sdpa(q, k, v, {
      qHeads,
      kvHeads,
      headDim,
      windowLeft,
      windowRight,
      qPosOffset: call.qPosOffset,
      sinks: this.sinks?.value,
      segments: call.segments,
      maxSegment: call.maxSegment,
    });
  }
  /** Self-attention over one fused qkv tensor read in place, with no slice
   *  copies. Same result as forward(q, k, v) on the sliced blocks. */
  forwardPacked(
    qkv: Value,
    call: { qPosOffset?: number; segments?: Value; maxSegment?: number } = {},
  ): Value {
    const { qHeads, kvHeads, headDim, windowLeft, windowRight } = this.opts;
    return sdpaPacked(qkv, {
      qHeads,
      kvHeads,
      headDim,
      windowLeft,
      windowRight,
      qPosOffset: call.qPosOffset,
      sinks: this.sinks?.value,
      segments: call.segments,
      maxSegment: call.maxSegment,
    });
  }
}
