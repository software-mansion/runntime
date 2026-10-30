/** Stock nn layers: each holds Parameters and calls the matching op. */

import { RunntimeError } from '../error.ts';
import { groupNorm, linear } from '../graph/compose.ts';
import {
  type ConvAct,
  type SlotAct,
  conv1d,
  conv2d,
  convTranspose2d,
  gatherRows,
  gatherRowsFrom,
  layerNorm,
  matmul,
  reshape,
  rmsNorm,
  sdpa,
  sdpaPacked,
  slice,
  snake,
} from '../graph/ops.ts';
import { matrix, type Value } from '../graph/value.ts';
import {
  MATMUL_QUANT_TILED_BK,
  matmulQuantTiledEligible,
} from '../kernels/matmul/matmulQuantTiled.ts';
import {
  convToTapMajor,
  f16ToF32,
  packConv1x1Hwc4F16,
  packConvHwc4F16,
  packConvWinogradF16,
  packDwHwc4F16,
  permuteConvTransposeF16,
} from '../weights/convert.ts';
import { foldBnIntoConv } from '../weights/foldBn.ts';
import { packQuantW } from '../weights/quantPack.ts';
import { derivedTensor, type LazyStateDict, type QuantTensor } from '../weights/safetensors.ts';
import { readFloats, transposeWeight } from '../weights/transforms.ts';
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
 *  A quantized checkpoint weight loads as quantW words with its f32 scales in
 *  `weightScale` [in / groupSize, out], and forward() runs the quantized
 *  matmul. Only the checkpoint says which kind it is, so transformStateDict
 *  retypes the slot at load.
 *
 *  `addend` and `activation` fold a residual add and a following tanh or GELU
 *  into the matmul epilogue, as act(x·W + bias) + addend. */
export class Linear extends Module {
  readonly weight: Parameter;
  readonly bias?: Parameter;
  readonly activation?: SlotAct;
  /** Set when a quantized weight loads. */
  weightScale?: Parameter;
  /** How the loaded weight is quantized, or undefined for a float weight. */
  private quantization?: { bits: 4 | 8; groupSize: number };
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
      throw new RunntimeError(
        'CHECKPOINT_MISMATCH',
        `Linear: ${name} has shape ${JSON.stringify(tensor.shape)}, ` +
          `expected torch layout [${this.outFeatures}, ${this.inFeatures}]`,
      );
    }
    if (tensor.kind === 'quant') {
      this.loadQuantized(sd, prefix, name, tensor);
      return;
    }
    sd.tensors.set(name, transposeWeight(tensor));
  }
  /** Replaces `name` with quantW words [in, out] and adds its scales
   *  [in / groupSize, out] right after it, so they bind in file order. The
   *  words carry the whole byteLength of the ints and scales. */
  private loadQuantized(
    sd: LazyStateDict,
    prefix: string,
    name: string,
    tensor: QuantTensor,
  ): void {
    const { bits, groupSize } = tensor;
    const k = this.inFeatures;
    const n = this.outFeatures;
    // The quantized matmul reads whole tiles of 4 columns and 8 rows and
    // packs int4 columns 8 to a word.
    if (!matmulQuantTiledEligible(k, n, groupSize) || n % (32 / bits) !== 0) {
      throw new RunntimeError(
        'CHECKPOINT_MISMATCH',
        `Linear: ${name} is int${bits} in groups of ${groupSize}, which needs in features ` +
          `a multiple of ${MATMUL_QUANT_TILED_BK} and of the group size, out features a ` +
          `multiple of ${32 / bits} and a group size a multiple of 4, got [out, in] = [${n}, ${k}]`,
      );
    }
    this.quantization = { bits, groupSize };
    this.weight.retype('quantW', (k * n * bits) / 32);
    this.weightScale = new Parameter(
      { elems: (k / groupSize) * n, dtype: 'f32', dims: [k / groupSize, n] },
      undefined,
      true,
    );
    const words = derivedTensor(
      [k, n],
      async () => packQuantW(await tensor.ints(), k, n, bits),
      tensor.byteLength,
    );
    const groups = k / groupSize;
    const scales = derivedTensor([groups, n], async () => {
      const data = await tensor.scales();
      const f32 = data instanceof Float32Array ? data : f16ToF32(data);
      // torch's [out, in / groupSize] to the kernels' [in / groupSize, out]
      const out = new Float32Array(f32.length);
      for (let o = 0; o < n; o++) {
        for (let g = 0; g < groups; g++) out[g * n + o] = f32[o * groups + g]!;
      }
      return out;
    });
    const scaleName = prefix ? `${prefix}.weightScale` : 'weightScale';
    const entries = [...sd.tensors];
    sd.tensors.clear();
    for (const [key, t] of entries) {
      sd.tensors.set(key, key === name ? words : t);
      if (key === name) sd.tensors.set(scaleName, scales);
    }
  }
  forward(x: Value, call: { addend?: Value } = {}): Value {
    if (!this.weightScale || !this.quantization) {
      return linear(x, this.weight.value, this.bias?.value, call.addend, this.activation);
    }
    return matmul(x, this.weight.value, {
      scales: this.weightScale.value,
      ...this.quantization,
      bias: this.bias?.value,
      addend: call.addend,
      activation: this.activation,
    });
  }
}

/** Strided, dilated 1D convolution over time-major [T, inChannels]. No batch
 *  dim, no groups.
 *
 *  `padding` takes torch's number, 'valid' or 'same'; padLeft and padRight pad
 *  each side independently instead. span − 1 on the left is causal, with
 *  span = dilation·(kernelSize − 1) + 1.
 *  weight is stored tap-major and permutes from torch layout at load. bias and
 *  `activation` fold into the conv epilogue. */
export class Conv1d extends Module {
  readonly weight: Parameter;
  readonly bias?: Parameter;
  readonly stride: number;
  readonly dilation: number;
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
      dilation?: number;
      bias?: boolean;
      padding?: number | 'valid' | 'same';
      padLeft?: number;
      padRight?: number;
      activation?: SlotAct;
    } = {},
  ) {
    super();
    this.stride = opts.stride ?? 1;
    this.dilation = opts.dilation ?? 1;
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
      throw new RunntimeError(
        'CHECKPOINT_MISMATCH',
        `Conv1d: ${name} has shape ${JSON.stringify(tensor.shape)}, ` +
          `expected torch layout [${this.outChannels}, ${this.inChannels}, ${this.kernelSize}]`,
      );
    }
    sd.tensors.set(
      name,
      derivedTensor(
        [this.kernelSize * this.inChannels, this.outChannels],
        async () =>
          convToTapMajor(
            await readFloats('Conv1d', tensor),
            this.outChannels,
            this.inChannels,
            this.kernelSize,
          ),
        tensor.byteLength,
      ),
    );
  }
  forward(x: Value): Value {
    return conv1d(x, this.weight.value, {
      kernelSize: this.kernelSize,
      stride: this.stride,
      dilation: this.dilation,
      padding: this.padding,
      padLeft: this.padLeft,
      padRight: this.padRight,
      bias: this.bias?.value,
      activation: this.activation,
    });
  }
}

/** Transposed 1D convolution over time-major [T, inChannels], like torch's
 *  ConvTranspose1d, for kernelSize a multiple of stride.
 *
 *  Runs as a plain conv: kernelSize / stride taps to stride·outChannels
 *  channels. Each output row holds `stride` frames side by side, so a reshape
 *  unfolds them and a slice cuts the padding. The weight permutes from torch
 *  layout [in, out, kernel] at load, and the bias repeats per frame. */
export class ConvTranspose1d extends Module {
  readonly weight: Parameter;
  readonly bias?: Parameter;
  readonly taps: number;
  readonly stride: number;
  readonly padding: number;
  readonly outputPadding: number;
  constructor(
    readonly inChannels: number,
    readonly outChannels: number,
    readonly kernelSize: number,
    opts: { stride: number; padding?: number; outputPadding?: number; bias?: boolean },
  ) {
    super();
    const { stride, padding = 0, outputPadding = 0 } = opts;
    if (kernelSize % stride !== 0) {
      throw new Error(
        `ConvTranspose1d: kernelSize ${kernelSize} must be a multiple of stride ${stride}`,
      );
    }
    if (outputPadding > padding) {
      throw new Error(
        `ConvTranspose1d: outputPadding ${outputPadding} must not exceed padding ${padding}`,
      );
    }
    this.taps = kernelSize / stride;
    this.stride = stride;
    this.padding = padding;
    this.outputPadding = outputPadding;
    this.weight = new Parameter(matrix(this.taps * inChannels, stride * outChannels));
    this.bias =
      opts.bias === false
        ? undefined
        : new Parameter({
            elems: stride * outChannels,
            dtype: 'f32',
            dims: [stride * outChannels],
          });
  }
  override transformStateDict(sd: LazyStateDict, prefix: string): void {
    const at = (n: string) => (prefix ? `${prefix}.${n}` : n);
    const { inChannels: cIn, outChannels: cOut, kernelSize: k, taps, stride: s } = this;
    const w = sd.tensors.get(at('weight'));
    if (w) {
      if (w.shape.length !== 3 || w.shape[0] !== cIn || w.shape[1] !== cOut || w.shape[2] !== k) {
        throw new RunntimeError(
          'CHECKPOINT_MISMATCH',
          `ConvTranspose1d: ${at('weight')} has shape ${JSON.stringify(w.shape)}, ` +
            `expected torch layout [${cIn}, ${cOut}, ${k}]`,
        );
      }
      // Row q, tap i reads input frame q − (taps − 1) + i. Its frame r in
      // that row uses kernel position (taps − 1 − i)·s + r.
      sd.tensors.set(
        at('weight'),
        derivedTensor(
          [taps * cIn, s * cOut],
          async () => {
            const src = await w.f32();
            const out = new Float32Array(taps * cIn * s * cOut);
            for (let i = 0; i < taps; i++) {
              for (let c = 0; c < cIn; c++) {
                const row = (i * cIn + c) * s * cOut;
                for (let r = 0; r < s; r++) {
                  const kk = (taps - 1 - i) * s + r;
                  for (let o = 0; o < cOut; o++) {
                    out[row + r * cOut + o] = src[(c * cOut + o) * k + kk]!;
                  }
                }
              }
            }
            return out;
          },
          w.byteLength,
        ),
      );
    }
    const b = sd.tensors.get(at('bias'));
    if (b && this.bias) {
      sd.tensors.set(
        at('bias'),
        derivedTensor(
          [s * cOut],
          async () => {
            const src = await b.f32();
            const out = new Float32Array(s * cOut);
            for (let r = 0; r < s; r++) out.set(src, r * cOut);
            return out;
          },
          b.byteLength,
        ),
      );
    }
  }
  forward(x: Value): Value {
    const { taps, stride: s, padding: p, outChannels } = this;
    const t = x.shape.dims![0]!;
    const y = conv1d(x, this.weight.value, {
      kernelSize: taps,
      stride: 1,
      padLeft: taps - 1,
      padRight: taps - 1,
      bias: this.bias?.value,
    });
    const rows = (t + taps - 1) * s;
    const frames = (t - 1) * s - 2 * p + this.kernelSize + this.outputPadding;
    const z = reshape(y, [rows, outChannels]);
    return p === 0 && frames === rows ? z : slice(z, 0, p, p + frames);
  }
}

/** Snake activation x + sin²(αx) / (α + 1e-9), one α per channel, over
 *  time-major [T, channels]. The weight is f32 [2, channels]: α, then
 *  1 / (α + 1e-9). A torch checkpoint's `alpha` [1, channels, 1] converts at
 *  load. */
export class Snake extends Module {
  readonly weight: Parameter;
  constructor(readonly channels: number) {
    super();
    this.weight = new Parameter(
      { elems: 2 * channels, dtype: 'f32', dims: [2, channels] },
      undefined,
      true, // pinned: the kernel reads α in f32 whatever the activations are
    );
  }
  override transformStateDict(sd: LazyStateDict, prefix: string): void {
    const at = (n: string) => (prefix ? `${prefix}.${n}` : n);
    const alpha = sd.tensors.get(at('alpha'));
    if (!alpha) return;
    sd.tensors.delete(at('alpha'));
    const c = this.channels;
    sd.tensors.set(
      at('weight'),
      derivedTensor(
        [2, c],
        async () => {
          const a = await alpha.f32();
          const out = new Float32Array(2 * c);
          for (let i = 0; i < c; i++) {
            out[i] = a[i]!;
            out[c + i] = 1 / (a[i]! + 1e-9);
          }
          return out;
        },
        alpha.byteLength,
      ),
    );
  }
  forward(x: Value): Value {
    return snake(x, this.weight.value);
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
      /** Runs a 3×3, stride 1, padding 1 conv as Winograd F(4×4, 3×3): a
       *  quarter of the multiplies, rounded differently in f16. The weight
       *  packs into 4× its size. */
      winograd?: boolean;
    },
  ) {
    super();
    const { kernelSize, stride = 1, padding = 0, groups = 1, bias = true, winograd } = opts;
    const [kH, kW] = typeof kernelSize === 'number' ? [kernelSize, kernelSize] : kernelSize;
    // The convs the dispatch runs on the 1×1 kernels.
    const pointwise = kH === 1 && kW === 1 && stride === 1 && padding === 0 && groups === 1;
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
        if (winograd) return packConvWinogradF16(data, outChannels, inChannels);
        if (pointwise) return packConv1x1Hwc4F16(data, outChannels, inChannels);
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
    const { kernelSize, stride = 1, padding = 0, groups = 1, activation, winograd } = this.opts;
    return conv2d(x, this.weight.value, this.bias?.value, {
      kernelSize,
      stride,
      padding,
      groups,
      activation,
      winograd,
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
