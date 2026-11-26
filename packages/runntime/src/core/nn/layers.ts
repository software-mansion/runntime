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
import { matmulQuantTiledEligible } from '../kernels/matmul/matmulQuantTiled.ts';
import {
  convToTapMajor,
  f16ToF32,
  packConv1x1Hwc4,
  packConvHwc4,
  packConvWinogradF16,
  packDwHwc4,
  permuteConvTransposeF16,
} from '../weights/convert.ts';
import { foldBnIntoConv } from '../weights/foldBn.ts';
import { packQuantW } from '../weights/quantPack.ts';
import {
  derive,
  deriveFromTensors,
  type LazyStateDict,
  type LazyTensor,
  type QuantTensor,
} from '../weights/safetensors.ts';
import { concatRows, dequantize, readFloats, transposeWeight } from '../weights/transforms.ts';
import { tensor } from '../gpu/buffers.ts';
import { CpuParameter, Module, Parameter, type AnyModule } from './module.ts';

/** Token-id lookup table: ids to rows of a [numEmbeddings, dims] weight.
 *
 *  On 'gpu' the weight doubles as a tied LM head, read through `weight.value`
 *  in its raw orientation. On 'cpu' it never reaches the GPU — forward()
 *  gathers the requested rows and uploads those — and tying is unavailable.
 *
 *  A quantized checkpoint weight loads as its dequantized floats. */
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
  override transformStateDict(sd: LazyStateDict, prefix: string): void {
    const name = prefix ? `${prefix}.weight` : 'weight';
    const t = sd.tensors.get(name);
    if (t?.kind === 'quant') sd.tensors.set(name, dequantize(t));
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

/** Turns `weight` into quantW words [k, n], packed from `ints` with one row
 *  of k per output, and puts its f32 scales right after it as
 *  `<prefix>.weightScale`, so both bind in file order. The words carry t's
 *  whole download size. Returns the scale parameter. */
function loadQuantWeight(
  sd: LazyStateDict,
  prefix: string,
  weight: Parameter,
  t: QuantTensor,
  shape: { k: number; n: number },
  ints: () => Promise<Int8Array>,
  scales: { dims: number[]; data: () => Promise<Float32Array> },
): Parameter {
  const { k, n } = shape;
  const at = (name: string) => (prefix ? `${prefix}.${name}` : name);
  weight.retype('quantW', (k * n * t.bits) / 32, [k, n]);
  const elems = scales.dims.reduce((a, b) => a * b, 1);
  const scale = new Parameter({ elems, dtype: 'f32', dims: scales.dims }, undefined, true);
  const words = deriveFromTensors([k, n], [t], async () => packQuantW(await ints(), k, n, t.bits));
  const entries = [...sd.tensors];
  sd.tensors.clear();
  for (const [key, tensor] of entries) {
    sd.tensors.set(key, key === at('weight') ? words : tensor);
    if (key === at('weight')) {
      sd.tensors.set(at('weightScale'), deriveFromTensors(scales.dims, [t], scales.data, 0));
    }
  }
  return scale;
}

/** A quantized weight its layer's quantized kernel cannot run, as its
 *  dequantized f32 values in torch's `shape`, and noted on `sd` for
 *  loadStateDict's warning. */
function asFloat(sd: LazyStateDict, t: QuantTensor, shape: readonly number[]): LazyTensor {
  (sd.quantizedAsFloat ??= []).push(...t.sources.map((s) => s.name));
  return dequantize(t, shape);
}

/** `t`'s scales in f32, as stored. */
async function scalesF32(t: QuantTensor): Promise<Float32Array> {
  const data = await t.scales();
  return data instanceof Float32Array ? data : f16ToF32(data);
}

/** `t`'s scales, torch's [out, groups], as the kernels' [groups, n] in f32.
 *  Columns past out stay 0. */
async function scaleColumns(t: QuantTensor, groups: number, n: number): Promise<Float32Array> {
  const f32 = await scalesF32(t);
  const out = new Float32Array(groups * n);
  for (let o = 0; o < t.shape[0]!; o++) {
    for (let g = 0; g < groups; g++) out[g * n + o] = f32[o * groups + g]!;
  }
  return out;
}

/** The FusedLinear each of its parts is in. Kept off the Linear: a Module
 *  field would register the FusedLinear as the part's child. */
const FUSED_INTO = new WeakMap<Linear, FusedLinear>();

/** Throws CHECKPOINT_MISMATCH unless `tensor` is `layer`'s weight in torch's
 *  [out, in]. Without it a square wrong-layout weight loads silently
 *  transposed. */
function checkTorchLayout(
  name: string,
  tensor: LazyTensor,
  layer: { inFeatures: number; outFeatures: number },
): void {
  const { inFeatures, outFeatures } = layer;
  if (
    tensor.shape.length === 2 &&
    tensor.shape[0] === outFeatures &&
    tensor.shape[1] === inFeatures
  )
    return;
  throw new RunntimeError(
    'CHECKPOINT_MISMATCH',
    `Linear: ${name} has shape ${JSON.stringify(tensor.shape)}, ` +
      `expected torch layout [${outFeatures}, ${inFeatures}]`,
  );
}

/** Whether the quantized matmul runs `t` as the weight of a Linear from
 *  `inFeatures` to `outFeatures`. It reads whole tiles of 4 columns and 8
 *  rows, groups of whole 4-row chunks, and packs int4 columns 8 to a word. */
const quantLinearFits = (inFeatures: number, outFeatures: number, t: QuantTensor): boolean =>
  matmulQuantTiledEligible(inFeatures, outFeatures, t.groupSize) &&
  outFeatures % (32 / t.bits) === 0;

/** y = x @ weight + bias. weight is stored [in, out]; torch checkpoints
 *  transpose once at load.
 *
 *  A quantized checkpoint weight loads as quantW words with its f32 scales in
 *  `weightScale` [in / groupSize, out], and forward() runs the quantized
 *  matmul. Only the checkpoint says which kind it is, so transformStateDict
 *  retypes the slot at load. A weight the quantized matmul cannot run loads
 *  as its dequantized floats.
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
  /** Whether a FusedLinear holds this layer's weight and bias. */
  get fused(): boolean {
    return FUSED_INTO.has(this);
  }
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
    if (this.fused) return; // the FusedLinear checks and stacks it
    const name = prefix ? `${prefix}.weight` : 'weight';
    const tensor = sd.tensors.get(name);
    if (!tensor) return; // missing keys are the key check's job, not ours
    checkTorchLayout(name, tensor, this);
    if (tensor.kind !== 'quant') {
      sd.tensors.set(name, transposeWeight(tensor));
      return;
    }
    if (quantLinearFits(this.inFeatures, this.outFeatures, tensor)) {
      this.loadQuantized(sd, prefix, tensor);
    } else sd.tensors.set(name, transposeWeight(asFloat(sd, tensor, tensor.shape)));
  }
  /** Loads the weight as quantW words [in, out] with its scales
   *  [in / groupSize, out]. */
  private loadQuantized(sd: LazyStateDict, prefix: string, tensor: QuantTensor): void {
    const { bits, groupSize } = tensor;
    const k = this.inFeatures;
    const n = this.outFeatures;
    this.quantization = { bits, groupSize };
    const groups = k / groupSize;
    this.weightScale = loadQuantWeight(
      sd,
      prefix,
      this.weight,
      tensor,
      { k, n },
      () => tensor.ints(),
      { dims: [groups, n], data: () => scaleColumns(tensor, groups, n) },
    );
  }
  /** A FusedLinear held outside the module tree is never loaded, so its
   *  parts refuse the load. */
  override fuseStateDict(
    _sd: LazyStateDict,
    paths: ReadonlyMap<AnyModule, readonly string[]>,
  ): void {
    const fusedInto = FUSED_INTO.get(this);
    if (fusedInto && !paths.has(fusedInto)) {
      throw new Error(
        `Linear at '${paths.get(this)?.[0]}': the FusedLinear over it is not in the module tree. ` +
          'Assign the FusedLinear to a field of a module of the model',
      );
    }
  }
  override *storageParameters(prefix = ''): Generator<readonly [string, Parameter]> {
    if (!this.fused) yield* super.storageParameters(prefix);
  }
  forward(x: Value, call: { addend?: Value } = {}): Value {
    if (this.fused) {
      throw new Error(
        'Linear: this layer is a part of a FusedLinear, which holds its weight. ' +
          "Call the FusedLinear's forward() and take this part's columns",
      );
    }
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

/** Several Linears over the same input as one matmul: the output is the
 *  parts' outputs side by side, in `parts` order.
 *
 *  The parts stay the fields the checkpoint names, and the FusedLinear is one
 *  more field, of any module of the model:
 *
 *      q_proj = new nn.Linear(hidden, hidden);
 *      k_proj = new nn.Linear(hidden, kvDim);
 *      qk = new nn.FusedLinear([this.q_proj, this.k_proj]);
 *
 *  The checkpoint keeps one weight per part. At load the parts' rows stack
 *  into one weight, once every transformStateDict hook has run, so it holds
 *  what the hooks made of each part. A part without a bias, next to one that
 *  has it, adds zeros. Quantized parts with the same bits and group size
 *  stack as ints with their scales, and run the quantized matmul when a
 *  Linear of the whole size would. Any other mix (float with quantized, int8
 *  with int4, two group sizes) stacks the dequantized floats.
 *
 *  The parts only name the tensors: namedParameters() lists their
 *  parameters, never loaded, and a part's own forward() throws. */
export class FusedLinear extends Module {
  /** The Linear that holds the stacked weight. The underscore keeps it out of
   *  the module tree, so the walk gives it no name. */
  private readonly _stacked: Linear;
  constructor(private readonly parts: readonly Linear[]) {
    super();
    const first = parts[0];
    if (!first) throw new Error('FusedLinear: needs at least one part');
    for (const [i, part] of parts.entries()) {
      if (parts.indexOf(part) !== i) throw new Error(`FusedLinear: part ${i} is listed twice`);
      if (part.inFeatures !== first.inFeatures) {
        throw new Error(
          `FusedLinear: parts read ${first.inFeatures} and ${part.inFeatures} features, need one input size`,
        );
      }
      if (part.activation) {
        throw new Error(`FusedLinear: part ${i} has the activation '${part.activation}'`);
      }
      if (part.fused) throw new Error(`FusedLinear: part ${i} is already in another FusedLinear`);
    }
    for (const part of parts) FUSED_INTO.set(part, this);
    this._stacked = new Linear(
      first.inFeatures,
      parts.reduce((sum, part) => sum + part.outFeatures, 0),
      { bias: parts.some((part) => part.bias) },
    );
  }
  override *storageParameters(prefix = ''): Generator<readonly [string, Parameter]> {
    yield* this._stacked.namedParameters(prefix);
  }
  override fuseStateDict(
    sd: LazyStateDict,
    paths: ReadonlyMap<AnyModule, readonly string[]>,
  ): void {
    const [prefix = '', alias] = paths.get(this) ?? [];
    if (alias !== undefined) {
      throw new Error(
        `FusedLinear: held in two fields, '${prefix}' and '${alias}'. Keep it in one`,
      );
    }
    const weights: { name: string; tensor: LazyTensor }[] = [];
    const biases: LazyTensor[] = [];
    for (const [i, part] of this.parts.entries()) {
      const [path, second] = paths.get(part) ?? [];
      if (path === undefined) {
        throw new Error(
          `FusedLinear at '${prefix}': part ${i} of ${this.parts.length} is not in the module ` +
            'tree. Assign every part to a field of a module of the model',
        );
      }
      if (second !== undefined) {
        throw new Error(
          `FusedLinear at '${prefix}': part ${i} is held in two fields, '${path}' and ` +
            `'${second}'. Keep it in one`,
        );
      }
      const name = `${path}.weight`;
      const tensor = sd.tensors.get(name);
      if (!tensor) return; // a missing key is reported by the key check
      checkTorchLayout(name, tensor, part);
      weights.push({ name, tensor });
      if (!part.bias) {
        biases.push(derive([part.outFeatures], [], () => new Float32Array(part.outFeatures)));
        continue;
      }
      const bias = sd.tensors.get(`${path}.bias`);
      if (!bias) return;
      if (bias.shape.length !== 1 || bias.shape[0] !== part.outFeatures) {
        throw new RunntimeError(
          'CHECKPOINT_MISMATCH',
          `Linear: ${path}.bias has shape ${JSON.stringify(bias.shape)}, ` +
            `expected [${part.outFeatures}]`,
        );
      }
      biases.push(bias);
    }
    /** A part as loadStateDict's warnings name it: by its file tensors. */
    const fileName = ({ name, tensor }: (typeof weights)[number]) =>
      tensor.sources.map((s) => s.name).join(' + ') || name;
    const setting = (t: LazyTensor) =>
      t.kind === 'quant' ? `int${t.bits} in groups of ${t.groupSize}` : 'float';
    const { inFeatures, outFeatures } = this._stacked;
    let stacked: LazyTensor;
    if (new Set(weights.map(({ tensor }) => setting(tensor))).size > 1) {
      (sd.fusedPartsDiffer ??= []).push(
        `${prefix}: ${weights.map((w) => `${fileName(w)} is ${setting(w.tensor)}`).join(', ')}`,
      );
      stacked = concatRows(
        weights.map(({ tensor }) => (tensor.kind === 'quant' ? dequantize(tensor) : tensor)),
      );
    } else {
      stacked = concatRows(weights.map(({ tensor }) => tensor));
      if (stacked.kind === 'quant' && !quantLinearFits(inFeatures, outFeatures, stacked)) {
        (sd.quantizedAsFloat ??= []).push(
          `${prefix} (FusedLinear of ${weights.map(fileName).join(', ')})`,
        );
        stacked = dequantize(stacked);
      }
    }
    sd.tensors.set(`${prefix}.weight`, stacked);
    if (this._stacked.bias) sd.tensors.set(`${prefix}.bias`, concatRows(biases));
    this._stacked.transformStateDict(sd, prefix);
  }
  forward(x: Value): Value {
    return this._stacked.forward(x);
  }
}

/** Throws CHECKPOINT_MISMATCH unless the quantized conv weight `t` is its
 *  torch weight with the dims after the first flattened, [rows, cols]. */
function checkQuantConvShape(
  layer: string,
  name: string,
  t: QuantTensor,
  { rows, cols, layout }: { rows: number; cols: number; layout: string },
): void {
  if (t.shape[0] !== rows || t.shape[1] !== cols) {
    throw new RunntimeError(
      'CHECKPOINT_MISMATCH',
      `${layer}: ${name} is quantized as [${t.shape.join(', ')}], expected ${layout} = [${rows}, ${cols}]`,
    );
  }
}

/** Whether a conv over `taps` taps of cIn channels to n columns runs on the
 *  quantized matmul. */
function quantConvFits(cIn: number, taps: number, n: number): boolean {
  return cIn % 4 === 0 && matmulQuantTiledEligible(taps * cIn, n);
}

/** Strided, dilated 1D convolution over time-major [T, inChannels]. No batch
 *  dim. Groups are 1 or fully depthwise.
 *
 *  `padding` takes torch's number, 'valid' or 'same'; padLeft and padRight pad
 *  each side independently instead. span − 1 on the left is causal, with
 *  span = dilation·(kernelSize − 1) + 1.
 *  weight is stored tap-major and permutes from torch layout at load. bias and
 *  `activation` fold into the conv epilogue.
 *
 *  A depthwise conv, `groups` equal to both channel counts, loads torch's
 *  [channels, 1, kernelSize] weight and runs causal only: stride 1, no
 *  dilation, padLeft kernelSize − 1, `bias: false`, no activation, channels a
 *  multiple of 4. forward() throws INVALID_ARGUMENT for anything else.
 *
 *  A quantized checkpoint weight is torch's [out, in / groups · kernel], one
 *  scale per group of values along each row. It loads as quantW words with its
 *  f32 scales in `weightScale` [in·kernel / groupSize, out]. Output channels
 *  pad up to the packing width, scales and bias with them, and are cut after
 *  the conv. A conv the quantized matmul cannot run, every depthwise one
 *  included, loads a quantized weight as its dequantized floats. */
export class Conv1d extends Module {
  readonly weight: Parameter;
  readonly bias?: Parameter;
  /** Set when a quantized weight loads. */
  weightScale?: Parameter;
  /** How the loaded weight is quantized, or undefined for a float weight. */
  private quantization?: { bits: 4 | 8; groupSize: number };
  readonly stride: number;
  readonly dilation: number;
  readonly padding?: number | 'valid' | 'same';
  readonly padLeft?: number;
  readonly padRight?: number;
  readonly activation?: SlotAct;
  readonly groups: number;
  /** Throws INVALID_ARGUMENT for `groups` that are neither 1 nor both channel
   *  counts. */
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
      groups?: number;
    } = {},
  ) {
    super();
    this.stride = opts.stride ?? 1;
    this.dilation = opts.dilation ?? 1;
    this.padding = opts.padding;
    this.padLeft = opts.padLeft;
    this.padRight = opts.padRight;
    this.activation = opts.activation;
    this.groups = opts.groups ?? 1;
    if (this.groups !== 1 && (this.groups !== inChannels || inChannels !== outChannels)) {
      throw new RunntimeError(
        'INVALID_ARGUMENT',
        `Conv1d: groups=${this.groups} is neither 1 nor depthwise (cIn=${inChannels}, cOut=${outChannels})`,
      );
    }
    const rows = kernelSize * (inChannels / this.groups);
    this.weight = new Parameter(matrix(rows, outChannels));
    this.bias =
      opts.bias === false
        ? undefined
        : new Parameter({ elems: outChannels, dtype: 'f32', dims: [outChannels] });
  }
  override transformStateDict(sd: LazyStateDict, prefix: string): void {
    const name = prefix ? `${prefix}.weight` : 'weight';
    let tensor = sd.tensors.get(name);
    if (!tensor) return; // a missing key is reported by the key check
    const { outChannels: cOut, kernelSize: kernel } = this;
    // Input channels each output reads.
    const cIn = this.inChannels / this.groups;
    if (tensor.kind === 'quant') {
      checkQuantConvShape('Conv1d', name, tensor, {
        rows: cOut,
        cols: cIn * kernel,
        layout: this.groups === 1 ? '[out, in·kernel]' : '[out, kernel]',
      });
      const width = 32 / tensor.bits;
      const n = Math.ceil(cOut / width) * width;
      if (this.groups === 1 && quantConvFits(cIn, kernel, n)) {
        this.loadQuantized(sd, prefix, name, tensor, n);
        return;
      }
      tensor = asFloat(sd, tensor, [cOut, cIn, kernel]);
    }
    // Without this a weight with equal in/out channels loads scrambled.
    if (
      tensor.shape.length !== 3 ||
      tensor.shape[0] !== cOut ||
      tensor.shape[1] !== cIn ||
      tensor.shape[2] !== kernel
    ) {
      throw new RunntimeError(
        'CHECKPOINT_MISMATCH',
        `Conv1d: ${name} has shape ${JSON.stringify(tensor.shape)}, ` +
          `expected torch layout [${cOut}, ${cIn}, ${kernel}]`,
      );
    }
    sd.tensors.set(
      name,
      deriveFromTensors([kernel * cIn, cOut], [tensor], async ([torch]) =>
        convToTapMajor(await readFloats('Conv1d', torch), cOut, cIn, kernel),
      ),
    );
  }
  /** Loads `name` as quantW words [kernel·in, n] with its scales
   *  [in·kernel / groupSize, n], where n is outChannels padded to the packing
   *  width. The bias pads to n too. */
  private loadQuantized(
    sd: LazyStateDict,
    prefix: string,
    name: string,
    t: QuantTensor,
    n: number,
  ): void {
    const { inChannels: cIn, outChannels: cOut, kernelSize: kernel } = this;
    const k = kernel * cIn;
    const groups = k / t.groupSize;
    this.quantization = { bits: t.bits, groupSize: t.groupSize };
    this.weightScale = loadQuantWeight(
      sd,
      prefix,
      this.weight,
      t,
      { k, n },
      async () => {
        // torch's [out, in, kernel] to one tap-major row per output
        const src = await t.ints();
        const out = new Int8Array(n * k);
        for (let o = 0; o < cOut; o++) {
          for (let c = 0; c < cIn; c++) {
            for (let tap = 0; tap < kernel; tap++) {
              out[o * k + tap * cIn + c] = src[(o * cIn + c) * kernel + tap]!;
            }
          }
        }
        return out;
      },
      { dims: [groups, n], data: () => scaleColumns(t, groups, n) },
    );
    const biasName = prefix ? `${prefix}.bias` : 'bias';
    const b = sd.tensors.get(biasName);
    if (n > cOut && b && this.bias) {
      if (b.shape.join() !== `${cOut}`) {
        throw new RunntimeError(
          'CHECKPOINT_MISMATCH',
          `Conv1d: ${biasName} has shape ${JSON.stringify(b.shape)}, expected [${cOut}]`,
        );
      }
      this.bias.retype(this.bias.shape.dtype, n, [n]);
      sd.tensors.set(
        biasName,
        deriveFromTensors([n], [b], async ([b]) => {
          const src = await readFloats('Conv1d', b);
          const out = src instanceof Uint16Array ? new Uint16Array(n) : new Float32Array(n);
          out.set(src);
          return out;
        }),
      );
    }
  }
  forward(x: Value): Value {
    const y = conv1d(x, this.weight.value, {
      kernelSize: this.kernelSize,
      stride: this.stride,
      dilation: this.dilation,
      padding: this.padding,
      padLeft: this.padLeft,
      padRight: this.padRight,
      bias: this.bias?.value,
      activation: this.activation,
      groups: this.groups,
      ...(this.weightScale && this.quantization
        ? { scales: this.weightScale.value, ...this.quantization }
        : {}),
    });
    const cols = this.weight.shape.dims![1]!;
    return cols > this.outChannels ? slice(y, 1, 0, this.outChannels) : y;
  }
}

/** Transposed 1D convolution over time-major [T, inChannels], like torch's
 *  ConvTranspose1d, for kernelSize a multiple of stride.
 *
 *  Runs as a plain conv: kernelSize / stride taps to stride·outChannels
 *  channels. Each output row holds `stride` frames side by side, so a reshape
 *  unfolds them and a slice cuts the padding. The weight permutes from torch
 *  layout [in, out, kernel] at load, and the bias repeats per frame.
 *
 *  A quantized checkpoint weight is torch's [in, out·kernel], one scale per
 *  group of values along each row. It loads as quantW words with its f32
 *  scales as stored in `weightScale`. A transposed conv the quantized matmul
 *  cannot run loads a quantized weight as its dequantized floats. */
export class ConvTranspose1d extends Module {
  readonly weight: Parameter;
  readonly bias?: Parameter;
  /** Set when a quantized weight loads. */
  weightScale?: Parameter;
  /** How the loaded weight is quantized, or undefined for a float weight. */
  private quantization?: { bits: 4 | 8; groupSize: number };
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
    let w = sd.tensors.get(at('weight'));
    if (w?.kind === 'quant') {
      checkQuantConvShape('ConvTranspose1d', at('weight'), w, {
        rows: cIn,
        cols: cOut * k,
        layout: '[in, out·kernel]',
      });
      const n = s * cOut;
      if (cOut % 4 === 0 && n % (32 / w.bits) === 0 && quantConvFits(cIn, taps, n)) {
        this.loadQuantized(sd, prefix, w);
      } else {
        w = asFloat(sd, w, [cIn, cOut, k]);
      }
    }
    if (w && w.kind !== 'quant') {
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
        derive([taps * cIn, s * cOut], [w], ([src]) => {
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
        }),
      );
    }
    const b = sd.tensors.get(at('bias'));
    if (b && this.bias) {
      sd.tensors.set(
        at('bias'),
        derive([s * cOut], [b], ([src]) => {
          const out = new Float32Array(s * cOut);
          for (let r = 0; r < s; r++) out.set(src, r * cOut);
          return out;
        }),
      );
    }
  }
  /** Loads `t` as quantW words [taps·in, stride·out] with its scales
   *  [in, out·kernel / groupSize] as stored. */
  private loadQuantized(sd: LazyStateDict, prefix: string, t: QuantTensor): void {
    const { inChannels: cIn, outChannels: cOut, kernelSize: kernel, taps, stride: s } = this;
    const n = s * cOut;
    const k = taps * cIn;
    const cols = cOut * kernel;
    this.quantization = { bits: t.bits, groupSize: t.groupSize };
    const groups = cols / t.groupSize;
    this.weightScale = loadQuantWeight(
      sd,
      prefix,
      this.weight,
      t,
      { k, n },
      async () => {
        // Row i·in + c, column r·out + o takes kernel position
        // (taps − 1 − i)·s + r.
        const src = await t.ints();
        const out = new Int8Array(n * k);
        for (let r = 0; r < s; r++) {
          for (let o = 0; o < cOut; o++) {
            for (let i = 0; i < taps; i++) {
              const kk = (taps - 1 - i) * s + r;
              for (let c = 0; c < cIn; c++) {
                out[(r * cOut + o) * k + i * cIn + c] = src[c * cols + o * kernel + kk]!;
              }
            }
          }
        }
        return out;
      },
      { dims: [cIn, groups], data: () => scalesF32(t) },
    );
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
      ...(this.weightScale && this.quantization
        ? {
            scales: this.weightScale.value,
            ...this.quantization,
            transposedKernel: this.kernelSize,
          }
        : {}),
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
      derive([2, c], [alpha], ([a]) => {
        const out = new Float32Array(2 * c);
        for (let i = 0; i < c; i++) {
          out[i] = a[i]!;
          out[c + i] = 1 / (a[i]! + 1e-9);
        }
        return out;
      }),
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

/** GroupNorm over [M, numChannels], statistics per group of adjacent
 *  channels, per-channel affine. numGroups must divide numChannels. */
export class GroupNorm extends Module {
  readonly weight: Parameter;
  readonly bias: Parameter;
  constructor(
    readonly numGroups: number,
    readonly numChannels: number,
    readonly eps = 1e-5,
  ) {
    super();
    if (!Number.isInteger(numGroups) || numGroups < 1 || numChannels % numGroups !== 0) {
      throw new Error(`GroupNorm: numGroups ${numGroups} must divide the ${numChannels} channels`);
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
 *  with toHwc4 and leave with toChw. Weight and bias are f32 until half()
 *  makes them f16, which needs shader-f16; the activations have the weight's
 *  dtype. Groups are 1 or fully depthwise. BatchNorm folds in at load rather
 *  than running, so `bias: true` is normal for BN-fused checkpoints. */
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
       *  packs into 4× its size. f16 only: loading an f32 weight throws
       *  INVALID_ARGUMENT. */
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
      { elems: outChannels * kk, dtype: 'f32', dims: [outChannels, kk], layout: 'hwc4' },
      (data) => {
        if (data instanceof Uint32Array) {
          throw new Error('Conv2d: weight must arrive as f32 values or f16 bits');
        }
        if (winograd) {
          if (!(data instanceof Uint16Array)) {
            throw new RunntimeError(
              'INVALID_ARGUMENT',
              'Conv2d: winograd has no f32 kernel yet, run the model in f16 with half()',
            );
          }
          return packConvWinogradF16(data, outChannels, inChannels);
        }
        if (pointwise) return packConv1x1Hwc4(data, outChannels, inChannels);
        return dw
          ? packDwHwc4(data, outChannels, kH, kW)
          : packConvHwc4(data, outChannels, inChannels, kH, kW);
      },
    );
    this.bias = bias
      ? new Parameter({ elems: outChannels, dtype: 'f32', dims: [outChannels] })
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
  readonly weight: Parameter;
  constructor(
    readonly n: number,
    readonly eps = 1e-6,
  ) {
    super();
    this.weight = new Parameter({ elems: n, dtype: 'f32', dims: [n] });
  }
  forward(x: Value): Value {
    return rmsNorm(x, this.weight.value, this.eps);
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
  /** Multiplies q·k before the softmax. Default 1/√headDim, as in PyTorch's
   *  `scaled_dot_product_attention`. */
  scale?: number;
  /** Registers a per-q-head sink-logit Parameter named `sinks`. */
  sinks?: boolean;
}

/** Banded, causal or cross multi-head attention over packed
 *  [T, heads·headDim] slabs.
 *
 *  Takes already-projected q/k/v — models own the projections — and scales
 *  the scores by `scale`, 1/√headDim unless the options say otherwise. */
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
    const { qHeads, kvHeads, headDim, windowLeft, windowRight, scale } = this.opts;
    return sdpa(q, k, v, {
      qHeads,
      kvHeads,
      headDim,
      windowLeft,
      windowRight,
      qPosOffset: call.qPosOffset,
      scale,
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
    const { qHeads, kvHeads, headDim, windowLeft, windowRight, scale } = this.opts;
    return sdpaPacked(qkv, {
      qHeads,
      kvHeads,
      headDim,
      windowLeft,
      windowRight,
      qPosOffset: call.qPosOffset,
      scale,
      sinks: this.sinks?.value,
      segments: call.segments,
      maxSegment: call.maxSegment,
    });
  }
}
