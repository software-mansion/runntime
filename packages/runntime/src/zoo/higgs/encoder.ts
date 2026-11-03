/** The Higgs Audio v2 codec encoder: 24 kHz mono audio to codes [8, frames]
 *  at 25 frames per second, loaded from the codec's transformers checkpoint.
 *
 *  Two branches give each frame its channels. DAC's encoder turns the audio
 *  into 256 acoustic channels. HuBERT reads the audio at 16 kHz; the mean of
 *  its 13 hidden states (its first layer's input and every layer's output),
 *  at every second frame, runs through a conv stack into 768 semantic
 *  channels. Side by side they go through fc, then residual vector
 *  quantization: each codebook takes the row nearest to the residual's
 *  projection, and that row's project_out leaves the next residual.
 *  Time-major throughout. */

import {
  add,
  cat,
  concatRows,
  derivedTensor,
  elu,
  gatherRowsFrom,
  gelu,
  logitStats,
  matrix,
  mul,
  nn,
  reshape,
  RunntimeError,
  scaleWeight,
  slice,
  splitRows,
  sub,
  type LazyStateDict,
  type LazyTensor,
  type Value,
} from '../../core/index.ts';
import { checkpointMismatch, requireTensor } from '../errors.ts';
import { HIGGS_ENCODER, type HiggsEncoderConfig } from './config.ts';
import { ResidualUnit } from './decoder.ts';
import { resample } from './resample.ts';

const WHAT = 'higgs encoder weights';

const at = (prefix: string, name: string) => (prefix ? `${prefix}.${name}` : name);

/** Removes and returns `prefix.name`; CHECKPOINT_MISMATCH when it is missing. */
function take(sd: LazyStateDict, prefix: string, name: string): LazyTensor {
  const t = requireTensor(sd, at(prefix, name), WHAT);
  sd.tensors.delete(at(prefix, name));
  return t;
}

/** Frames out of an unpadded conv over `frames` frames. */
const convFrames = (frames: number, kernel: number, stride: number) =>
  Math.floor((frames - kernel) / stride) + 1;

/** Three residual units with dilations 1, 3 and 9, Snake, then the conv that
 *  downsamples by `stride` and doubles the channels. */
class AcousticBlock extends nn.Module {
  readonly res_unit1: ResidualUnit;
  readonly res_unit2: ResidualUnit;
  readonly res_unit3: ResidualUnit;
  readonly snake1: nn.Snake;
  readonly conv1: nn.Conv1d;
  constructor(channels: number, stride: number, kernel: number) {
    super();
    this.res_unit1 = new ResidualUnit(channels, kernel, 1);
    this.res_unit2 = new ResidualUnit(channels, kernel, 3);
    this.res_unit3 = new ResidualUnit(channels, kernel, 9);
    this.snake1 = new nn.Snake(channels);
    this.conv1 = new nn.Conv1d(channels, 2 * channels, 2 * stride, {
      stride,
      padding: Math.ceil(stride / 2),
    });
  }
  override forward(x: Value): Value {
    const h = this.res_unit3.forward(this.res_unit2.forward(this.res_unit1.forward(x)));
    return this.conv1.forward(this.snake1.forward(h));
  }
}

class AcousticEncoder extends nn.Module {
  readonly conv1: nn.Conv1d;
  readonly block: nn.ModuleList<AcousticBlock>;
  readonly snake1: nn.Snake;
  readonly conv2: nn.Conv1d;
  constructor({ channels, strides, kernel, outKernel, latent }: HiggsEncoderConfig['acoustic']) {
    super();
    this.conv1 = new nn.Conv1d(1, channels, kernel, { padding: (kernel - 1) / 2 });
    this.block = new nn.ModuleList(
      strides.map((s, i) => new AcousticBlock(channels * 2 ** i, s, kernel)),
    );
    const last = channels * 2 ** strides.length;
    this.snake1 = new nn.Snake(last);
    this.conv2 = new nn.Conv1d(last, latent, outKernel, { padding: (outKernel - 1) / 2 });
  }
  override forward(x: Value): Value {
    return this.conv2.forward(this.snake1.forward(this.block.forward(this.conv1.forward(x))));
  }
}

/** One feature conv, without bias, then GELU. The first one normalizes each
 *  channel over time before the GELU. */
class FeatureConv extends nn.Module {
  readonly conv: nn.Conv1d;
  readonly layer_norm?: nn.GroupNorm;
  constructor(cIn: number, cOut: number, kernel: number, stride: number, norm: boolean) {
    super();
    this.conv = new nn.Conv1d(cIn, cOut, kernel, {
      stride,
      bias: false,
      activation: norm ? undefined : 'gelu',
    });
    if (norm) this.layer_norm = new nn.GroupNorm(cOut, cOut);
  }
  override forward(x: Value): Value {
    const h = this.conv.forward(x);
    return this.layer_norm ? gelu(this.layer_norm.forward(h)) : h;
  }
}

class FeatureExtractor extends nn.Module {
  readonly conv_layers: nn.ModuleList<FeatureConv>;
  constructor({ convDim, convKernels, convStrides }: HiggsEncoderConfig['hubert']) {
    super();
    this.conv_layers = new nn.ModuleList(
      convKernels.map(
        (k, i) => new FeatureConv(i === 0 ? 1 : convDim, convDim, k, convStrides[i]!, i === 0),
      ),
    );
  }
  override forward(x: Value): Value {
    return this.conv_layers.forward(x);
  }
}

class FeatureProjection extends nn.Module {
  readonly layer_norm: nn.LayerNorm;
  readonly projection: nn.Linear;
  constructor({ convDim, hidden }: HiggsEncoderConfig['hubert']) {
    super();
    this.layer_norm = new nn.LayerNorm(convDim, 1e-5, { bias: true });
    this.projection = new nn.Linear(convDim, hidden);
  }
  override forward(x: Value): Value {
    return this.projection.forward(this.layer_norm.forward(x));
  }
}

/** HuBERT's conv position embedding: a grouped conv with weight norm, as one
 *  conv per group of channels, then GELU. An even kernel pads one frame less
 *  on the right, the frame torch cuts. The load folds the weight norm and
 *  splits the weight and bias into the groups. */
class PositionConv extends nn.Module {
  readonly convs: nn.ModuleList<nn.Conv1d>;
  constructor(
    readonly hidden: number,
    readonly kernel: number,
    readonly groups: number,
  ) {
    super();
    const width = hidden / groups;
    const pad = Math.floor(kernel / 2);
    this.convs = new nn.ModuleList(
      Array.from(
        { length: groups },
        () =>
          new nn.Conv1d(width, width, kernel, {
            padLeft: pad,
            padRight: kernel % 2 === 0 ? pad - 1 : pad,
            activation: 'gelu',
          }),
      ),
    );
  }
  override transformStateDict(sd: LazyStateDict, prefix: string): void {
    const g = take(sd, prefix, 'conv.parametrizations.weight.original0');
    const v = take(sd, prefix, 'conv.parametrizations.weight.original1');
    const bias = take(sd, prefix, 'conv.bias');
    const { hidden, kernel, groups } = this;
    const width = hidden / groups;
    if (
      v.shape.join() !== [hidden, width, kernel].join() ||
      g.shape.join() !== [1, 1, kernel].join()
    ) {
      throw checkpointMismatch(
        WHAT,
        `pos_conv_embed weight norm has shapes [${g.shape}] and [${v.shape}], ` +
          `expected [1,1,${kernel}] and [${hidden},${width},${kernel}]`,
      );
    }
    const sizes = Array<number>(groups).fill(width);
    const weights = splitRows(foldWeightNorm(g, v), sizes);
    const biases = splitRows(bias, sizes);
    for (let i = 0; i < groups; i++) {
      sd.tensors.set(at(prefix, `convs.${i}.weight`), weights[i]!);
      sd.tensors.set(at(prefix, `convs.${i}.bias`), biases[i]!);
    }
  }
  override forward(x: Value): Value {
    const width = this.hidden / this.groups;
    const parts = this.convs.items.map((conv, i) =>
      conv.forward(slice(x, 1, i * width, (i + 1) * width)),
    );
    return cat(parts, 1);
  }
}

/** torch's weight norm over dim 2 of v [out, in, kernel]: each tap's [out,
 *  in] slice scaled to norm g[tap]. In f64. */
function foldWeightNorm(g: LazyTensor, v: LazyTensor): LazyTensor {
  const [out, cIn, kernel] = v.shape as [number, number, number];
  return derivedTensor(
    [out, cIn, kernel],
    async () => {
      const gs = await g.f32();
      const vs = await v.f32();
      const norm = new Float64Array(kernel);
      for (let i = 0; i < vs.length; i++) norm[i % kernel]! += vs[i]! ** 2;
      const w = new Float32Array(vs.length);
      for (let i = 0; i < vs.length; i++) {
        const k = i % kernel;
        w[i] = (vs[i]! * gs[k]!) / Math.sqrt(norm[k]!);
      }
      return w;
    },
    g.byteLength + v.byteLength,
  );
}

/** Self-attention with q, k and v in one projection. The load stacks them
 *  and folds the score scale into q. */
class HubertAttention extends nn.Module {
  readonly qkv: nn.Linear;
  readonly mha: nn.MultiHeadAttention;
  readonly out_proj: nn.Linear;
  constructor(
    readonly hidden: number,
    readonly heads: number,
  ) {
    super();
    this.qkv = new nn.Linear(hidden, 3 * hidden);
    this.mha = new nn.MultiHeadAttention({
      qHeads: heads,
      kvHeads: heads,
      headDim: hidden / heads,
      windowLeft: Infinity,
      windowRight: Infinity,
    });
    this.out_proj = new nn.Linear(hidden, hidden);
  }
  override transformStateDict(sd: LazyStateDict, prefix: string): void {
    const scale = 1 / Math.sqrt(this.hidden / this.heads);
    for (const part of ['weight', 'bias']) {
      const [q, k, v] = ['q', 'k', 'v'].map((p) => take(sd, prefix, `${p}_proj.${part}`));
      sd.tensors.set(at(prefix, `qkv.${part}`), concatRows([scaleWeight(q!, scale), k!, v!]));
    }
  }
  override forward(x: Value): Value {
    return this.out_proj.forward(this.mha.forwardPacked(this.qkv.forward(x)), { addend: x });
  }
}

class FeedForward extends nn.Module {
  readonly intermediate_dense: nn.Linear;
  readonly output_dense: nn.Linear;
  constructor(hidden: number, ffn: number) {
    super();
    this.intermediate_dense = new nn.Linear(hidden, ffn, { activation: 'gelu' });
    this.output_dense = new nn.Linear(ffn, hidden);
  }
  override forward(x: Value): Value {
    return this.output_dense.forward(this.intermediate_dense.forward(x), { addend: x });
  }
}

/** A post-norm layer: attention plus the input, LayerNorm, the MLP plus its
 *  input, LayerNorm. */
class HubertLayer extends nn.Module {
  readonly attention: HubertAttention;
  readonly layer_norm: nn.LayerNorm;
  readonly feed_forward: FeedForward;
  readonly final_layer_norm: nn.LayerNorm;
  constructor({ hidden, heads, ffn }: HiggsEncoderConfig['hubert']) {
    super();
    this.attention = new HubertAttention(hidden, heads);
    this.layer_norm = new nn.LayerNorm(hidden, 1e-5, { bias: true });
    this.feed_forward = new FeedForward(hidden, ffn);
    this.final_layer_norm = new nn.LayerNorm(hidden, 1e-5, { bias: true });
  }
  override forward(x: Value): Value {
    const h = this.layer_norm.forward(this.attention.forward(x));
    return this.final_layer_norm.forward(this.feed_forward.forward(h));
  }
}

/** The position embedding added, LayerNorm, then the layers. Returns the sum
 *  of the hidden states: the layers' input and each layer's output. */
class HubertEncoder extends nn.Module {
  readonly pos_conv_embed: PositionConv;
  readonly layer_norm: nn.LayerNorm;
  readonly layers: nn.ModuleList<HubertLayer>;
  constructor(cfg: HiggsEncoderConfig['hubert']) {
    super();
    this.pos_conv_embed = new PositionConv(cfg.hidden, cfg.posKernel, cfg.posGroups);
    this.layer_norm = new nn.LayerNorm(cfg.hidden, 1e-5, { bias: true });
    this.layers = new nn.ModuleList(Array.from({ length: cfg.layers }, () => new HubertLayer(cfg)));
  }
  override forward(x: Value): Value {
    let h = this.layer_norm.forward(add(x, this.pos_conv_embed.forward(x)));
    let sum = h;
    for (const layer of this.layers.items) {
      h = layer.forward(h);
      sum = add(sum, h);
    }
    return sum;
  }
}

class Hubert extends nn.Module {
  readonly feature_extractor: FeatureExtractor;
  readonly feature_projection: FeatureProjection;
  readonly encoder: HubertEncoder;
  constructor(private readonly cfg: HiggsEncoderConfig['hubert']) {
    super();
    this.feature_extractor = new FeatureExtractor(cfg);
    this.feature_projection = new FeatureProjection(cfg);
    this.encoder = new HubertEncoder(cfg);
  }
  /** Samples [n, 1] to the mean hidden state at every frameStride-th frame.
   *  Whole codec frames give a whole number of strides; each stride's frames
   *  sit side by side in one row of a reshape, the first of them kept. */
  override forward(x: Value): Value {
    const features = this.feature_projection.forward(this.feature_extractor.forward(x));
    const { frameStride, hidden, layers } = this.cfg;
    const frames = features.shape.dims![0]!;
    if (frames % frameStride !== 0) {
      throw new RunntimeError(
        'INVALID_ARGUMENT',
        `HiggsEncoder: ${frames} HuBERT frames are not whole strides of ${frameStride}`,
      );
    }
    const sum = this.encoder.forward(features);
    const strides = reshape(sum, [frames / frameStride, frameStride * hidden]);
    return mul(slice(strides, 1, 0, hidden), 1 / (layers + 1));
  }
}

/** ELU, a conv without bias with ELU after it, a 1×1 conv without bias, plus
 *  the input. */
class SemanticUnit extends nn.Module {
  readonly conv1: nn.Conv1d;
  readonly conv2: nn.Conv1d;
  constructor(channels: number, kernel: number, dilation: number) {
    super();
    this.conv1 = new nn.Conv1d(channels, channels, kernel, {
      dilation,
      padding: ((kernel - 1) / 2) * dilation,
      bias: false,
      activation: 'elu',
    });
    this.conv2 = new nn.Conv1d(channels, channels, 1, { bias: false });
  }
  override forward(x: Value): Value {
    return add(x, this.conv2.forward(this.conv1.forward(elu(x))));
  }
}

class SemanticBlock extends nn.Module {
  readonly res_units: nn.ModuleList<SemanticUnit>;
  readonly conv: nn.Conv1d;
  constructor(channels: number, kernel: number, dilations: readonly number[]) {
    super();
    this.res_units = new nn.ModuleList(dilations.map((d) => new SemanticUnit(channels, kernel, d)));
    this.conv = new nn.Conv1d(channels, channels, kernel, { padding: (kernel - 1) / 2 });
  }
  override forward(x: Value): Value {
    return this.conv.forward(this.res_units.forward(x));
  }
}

class SemanticEncoder extends nn.Module {
  readonly conv: nn.Conv1d;
  readonly conv_blocks: nn.ModuleList<SemanticBlock>;
  constructor({
    hubert,
    semanticBlocks,
    semanticDilations,
    semanticKernel: k,
  }: HiggsEncoderConfig) {
    super();
    const c = hubert.hidden;
    this.conv = new nn.Conv1d(c, c, k, { padding: (k - 1) / 2, bias: false });
    this.conv_blocks = new nn.ModuleList(
      Array.from({ length: semanticBlocks }, () => new SemanticBlock(c, k, semanticDilations)),
    );
  }
  override forward(x: Value): Value {
    return this.conv_blocks.forward(this.conv.forward(x));
  }
}

/** A codebook's rows, and a score per row that is largest at the row nearest
 *  to the input: 2·row·x − |row|², the squared distance less |x|², negated.
 *  The load derives the score from the rows and drops the training state. */
class Codebook extends nn.Module {
  readonly embed: nn.Parameter;
  readonly score: nn.Linear;
  constructor(size: number, dim: number) {
    super();
    this.embed = new nn.Parameter(matrix(size, dim));
    this.score = new nn.Linear(dim, size);
  }
  override transformStateDict(sd: LazyStateDict, prefix: string): void {
    const embed = requireTensor(sd, at(prefix, 'embed'), WHAT);
    const [size, dim] = embed.shape as [number, number];
    for (const name of ['cluster_size', 'embed_avg', 'inited']) sd.tensors.delete(at(prefix, name));
    // The rows' bytes count once, under embed.
    sd.tensors.set(
      at(prefix, 'score.weight'),
      derivedTensor([size, dim], async () => (await embed.f32()).map((x) => 2 * x)),
    );
    sd.tensors.set(
      at(prefix, 'score.bias'),
      derivedTensor([size], async () => {
        const e = await embed.f32();
        const out = new Float32Array(size);
        for (let i = 0; i < size; i++) {
          let sq = 0;
          for (let j = 0; j < dim; j++) sq += e[i * dim + j]! ** 2;
          out[i] = -sq;
        }
        return out;
      }),
    );
  }
}

class VectorQuantizer extends nn.Module {
  readonly project_in: nn.Linear;
  readonly codebook: Codebook;
  readonly project_out: nn.Linear;
  constructor(hidden: number, size: number, dim: number) {
    super();
    this.project_in = new nn.Linear(hidden, dim);
    this.codebook = new Codebook(size, dim);
    this.project_out = new nn.Linear(dim, hidden);
  }
  /** The nearest row's index [T, 1] for residual rows [T, hidden], and that
   *  row through project_out. */
  quantize(residual: Value): { code: Value; quantized: Value } {
    const scores = this.codebook.score.forward(this.project_in.forward(residual));
    const size = scores.shape.dims![1]!;
    const stats = logitStats(scores, { blocks: 1, block: size, argmaxCols: size });
    const code = slice(stats, 1, 0, 1);
    const quantized = this.project_out.forward(gatherRowsFrom(this.codebook.embed.value, code));
    return { code, quantized };
  }
}

class ResidualVectorQuantizer extends nn.Module {
  readonly quantizers: nn.ModuleList<VectorQuantizer>;
  constructor({ acoustic, hubert, codebooks, codebookSize, codebookDim }: HiggsEncoderConfig) {
    super();
    const hidden = acoustic.latent + hubert.hidden;
    this.quantizers = new nn.ModuleList(
      Array.from(
        { length: codebooks },
        () => new VectorQuantizer(hidden, codebookSize, codebookDim),
      ),
    );
  }
  /** Rows [T, hidden] to codes [T, codebooks], as f32. */
  override forward(x: Value): Value {
    let residual = x;
    const codes: Value[] = [];
    for (const q of this.quantizers.items) {
      const { code, quantized } = q.quantize(residual);
      codes.push(code);
      residual = sub(residual, quantized);
    }
    return cat(codes, 1);
  }
}

/** Prefixes of the tensors the encoder reads; the decoder's are dropped. */
const ENCODER_KEYS = [
  'acoustic_encoder.',
  'semantic_model.',
  'encoder_semantic.',
  'fc.',
  'quantizer.',
];

export class HiggsEncoder extends nn.Module<[Value, Value], Value> {
  readonly acoustic_encoder: AcousticEncoder;
  readonly semantic_model: Hubert;
  readonly encoder_semantic: SemanticEncoder;
  readonly fc: nn.Linear;
  readonly quantizer: ResidualVectorQuantizer;

  constructor(readonly cfg: HiggsEncoderConfig = HIGGS_ENCODER) {
    super();
    this.acoustic_encoder = new AcousticEncoder(cfg.acoustic);
    this.semantic_model = new Hubert(cfg.hubert);
    this.encoder_semantic = new SemanticEncoder(cfg);
    const hidden = cfg.acoustic.latent + cfg.hubert.hidden;
    this.fc = new nn.Linear(hidden, hidden);
    this.quantizer = new ResidualVectorQuantizer(cfg);
  }

  /** Samples per frame. */
  get hop(): number {
    return this.cfg.acoustic.strides.reduce((a, b) => a * b, 1);
  }

  /** Drops the tensors the encoder does not read: the decoders, fc1, fc2
   *  and HuBERT's mask embedding. */
  override transformStateDict(sd: LazyStateDict, prefix: string): void {
    const scope = prefix ? `${prefix}.` : '';
    for (const name of [...sd.tensors.keys()]) {
      if (!name.startsWith(scope)) continue;
      const own = name.slice(scope.length);
      if (
        !ENCODER_KEYS.some((k) => own.startsWith(k)) ||
        own === 'semantic_model.masked_spec_embed'
      ) {
        sd.tensors.delete(name);
      }
    }
  }

  /** The two inputs for 24 kHz mono `samples`, zero-padded on the right to
   *  whole frames as the codec's feature extractor pads them: the samples
   *  [frames·hop, 1], and HuBERT's [n, 1], the padded samples at 16 kHz with
   *  zeros on both sides. */
  inputs(samples: Float32Array): {
    acoustic: Float32Array;
    semantic: Float32Array;
    frames: number;
  } {
    if (!(samples instanceof Float32Array) || samples.length === 0) {
      throw new RunntimeError('INVALID_ARGUMENT', 'HiggsEncoder: needs a non-empty Float32Array');
    }
    const { sampleRate, semanticRate, hubert } = this.cfg;
    const frames = Math.ceil(samples.length / this.hop);
    const acoustic = new Float32Array(frames * this.hop);
    acoustic.set(samples);
    const resampled = resample(acoustic, sampleRate, semanticRate);
    const semantic = new Float32Array(resampled.length + 2 * hubert.pad);
    semantic.set(resampled, hubert.pad);
    return { acoustic, semantic, frames };
  }

  /** Bytes of the largest buffer an encode of `frames` frames makes: the
   *  first DAC conv's output, or the first HuBERT conv's. */
  private largestBuffer(frames: number): number {
    const { sampleRate, semanticRate, acoustic, hubert } = this.cfg;
    const samples = frames * this.hop;
    const semantic = Math.ceil((samples * semanticRate) / sampleRate) + 2 * hubert.pad;
    const rows = convFrames(semantic, hubert.convKernels[0]!, hubert.convStrides[0]!);
    return 4 * Math.max(samples * acoustic.channels, rows * hubert.convDim);
  }

  /** The most frames one encode takes when no buffer may pass `maxBytes`. */
  maxFrames(maxBytes: number): number {
    if (!Number.isFinite(maxBytes)) throw new Error(`maxFrames: ${maxBytes} bytes must be finite`);
    let lo = 0;
    let hi = 1;
    while (this.largestBuffer(hi) <= maxBytes) hi *= 2;
    while (hi - lo > 1) {
      const mid = (lo + hi) >> 1;
      if (this.largestBuffer(mid) <= maxBytes) lo = mid;
      else hi = mid;
    }
    return lo;
  }

  /** Codes [frames, codebooks] as f32, from the two inputs of inputs(). */
  override forward(acoustic: Value, semantic: Value): Value {
    const a = this.acoustic_encoder.forward(acoustic);
    const s = this.encoder_semantic.forward(this.semantic_model.forward(semantic));
    const [frames] = a.shape.dims!;
    if (s.shape.dims![0] !== frames) {
      throw new RunntimeError(
        'INVALID_ARGUMENT',
        `HiggsEncoder: ${frames} acoustic frames but ${s.shape.dims![0]} semantic frames`,
      );
    }
    return this.quantizer.forward(this.fc.forward(cat([a, s], 1)));
  }

  /** Codes [codebooks, frames], codebook-major, from forward()'s rows. */
  codes(rows: Float32Array, frames: number): Int32Array {
    const { codebooks } = this.cfg;
    if (rows.length !== codebooks * frames) {
      throw new RunntimeError(
        'INVALID_ARGUMENT',
        `HiggsEncoder: ${rows.length} rows for ${codebooks} codebooks × ${frames} frames`,
      );
    }
    const out = new Int32Array(codebooks * frames);
    for (let t = 0; t < frames; t++) {
      for (let c = 0; c < codebooks; c++) out[c * frames + t] = rows[t * codebooks + c]!;
    }
    return out;
  }
}
