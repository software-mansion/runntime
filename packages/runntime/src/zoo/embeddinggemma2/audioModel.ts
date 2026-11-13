/** EmbeddingGemma 2 audio tower: a conformer over a clip's log-mel frames,
 *  one token per 40 ms, projected to the text model's width.
 *
 *  Two stride-2 convs shrink time and frequency 4×. Each layer is a
 *  half-weighted feed-forward, local attention over this token and the 11
 *  before it, a gated causal depthwise conv, a second feed-forward and a
 *  norm. Most linears clamp their input and output to bounds from the
 *  checkpoint.
 *
 *  Module fields are named like the checkpoint: the tower under
 *  `audio_tower`, its projection to the text model under `embed_audio`. */

import {
  add,
  clamp,
  gluConv1d,
  gatherRowsFrom,
  localAttention,
  nn,
  reshape,
  rmsNorm,
  silu,
  type LazyStateDict,
  type Value,
} from '../../core/index.ts';
import {
  EMBEDDINGGEMMA2_AUDIO,
  EMBEDDINGGEMMA2_TEXT,
  type EmbeddingGemma2AudioConfig,
} from './config.ts';
import { transformEmbeddingGemma2AudioStateDict } from './stateDictHooks.ts';

/** k's scale in every score, fixed by the architecture. */
const K_SCALE = Math.log1p(Math.E) / Math.LN2;

/** One clip's inputs. */
export interface EmbeddingGemma2AudioInputs {
  /** The first conv's 3×3 taps [P + 1, 9] for each of its P output pixels,
   *  time-major, then a row of zeros. */
  patches: Value;
  /** For each tap of each output pixel of the second conv, the first conv's
   *  pixel it reads, f32; P, the zero row, where it reads padding. */
  taps: Value;
  /** Tokens: frames / 4, rounded up. */
  tokens: number;
}

/** A bound of a clipped linear, read on the CPU. */
const bound = () => new nn.CpuParameter({ elems: 1, dtype: 'f32', dims: [1] });

/** A linear that clamps its input and output to the checkpoint's bounds. */
class ClippedLinear extends nn.Module {
  readonly linear: nn.Linear;
  readonly input_min = bound();
  readonly input_max = bound();
  readonly output_min = bound();
  readonly output_max = bound();

  constructor(inFeatures: number, outFeatures: number) {
    super();
    this.linear = new nn.Linear(inFeatures, outFeatures, { bias: false });
  }

  /** Bounds of the output. */
  get outBounds(): { lo: number; hi: number } {
    return { lo: this.output_min.data[0]!, hi: this.output_max.data[0]! };
  }

  /** The matmul on the clamped input, its output left unclamped. */
  unclamped(x: Value): Value {
    return this.linear.forward(clamp(x, this.input_min.data[0]!, this.input_max.data[0]!));
  }

  forward(x: Value): Value {
    const { lo, hi } = this.outBounds;
    return clamp(this.unclamped(x), lo, hi);
  }
}

class FeedForward extends nn.Module<[Value], Value> {
  readonly pre_layer_norm: nn.RMSNorm;
  readonly ffw_layer_1: ClippedLinear;
  readonly ffw_layer_2: ClippedLinear;
  /** Its weight carries the residual weight, folded in at load. */
  readonly post_layer_norm: nn.RMSNorm;

  constructor({ hidden, ffn, eps }: EmbeddingGemma2AudioConfig) {
    super();
    this.pre_layer_norm = new nn.RMSNorm(hidden, eps);
    this.ffw_layer_1 = new ClippedLinear(hidden, ffn);
    this.ffw_layer_2 = new ClippedLinear(ffn, hidden);
    this.post_layer_norm = new nn.RMSNorm(hidden, eps);
  }

  override forward(x: Value): Value {
    const h = silu(this.ffw_layer_1.forward(this.pre_layer_norm.forward(x)));
    const post = this.post_layer_norm;
    return rmsNorm(this.ffw_layer_2.forward(h), post.weight.value, post.eps, { residual: x });
  }
}

/** The depthwise conv's taps [convKernel, hidden], oldest first, permuted
 *  from the checkpoint's [hidden, 1, convKernel] at load. */
class DepthwiseConv extends nn.Module {
  readonly weight: nn.Parameter;
  constructor({ hidden, convKernel }: EmbeddingGemma2AudioConfig) {
    super();
    this.weight = new nn.Parameter({
      elems: convKernel * hidden,
      dtype: 'f32',
      dims: [convKernel, hidden],
    });
  }
}

class LightConv extends nn.Module<[Value], Value> {
  readonly pre_layer_norm: nn.RMSNorm;
  /** The two GLU halves, value then gate. */
  readonly linear_start: ClippedLinear;
  readonly depthwise_conv1d: DepthwiseConv;
  readonly conv_norm: nn.RMSNorm;
  readonly linear_end: ClippedLinear;

  constructor(cfg: EmbeddingGemma2AudioConfig) {
    super();
    const { hidden, eps } = cfg;
    this.pre_layer_norm = new nn.RMSNorm(hidden, eps);
    this.linear_start = new ClippedLinear(hidden, 2 * hidden);
    this.depthwise_conv1d = new DepthwiseConv(cfg);
    this.conv_norm = new nn.RMSNorm(hidden, eps);
    this.linear_end = new ClippedLinear(hidden, hidden);
  }

  override forward(x: Value): Value {
    const z = this.linear_start.unclamped(this.pre_layer_norm.forward(x));
    const g = gluConv1d(z, this.depthwise_conv1d.weight.value, this.linear_start.outBounds);
    return add(this.linear_end.forward(silu(this.conv_norm.forward(g))), x);
  }
}

/** Local attention: the projections fused at load, the distance bias and q
 *  scale computed from the checkpoint at load. */
class AudioAttention extends nn.Module {
  /** q | k | v rows, with the bounds of all three. */
  readonly qkv: ClippedLinear;
  /** One bias row per key distance [window, hidden], f32, from
   *  `relative_k_proj`. */
  readonly rel: nn.Parameter;
  /** Scale of each column of q [headDim], f32, from `per_dim_scale`. */
  readonly q_scale: nn.Parameter;
  readonly post: ClippedLinear;

  constructor({ hidden, headDim, window }: EmbeddingGemma2AudioConfig) {
    super();
    // Pinned: the attention kernel reads them as f32.
    const f32 = (...dims: number[]) =>
      new nn.Parameter(
        { elems: dims.reduce((a, b) => a * b), dtype: 'f32', dims },
        undefined,
        true,
      );
    this.qkv = new ClippedLinear(hidden, 3 * hidden);
    this.rel = f32(window, hidden);
    this.q_scale = f32(headDim);
    this.post = new ClippedLinear(hidden, hidden);
  }
}

class AudioLayer extends nn.Module<[Value], Value> {
  readonly feed_forward1: FeedForward;
  readonly norm_pre_attn: nn.RMSNorm;
  readonly self_attn: AudioAttention;
  readonly norm_post_attn: nn.RMSNorm;
  readonly lconv1d: LightConv;
  readonly feed_forward2: FeedForward;
  readonly norm_out: nn.RMSNorm;

  constructor(private readonly cfg: EmbeddingGemma2AudioConfig) {
    super();
    const { hidden, eps } = cfg;
    this.feed_forward1 = new FeedForward(cfg);
    this.norm_pre_attn = new nn.RMSNorm(hidden, eps);
    this.self_attn = new AudioAttention(cfg);
    this.norm_post_attn = new nn.RMSNorm(hidden, eps);
    this.lconv1d = new LightConv(cfg);
    this.feed_forward2 = new FeedForward(cfg);
    this.norm_out = new nn.RMSNorm(hidden, eps);
  }

  override forward(x: Value): Value {
    const { heads, headDim, window, logitCap } = this.cfg;
    const { self_attn } = this;
    x = this.feed_forward1.forward(x);
    const qkv = self_attn.qkv.forward(this.norm_pre_attn.forward(x));
    const a = localAttention(qkv, self_attn.rel.value, self_attn.q_scale.value, {
      heads,
      headDim,
      window,
      kScale: K_SCALE,
      softcap: logitCap,
    });
    const norm = this.norm_post_attn;
    x = rmsNorm(self_attn.post.forward(a), norm.weight.value, norm.eps, { residual: x });
    x = this.lconv1d.forward(x);
    return this.norm_out.forward(this.feed_forward2.forward(x));
  }
}

/** A stride-2 conv as a matmul over its taps, LayerNorm over channels, ReLU. */
class ConvLayer extends nn.Module {
  readonly conv: nn.Linear;
  readonly norm: nn.LayerNorm;
  constructor(taps: number, channels: number, eps: number) {
    super();
    this.conv = new nn.Linear(taps, channels, { bias: false });
    this.norm = new nn.LayerNorm(channels, eps);
  }
  forward(x: Value): Value {
    return clamp(this.norm.forward(this.conv.forward(x)), 0, Infinity);
  }
}

/** The two convs over the mel frames and the projection to the layers'
 *  width. layer0 reads [P + 1, 9] taps of the one-channel input, layer1 the
 *  [P', 9·c0] taps of layer0's output; both weights permute at load. */
class SubsampleConvProjection extends nn.Module {
  readonly layer0: ConvLayer;
  readonly layer1: ConvLayer;
  readonly input_proj_linear: nn.Linear;
  constructor({ hidden, mels, eps, convChannels: [c0, c1] }: EmbeddingGemma2AudioConfig) {
    super();
    this.layer0 = new ConvLayer(9, c0, eps);
    this.layer1 = new ConvLayer(9 * c0, c1, eps);
    this.input_proj_linear = new nn.Linear((mels / 4) * c1, hidden, { bias: false });
  }
}

class AudioTower extends nn.Module {
  readonly subsample_conv_projection: SubsampleConvProjection;
  readonly layers: nn.ModuleList<AudioLayer>;
  readonly output_proj: nn.Linear;
  constructor(cfg: EmbeddingGemma2AudioConfig) {
    super();
    this.subsample_conv_projection = new SubsampleConvProjection(cfg);
    this.layers = new nn.ModuleList(Array.from({ length: cfg.layers }, () => new AudioLayer(cfg)));
    this.output_proj = new nn.Linear(cfg.hidden, cfg.outputDim);
  }
}

/** The tower output's projection to the text model's width. */
class EmbedAudio extends nn.Module {
  private readonly eps: number;
  /** A weightless norm: its weight is ones, a Buffer since the checkpoint
   *  has none. */
  readonly embedding_pre_projection_norm: nn.Buffer;
  readonly embedding_projection: nn.Linear;
  constructor({ outputDim, eps }: EmbeddingGemma2AudioConfig) {
    super();
    this.eps = eps;
    this.embedding_pre_projection_norm = new nn.Buffer(new Float32Array(outputDim).fill(1), {
      elems: outputDim,
      dtype: 'f32',
      dims: [outputDim],
    });
    this.embedding_projection = new nn.Linear(outputDim, EMBEDDINGGEMMA2_TEXT.hidden, {
      bias: false,
    });
  }
  forward(x: Value): Value {
    return this.embedding_projection.forward(
      rmsNorm(x, this.embedding_pre_projection_norm.value, this.eps),
    );
  }
}

export class EmbeddingGemma2AudioModel extends nn.Module<[EmbeddingGemma2AudioInputs], Value> {
  readonly audio_tower: AudioTower;
  readonly embed_audio: EmbedAudio;

  constructor(readonly cfg: EmbeddingGemma2AudioConfig = EMBEDDINGGEMMA2_AUDIO) {
    super();
    this.audio_tower = new AudioTower(cfg);
    this.embed_audio = new EmbedAudio(cfg);
  }

  /** One clip to its tokens [tokens, text hidden], ready to stand in for
   *  the placeholder rows of the text model's input. */
  override forward(inp: EmbeddingGemma2AudioInputs): Value {
    const [c0, c1] = this.cfg.convChannels;
    const { subsample_conv_projection: sub, layers, output_proj } = this.audio_tower;
    const y0 = sub.layer0.forward(inp.patches);
    const pixels = inp.taps.shape.elems / 9;
    const y = sub.layer1.forward(reshape(gatherRowsFrom(y0, inp.taps), [pixels, 9 * c0]));
    // Rows are (token, frequency); each token's frequencies become one row.
    let x = sub.input_proj_linear.forward(reshape(y, [inp.tokens, (pixels / inp.tokens) * c1]));
    for (const layer of layers.items) x = layer.forward(x);
    return this.embed_audio.forward(output_proj.forward(x));
  }

  override transformStateDict(sd: LazyStateDict, prefix: string): void {
    transformEmbeddingGemma2AudioStateDict(sd, prefix, this.cfg);
  }
}
