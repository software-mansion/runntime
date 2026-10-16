/** EmbeddingGemma 2 audio tower: a conformer over a clip's log-mel frames,
 *  one token per 40 ms, projected to the text model's width.
 *
 *  Two stride-2 convs shrink time and frequency 4×. Each layer is a
 *  half-weighted feed-forward, local attention over this token and the 11
 *  before it, a gated causal depthwise conv, a second feed-forward and a
 *  norm. Most linears clamp their input and output to bounds from the
 *  checkpoint. */

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

/** A linear that clamps its input and output to the checkpoint's bounds. */
class ClippedLinear extends nn.Linear {
  /** [inMin, inMax, outMin, outMax], read on the CPU. */
  readonly clip = new nn.CpuParameter({ elems: 4, dtype: 'f32', dims: [4] });

  constructor(inFeatures: number, outFeatures: number) {
    super(inFeatures, outFeatures, { bias: false });
  }

  /** Bounds of the output. */
  get outBounds(): { lo: number; hi: number } {
    const [, , lo = -Infinity, hi = Infinity] = this.clip.data;
    return { lo, hi };
  }

  /** The matmul on the clamped input, its output left unclamped. */
  unclamped(x: Value): Value {
    const [lo = -Infinity, hi = Infinity] = this.clip.data;
    return super.forward(clamp(x, lo, hi));
  }

  override forward(x: Value): Value {
    const { lo, hi } = this.outBounds;
    return clamp(this.unclamped(x), lo, hi);
  }
}

class FeedForward extends nn.Module<[Value], Value> {
  readonly pre_layer_norm: nn.RMSNorm;
  readonly ffw_layer_1: ClippedLinear;
  readonly ffw_layer_2: ClippedLinear;
  /** Its scale carries the residual weight. */
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
    return rmsNorm(this.ffw_layer_2.forward(h), post.scale.value, post.eps, { residual: x });
  }
}

class LightConv extends nn.Module<[Value], Value> {
  readonly pre_layer_norm: nn.RMSNorm;
  /** The two GLU halves, value then gate. */
  readonly linear_start: ClippedLinear;
  /** Depthwise taps [convKernel, hidden], oldest first. */
  readonly conv: nn.Parameter;
  readonly conv_norm: nn.RMSNorm;
  readonly linear_end: ClippedLinear;

  constructor({ hidden, eps, convKernel }: EmbeddingGemma2AudioConfig) {
    super();
    this.pre_layer_norm = new nn.RMSNorm(hidden, eps);
    this.linear_start = new ClippedLinear(hidden, 2 * hidden);
    this.conv = new nn.Parameter({
      elems: convKernel * hidden,
      dtype: 'f32',
      dims: [convKernel, hidden],
    });
    this.conv_norm = new nn.RMSNorm(hidden, eps);
    this.linear_end = new ClippedLinear(hidden, hidden);
  }

  override forward(x: Value): Value {
    const z = this.linear_start.unclamped(this.pre_layer_norm.forward(x));
    const g = gluConv1d(z, this.conv.value, this.linear_start.outBounds);
    return add(this.linear_end.forward(silu(this.conv_norm.forward(g))), x);
  }
}

class AudioLayer extends nn.Module<[Value], Value> {
  readonly feed_forward1: FeedForward;
  readonly norm_pre_attn: nn.RMSNorm;
  /** q | k | v rows. */
  readonly qkv: ClippedLinear;
  /** One bias row per key distance [window, hidden], f32. */
  readonly rel: nn.Parameter;
  /** Scale of each column of q [headDim], f32. */
  readonly q_scale: nn.Parameter;
  readonly post: ClippedLinear;
  readonly norm_post_attn: nn.RMSNorm;
  readonly lconv: LightConv;
  readonly feed_forward2: FeedForward;
  readonly norm_out: nn.RMSNorm;

  constructor(private readonly cfg: EmbeddingGemma2AudioConfig) {
    super();
    const { hidden, headDim, window, eps } = cfg;
    // Pinned: the attention kernel reads them as f32.
    const f32 = (...dims: number[]) =>
      new nn.Parameter(
        { elems: dims.reduce((a, b) => a * b), dtype: 'f32', dims },
        undefined,
        true,
      );
    this.feed_forward1 = new FeedForward(cfg);
    this.norm_pre_attn = new nn.RMSNorm(hidden, eps);
    this.qkv = new ClippedLinear(hidden, 3 * hidden);
    this.rel = f32(window, hidden);
    this.q_scale = f32(headDim);
    this.post = new ClippedLinear(hidden, hidden);
    this.norm_post_attn = new nn.RMSNorm(hidden, eps);
    this.lconv = new LightConv(cfg);
    this.feed_forward2 = new FeedForward(cfg);
    this.norm_out = new nn.RMSNorm(hidden, eps);
  }

  override forward(x: Value): Value {
    const { heads, headDim, window, logitCap } = this.cfg;
    x = this.feed_forward1.forward(x);
    const qkv = this.qkv.forward(this.norm_pre_attn.forward(x));
    const a = localAttention(qkv, this.rel.value, this.q_scale.value, {
      heads,
      headDim,
      window,
      kScale: K_SCALE,
      softcap: logitCap,
    });
    const norm = this.norm_post_attn;
    x = rmsNorm(this.post.forward(a), norm.scale.value, norm.eps, { residual: x });
    x = this.lconv.forward(x);
    return this.norm_out.forward(this.feed_forward2.forward(x));
  }
}

export class EmbeddingGemma2AudioModel extends nn.Module<[EmbeddingGemma2AudioInputs], Value> {
  readonly conv0: nn.Linear;
  readonly norm0: nn.LayerNorm;
  readonly conv1: nn.Linear;
  readonly norm1: nn.LayerNorm;
  readonly input_proj: nn.Linear;
  readonly layers: nn.ModuleList<AudioLayer>;
  readonly output_proj: nn.Linear;
  readonly embedding_pre_projection_norm: nn.RMSNorm;
  readonly embedding_projection: nn.Linear;

  constructor(readonly cfg: EmbeddingGemma2AudioConfig = EMBEDDINGGEMMA2_AUDIO) {
    super();
    const { hidden, mels, eps, outputDim } = cfg;
    const [c0, c1] = cfg.convChannels;
    this.conv0 = new nn.Linear(9, c0, { bias: false });
    this.norm0 = new nn.LayerNorm(c0, eps);
    this.conv1 = new nn.Linear(9 * c0, c1, { bias: false });
    this.norm1 = new nn.LayerNorm(c1, eps);
    this.input_proj = new nn.Linear((mels / 4) * c1, hidden, { bias: false });
    this.layers = new nn.ModuleList(Array.from({ length: cfg.layers }, () => new AudioLayer(cfg)));
    this.output_proj = new nn.Linear(hidden, outputDim);
    this.embedding_pre_projection_norm = new nn.RMSNorm(outputDim, eps);
    this.embedding_projection = new nn.Linear(outputDim, EMBEDDINGGEMMA2_TEXT.hidden, {
      bias: false,
    });
  }

  /** One clip to its tokens [tokens, text hidden], ready to stand in for
   *  the placeholder rows of the text model's input. */
  override forward(inp: EmbeddingGemma2AudioInputs): Value {
    const [c0, c1] = this.cfg.convChannels;
    const relu = (v: Value) => clamp(v, 0, Infinity);
    // Each conv is a matmul over its taps, LayerNorm over channels, ReLU.
    const y0 = relu(this.norm0.forward(this.conv0.forward(inp.patches)));
    const pixels = inp.taps.shape.elems / 9;
    const y1 = reshape(gatherRowsFrom(y0, inp.taps), [pixels, 9 * c0]);
    const y = relu(this.norm1.forward(this.conv1.forward(y1)));
    // Rows are (token, frequency); each token's frequencies become one row.
    let x = this.input_proj.forward(reshape(y, [inp.tokens, (pixels / inp.tokens) * c1]));
    for (const layer of this.layers.items) x = layer.forward(x);
    const out = this.embedding_pre_projection_norm.forward(this.output_proj.forward(x));
    return this.embedding_projection.forward(out);
  }

  override transformStateDict(sd: LazyStateDict, prefix: string): void {
    transformEmbeddingGemma2AudioStateDict(sd, prefix, this.cfg);
  }
}
