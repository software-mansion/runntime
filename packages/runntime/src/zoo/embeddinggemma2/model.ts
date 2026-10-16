/** EmbeddingGemma 2 text model: a bidirectional Gemma encoder with
 *  per-layer embeddings (PLE), mean pooled and projected to 768.
 *
 *  Every layer is attention, a GeGLU MLP and a PLE block, each between a pre
 *  and a post RMSNorm. Five sliding layers (head 256, two kv heads) alternate
 *  with one full layer (head 512, one kv head). q, k and v are RMS-normed per
 *  head, so scores carry no 1/√d scale.
 *
 *  Built for few dispatches: q/k/v are one matmul, their per-head norms one
 *  grouped RMSNorm, and every post-norm also adds the residual. GELU rides
 *  the gate matmul's epilogue. Long inputs run attention per head as two
 *  matmuls around a masked softmax, short ones as one sdpa. */

import {
  cat,
  matmul,
  mul,
  nn,
  rmsNorm,
  rope,
  sdpa,
  slice,
  softmax,
  transpose,
  type LazyStateDict,
  type Value,
} from '../../core/index.ts';
import { EMBEDDINGGEMMA2_TEXT, type EmbeddingGemma2Config, type LayerType } from './config.ts';
import { transformEmbeddingGemma2StateDict } from './stateDictHooks.ts';

/** One forward's inputs: T rows, one or more inputs packed back to back. */
export interface EmbeddingGemma2Inputs {
  /** Embedded rows [T, hidden], from embedRows. */
  x0: Value;
  /** Rope rows [T, headDim] per layer kind, f32, picked by each row's position. */
  rope: Record<LayerType, { cos: Value; sin: Value }>;
  /** [B, T] mean-pool weights: row b holds 1/L on input b's tokens. */
  selector: Value;
  /** [T, 2] key range per row, so packed inputs never attend to each other. */
  segments: Value;
  /** Longest segment; sizes the attention key split. */
  maxSegment: number;
}

/** Longest-input length from which attention runs on matmuls. */
const MATMUL_ATTENTION_MIN_SEGMENT = 256;
/** Past this many rows a sliding layer keeps sdpa, which skips the keys
 *  outside its window. */
const MATMUL_ATTENTION_MAX_SLIDING_ROWS = 1024;

class Layer extends nn.Module<[Value, Value, EmbeddingGemma2Inputs], Value> {
  readonly input_layernorm: nn.RMSNorm;
  /** q | k | v rows, with q and k permuted for the interleaved rope. */
  readonly qkv: nn.Linear;
  /** q_norm tiled per q head, k_norm per kv head, ones for v. */
  readonly qkv_norm: nn.Parameter;
  readonly o_proj: nn.Linear;
  readonly post_attention_layernorm: nn.RMSNorm;
  readonly pre_feedforward_layernorm: nn.RMSNorm;
  readonly gate_proj: nn.Linear;
  readonly up_proj: nn.Linear;
  readonly down_proj: nn.Linear;
  readonly post_feedforward_layernorm: nn.RMSNorm;
  readonly per_layer_input_gate: nn.Linear;
  readonly per_layer_projection: nn.Linear;
  readonly post_per_layer_input_norm: nn.RMSNorm;
  /** One number that scales the layer's output; read on the CPU. */
  readonly layer_scalar: nn.CpuParameter;

  constructor(
    private readonly cfg: EmbeddingGemma2Config,
    readonly type: LayerType,
  ) {
    super();
    const { hidden, heads, ffn, pleDim, eps } = cfg;
    const { headDim, kvHeads } = cfg.attention[type];
    const width = (heads + 2 * kvHeads) * headDim;
    const norm = () => new nn.RMSNorm(hidden, eps);
    this.input_layernorm = norm();
    this.qkv = new nn.Linear(hidden, width, { bias: false });
    this.qkv_norm = new nn.Parameter({ elems: width, dtype: 'f32', dims: [width] });
    this.o_proj = new nn.Linear(heads * headDim, hidden, { bias: false });
    this.post_attention_layernorm = norm();
    this.pre_feedforward_layernorm = norm();
    this.gate_proj = new nn.Linear(hidden, ffn, { bias: false, activation: 'geluTanh' });
    this.up_proj = new nn.Linear(hidden, ffn, { bias: false });
    this.down_proj = new nn.Linear(ffn, hidden, { bias: false });
    this.post_feedforward_layernorm = norm();
    this.per_layer_input_gate = new nn.Linear(hidden, pleDim, {
      bias: false,
      activation: 'geluTanh',
    });
    this.per_layer_projection = new nn.Linear(pleDim, hidden, { bias: false });
    this.post_per_layer_input_norm = norm();
    this.layer_scalar = new nn.CpuParameter({ elems: 1, dtype: 'f32', dims: [1] });
  }

  override forward(x: Value, perLayer: Value, inp: EmbeddingGemma2Inputs): Value {
    const { heads, eps, slidingWindow } = this.cfg;
    const { headDim: d, kvHeads } = this.cfg.attention[this.type];
    const { cos, sin } = inp.rope[this.type];
    const window = this.type === 'sliding' ? slidingWindow : Infinity;
    const postNorm = (norm: nn.RMSNorm, y: Value, residual: Value, scale?: number) =>
      rmsNorm(y, norm.scale.value, norm.eps, { residual, scale });

    // Attention: qkv, one grouped norm for all heads, rope q and k. Columns
    // of n: q heads, then k heads, then v heads.
    const qkv = this.qkv.forward(this.input_layernorm.forward(x));
    const n = rmsNorm(qkv, this.qkv_norm.value, eps, { group: d });
    const roped = (start: number, width: number) =>
      rope(n, cos, sin, { headDim: d, srcStart: start, width });
    const vStart = (heads + kvHeads) * d;
    const rows = x.shape.dims![0]!;
    const perHead =
      inp.maxSegment >= MATMUL_ATTENTION_MIN_SEGMENT &&
      (this.type === 'full' || rows <= MATMUL_ATTENTION_MAX_SLIDING_ROWS);
    let a: Value;
    if (perHead) {
      // softmax(q·kᵀ) · v per head, the softmax masked like sdpa.
      const kT = Array.from({ length: kvHeads }, (_, h) => transpose(roped((heads + h) * d, d)));
      const v = Array.from({ length: kvHeads }, (_, h) =>
        slice(n, 1, vStart + h * d, vStart + (h + 1) * d),
      );
      const group = heads / kvHeads;
      const mask = { segments: inp.segments, windowLeft: window, windowRight: window };
      const outs = Array.from({ length: heads }, (_, h) => {
        const g = Math.floor(h / group);
        return matmul(softmax(matmul(roped(h * d, d), kT[g]!), mask), v[g]!);
      });
      a = cat(outs, 1);
    } else {
      a = sdpa(
        roped(0, heads * d),
        roped(heads * d, kvHeads * d),
        slice(n, 1, vStart, vStart + kvHeads * d),
        {
          qHeads: heads,
          kvHeads,
          headDim: d,
          windowLeft: window,
          windowRight: window,
          segments: inp.segments,
          maxSegment: inp.maxSegment,
        },
      );
    }
    x = postNorm(this.post_attention_layernorm, this.o_proj.forward(a), x);

    // GeGLU MLP, GELU in the gate matmul's epilogue.
    const f = this.pre_feedforward_layernorm.forward(x);
    const m = this.down_proj.forward(mul(this.gate_proj.forward(f), this.up_proj.forward(f)));
    x = postNorm(this.post_feedforward_layernorm, m, x);

    // PLE: gate against this layer's slice, project back, then scale the layer.
    const g = mul(this.per_layer_input_gate.forward(x), perLayer);
    const p = this.per_layer_projection.forward(g);
    return postNorm(this.post_per_layer_input_norm, p, x, this.layer_scalar.data[0]!);
  }
}

class Ple extends nn.Module<[Value], Value> {
  readonly per_layer_model_projection: nn.Linear;
  readonly per_layer_projection_norm: nn.RMSNorm;
  constructor(private readonly cfg: EmbeddingGemma2Config) {
    super();
    const { hidden, layers, pleDim, eps } = cfg;
    this.per_layer_model_projection = new nn.Linear(hidden, layers * pleDim, { bias: false });
    this.per_layer_projection_norm = new nn.RMSNorm(pleDim, eps);
  }
  /** [T, hidden] to [T, layers·pleDim], each layer's slice RMS-normed. The
   *  hidden^-½ scale is folded into the projection at load. */
  override forward(x: Value): Value {
    const norm = this.per_layer_projection_norm;
    return rmsNorm(this.per_layer_model_projection.forward(x), norm.scale.value, norm.eps, {
      group: this.cfg.pleDim,
    });
  }
}

export class EmbeddingGemma2Model extends nn.Module<[EmbeddingGemma2Inputs], Value> {
  readonly embed_tokens: nn.Embedding;
  readonly ple: Ple;
  readonly layers: nn.ModuleList<Layer>;
  readonly norm: nn.RMSNorm;
  readonly embedding_projection: nn.Linear;
  constructor(readonly cfg: EmbeddingGemma2Config = EMBEDDINGGEMMA2_TEXT) {
    super();
    // 262k rows of 512, and a call reads a handful: keep it on the CPU.
    this.embed_tokens = new nn.Embedding(cfg.vocab, cfg.hidden, { device: 'cpu' });
    this.ple = new Ple(cfg);
    this.layers = new nn.ModuleList(cfg.layerTypes.map((type) => new Layer(cfg, type)));
    this.norm = new nn.RMSNorm(cfg.hidden, cfg.eps);
    this.embedding_projection = new nn.Linear(cfg.hidden, cfg.embeddingDim, { bias: false });
  }

  /** Token ids to input rows [ids.length · hidden], scaled by √hidden,
   *  written into `out` at row `offset`. */
  embedRows(ids: readonly number[], out: Float32Array, offset: number): void {
    const rows = this.embed_tokens.gather(ids);
    const s = Math.sqrt(this.cfg.hidden);
    const base = offset * this.cfg.hidden;
    for (let i = 0; i < rows.length; i++) out[base + i] = rows[i]! * s;
  }

  /** Packed inputs to their pooled, projected embeddings [B, embeddingDim],
   *  not yet unit length. */
  override forward(inp: EmbeddingGemma2Inputs): Value {
    const perLayer = this.ple.forward(inp.x0);
    const pd = this.cfg.pleDim;
    let x = inp.x0;
    this.layers.items.forEach((layer, i) => {
      x = layer.forward(x, slice(perLayer, 1, i * pd, (i + 1) * pd), inp);
    });
    // Pooling before the projection: both are linear, and this projects B
    // rows instead of T.
    return this.embedding_projection.forward(matmul(inp.selector, this.norm.forward(x)));
  }

  override transformStateDict(sd: LazyStateDict, prefix: string): void {
    transformEmbeddingGemma2StateDict(sd, prefix, this.cfg);
  }
}
