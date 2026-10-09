/** EmbeddingGemma 2, text tower: a bidirectional Gemma encoder with
 *  per-layer embeddings (PLE), mean pooled and projected to 768.
 *
 *  Every layer is attention, a gated-GELU MLP and a PLE block, each wrapped
 *  in pre and post RMSNorm. Five sliding layers (head 256, two kv heads,
 *  theta 1e4) alternate with one global layer (head 512, one kv head, theta
 *  1e6). q, k and v are RMS-normed per head, so scores carry no 1/√d scale.
 *
 *  Built for few dispatches, like MiniLM: q/k/v are one matmul, their three
 *  per-head norms one grouped RMSNorm, attention one fused sdpa, and every
 *  post-norm block ends in one RMSNorm that also adds the residual. GELU
 *  rides the matmul epilogue. Inputs are capped at the 512-token sliding
 *  window, which makes every layer full attention.
 *
 *  Runs in f32 only: the model card warns its activations overflow f16. */

import {
  cat,
  matmul,
  mul,
  nn,
  rope,
  sdpa,
  slice,
  softmax,
  tensor,
  transpose,
  type LazyStateDict,
  type Value,
} from '../../core/index.ts';
import { rmsNormFused } from '../../core/graph/ops/reduce.ts';
import { transformEmbeddingGemma2StateDict } from './stateDictHooks.ts';

export interface EmbeddingGemma2Config {
  vocab: number;
  hidden: number;
  ffn: number;
  layers: number;
  heads: number;
  pleDim: number;
  embeddingDim: number;
  eps: number;
  /** Per layer: head size, kv heads and rope theta. */
  layerTypes: readonly ('sliding' | 'full')[];
}

export const EMBEDDINGGEMMA2_TEXT: EmbeddingGemma2Config = {
  vocab: 262144,
  hidden: 512,
  ffn: 2048,
  layers: 24,
  heads: 4,
  pleDim: 512,
  embeddingDim: 768,
  eps: 1e-6,
  layerTypes: Array.from({ length: 24 }, (_, i) => (i % 6 === 5 ? 'full' : 'sliding')),
};

/** The sliding window; longer inputs would need the band mask. */
export const EMBEDDINGGEMMA2_MAX_TOKENS = 512;

export const ATTN = {
  sliding: { headDim: 256, kvHeads: 2, theta: 10000 },
  full: { headDim: 512, kvHeads: 1, theta: 1000000 },
} as const;

/** Full bidirectional attention with no score scale, as matmul, softmax,
 *  matmul per head. The vision tower uses it at 2.4k patches, where the
 *  tiled matmul beats the fused f16 sdpa route. */
export function headwiseAttention(
  q: Value,
  k: Value,
  v: Value,
  heads: number,
  kvHeads: number,
  d: number,
): Value {
  const group = heads / kvHeads;
  const kT = Array.from({ length: kvHeads }, (_, h) => transpose(slice(k, 1, h * d, (h + 1) * d)));
  const vs = Array.from({ length: kvHeads }, (_, h) => slice(v, 1, h * d, (h + 1) * d));
  const outs = Array.from({ length: heads }, (_, h) => {
    const kv = Math.floor(h / group);
    const p = softmax(matmul(slice(q, 1, h * d, (h + 1) * d), kT[kv]!));
    return matmul(p, vs[kv]!);
  });
  return cat(outs, 1);
}

/** Pair-duplicated rope tables for the interleaved kernel. The q and k
 *  weights are permuted at load so the kernel's pairs are HF's half split. */
export function ropeTables(tokens: number, headDim: number, theta: number) {
  const cos = new Float32Array(tokens * headDim);
  const sin = new Float32Array(tokens * headDim);
  for (let t = 0; t < tokens; t++) {
    for (let p = 0; p < headDim / 2; p++) {
      const angle = t / theta ** ((2 * p) / headDim);
      cos[t * headDim + 2 * p] = cos[t * headDim + 2 * p + 1] = Math.cos(angle);
      sin[t * headDim + 2 * p] = sin[t * headDim + 2 * p + 1] = Math.sin(angle);
    }
  }
  return { cos, sin };
}

/** What one forward needs beyond the input rows. Built once per length and
 *  reused, so a replay records them as constants. */
export interface TextForwardCtx {
  tables: Record<'sliding' | 'full', { cos: Value; sin: Value }>;
  /** [1, T] mean-pool weights: 1/L on real tokens, 0 on padding. */
  selector: Value;
  /** [T, 2] key range per row, so padding never mixes into real tokens. */
  segments?: Value;
  maxSegment?: number;
}

/** Context for an unpadded input of `t` tokens. */
export function plainCtx(t: number): TextForwardCtx {
  const table = (type: 'sliding' | 'full') => {
    const { headDim, theta } = ATTN[type];
    const { cos, sin } = ropeTables(t, headDim, theta);
    return { cos: tensor(cos, [t, headDim]), sin: tensor(sin, [t, headDim]) };
  };
  return {
    tables: { sliding: table('sliding'), full: table('full') },
    selector: tensor(new Float32Array(t).fill(1 / t), [1, t]),
  };
}

class Layer extends nn.Module<[Value, Value, TextForwardCtx], Value> {
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
  readonly headDim: number;
  readonly kvHeads: number;

  constructor(
    private readonly cfg: EmbeddingGemma2Config,
    readonly type: 'sliding' | 'full',
  ) {
    super();
    const { hidden, heads, ffn, pleDim, eps } = cfg;
    const { headDim: d, kvHeads } = ATTN[type];
    this.headDim = d;
    this.kvHeads = kvHeads;
    const width = (heads + 2 * kvHeads) * d;
    const norm = () => new nn.RMSNorm(hidden, eps);
    this.input_layernorm = norm();
    this.qkv = new nn.Linear(hidden, width, { bias: false });
    this.qkv_norm = new nn.Parameter({ elems: width, dtype: 'f32', dims: [width] });
    this.o_proj = new nn.Linear(heads * d, hidden, { bias: false });
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

  override forward(x: Value, perLayer: Value, ctx: TextForwardCtx): Value {
    const { heads, eps } = this.cfg;
    const { headDim: d, kvHeads } = this;
    const { cos, sin } = ctx.tables[this.type];
    const postNorm = (norm: nn.RMSNorm, y: Value, residual: Value, scale?: number) =>
      rmsNormFused(y, norm.scale.value, norm.eps, { residual, scale });

    // Attention: qkv, one grouped norm for all heads, rope q and k, sdpa.
    const qkv = this.qkv.forward(this.input_layernorm.forward(x));
    const n = rmsNormFused(qkv, this.qkv_norm.value, eps, { group: d });
    const q = rope(n, cos, sin, { headDim: d, srcStart: 0, width: heads * d });
    const k = rope(n, cos, sin, { headDim: d, srcStart: heads * d, width: kvHeads * d });
    const v = slice(n, 1, (heads + kvHeads) * d, (heads + 2 * kvHeads) * d);
    const a = sdpa(q, k, v, {
      qHeads: heads,
      kvHeads,
      headDim: d,
      windowLeft: Infinity,
      windowRight: Infinity,
      segments: ctx.segments,
      maxSegment: ctx.maxSegment,
    });
    x = postNorm(this.post_attention_layernorm, this.o_proj.forward(a), x);

    // Gated-GELU MLP, GELU in the gate matmul's epilogue.
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
    return rmsNormFused(this.per_layer_model_projection.forward(x), norm.scale.value, norm.eps, {
      group: this.cfg.pleDim,
    });
  }
}

export class EmbeddingGemma2TextModel extends nn.Module<[readonly number[]], Value> {
  readonly embed_tokens: nn.Embedding;
  readonly ple: Ple;
  readonly layers: nn.ModuleList<Layer>;
  readonly norm: nn.RMSNorm;
  readonly embedding_projection: nn.Linear;
  constructor(readonly cfg: EmbeddingGemma2Config = EMBEDDINGGEMMA2_TEXT) {
    super();
    // 512 MB as f32, and a call reads a handful of rows: keep it on the CPU.
    this.embed_tokens = new nn.Embedding(cfg.vocab, cfg.hidden, { device: 'cpu' });
    this.ple = new Ple(cfg);
    this.layers = new nn.ModuleList(cfg.layerTypes.map((type) => new Layer(cfg, type)));
    this.norm = new nn.RMSNorm(cfg.hidden, cfg.eps);
    this.embedding_projection = new nn.Linear(cfg.hidden, cfg.embeddingDim, { bias: false });
  }

  /** Eager: token ids to the pooled, projected embedding [1, embeddingDim],
   *  not yet unit length. */
  override forward(ids: readonly number[]): Value {
    const t = ids.length;
    return this.forwardEmbeds(tensor(this.embedRows(ids), [t, this.cfg.hidden]), plainCtx(t));
  }

  /** Token ids to scaled input rows [t·hidden] on the CPU, ready to upload
   *  or write into a replay plane. */
  embedRows(ids: readonly number[], out?: Float32Array, offset = 0): Float32Array {
    const rows = this.embed_tokens.gather(ids);
    const s = Math.sqrt(this.cfg.hidden);
    const dst = out ?? new Float32Array(rows.length);
    for (let i = 0; i < rows.length; i++) dst[offset + i] = rows[i]! * s;
    return dst;
  }

  /** Input rows [T, hidden] (text, or image soft tokens) to the pooled,
   *  projected embedding [1, embeddingDim]. */
  forwardEmbeds(x0: Value, ctx: TextForwardCtx): Value {
    const t = x0.shape.dims![0]!;
    if (t < 1 || t > EMBEDDINGGEMMA2_MAX_TOKENS) {
      throw new Error(`embeddinggemma2: ${t} tokens, expected 1..${EMBEDDINGGEMMA2_MAX_TOKENS}`);
    }
    const perLayer = this.ple.forward(x0);
    const pd = this.cfg.pleDim;
    let x = x0;
    this.layers.items.forEach((layer, i) => {
      x = layer.forward(x, slice(perLayer, 1, i * pd, (i + 1) * pd), ctx);
    });
    return this.embedding_projection.forward(matmul(ctx.selector, this.norm.forward(x)));
  }

  override transformStateDict(sd: LazyStateDict, prefix: string): void {
    transformEmbeddingGemma2StateDict(sd, prefix, this.cfg);
  }
}
