/** Reverb: a bidirectional Qwen3 backbone that fills in masked Higgs codec
 *  tokens.
 *
 *  The sequence is a prefix (prompt tokens, then the reference frames) and
 *  the target frames. The prefix never attends to the target, so it runs once
 *  into a key/value cache. Each later pass runs only the target rows.
 *
 *  A layer is RMSNorm, attention, residual, RMSNorm, SwiGLU MLP, residual.
 *  Attention RMS-norms q and k per head and ropes them. Residuals ride the o
 *  and down matmuls. The modules are named like the checkpoint; the fused
 *  and centered weights are built at load (stateDictHooks). */

import {
  gatherRowsFrom,
  nn,
  rmsNorm,
  rope,
  sdpa,
  swigluChunk,
  writeRows,
  type LazyStateDict,
  type Value,
} from '../../core/index.ts';
import { COARSE_CODEBOOKS, REVERB, type ReverbConfig } from './config.ts';
import { headRows, transformReverbStateDict } from './stateDictHooks.ts';

const projection = (inF: number, outF: number) => new nn.Linear(inF, outF, { bias: false });

/** One layer's caller-owned key and value rows, [prefix + target, kvHeads·d]. */
interface LayerCache {
  k: Value;
  v: Value;
}

/** What one forward needs besides its input rows. */
export interface ReverbCall {
  /** Rope rows [T, headDim] for the rows this forward runs. */
  cos: Value;
  sin: Value;
  /** Per layer, where the forward's keys and values go. */
  cache: readonly LayerCache[];
  /** The cache row of this forward's first row: 0 for the prefix, the
   *  prefix length for a target pass. */
  start: number;
  /** Added to the attention scores of keys [start, end) in every layer and
   *  head. */
  keyBias?: { start: number; end: number; value: number };
  /** Collects the cache writes, for a forward that only fills the cache. */
  written?: Value[];
}

/** The attention projections, q and k fused at load. */
class Attention extends nn.Module {
  /** q heads then k heads. */
  readonly qk: nn.Linear;
  /** q_norm tiled over the q heads, then k_norm over the kv heads. */
  readonly qk_norm: nn.Parameter;
  readonly v_proj: nn.Linear;
  readonly o_proj: nn.Linear;
  constructor({ hidden, heads, kvHeads, headDim: d }: ReverbConfig) {
    super();
    const qkWidth = (heads + kvHeads) * d;
    this.qk = projection(hidden, qkWidth);
    this.qk_norm = new nn.Parameter({ elems: qkWidth, dtype: 'f32', dims: [qkWidth] });
    this.v_proj = projection(hidden, kvHeads * d);
    this.o_proj = projection(heads * d, hidden);
  }
}

/** SwiGLU MLP, up and gate fused at load. */
class Mlp extends nn.Module {
  /** up then gate. */
  readonly up_gate: nn.Linear;
  readonly down_proj: nn.Linear;
  constructor({ hidden, ffn }: ReverbConfig) {
    super();
    this.up_gate = projection(hidden, 2 * ffn);
    this.down_proj = projection(ffn, hidden);
  }
}

class Layer extends nn.Module<[Value, ReverbCall, number], Value> {
  readonly input_layernorm: nn.RMSNorm;
  readonly self_attn: Attention;
  readonly post_attention_layernorm: nn.RMSNorm;
  readonly mlp: Mlp;

  constructor(private readonly cfg: ReverbConfig) {
    super();
    this.input_layernorm = new nn.RMSNorm(cfg.hidden, cfg.eps);
    this.self_attn = new Attention(cfg);
    this.post_attention_layernorm = new nn.RMSNorm(cfg.hidden, cfg.eps);
    this.mlp = new Mlp(cfg);
  }

  override forward(x: Value, call: ReverbCall, index: number): Value {
    const { heads, kvHeads, headDim: d, eps } = this.cfg;
    const { self_attn: attn, mlp } = this;
    const h = this.input_layernorm.forward(x);
    const qk = rmsNorm(attn.qk.forward(h), attn.qk_norm.value, eps, { group: d });
    const roped = (start: number, width: number) =>
      rope(qk, call.cos, call.sin, { headDim: d, layout: 'half', srcStart: start, width });
    const q = roped(0, heads * d);
    const cache = call.cache[index]!;
    const k = writeRows(cache.k, roped(heads * d, kvHeads * d), call.start);
    const v = writeRows(cache.v, attn.v_proj.forward(h), call.start);
    call.written?.push(k, v);
    const a = sdpa(q, k, v, {
      qHeads: heads,
      kvHeads,
      headDim: d,
      windowLeft: Infinity,
      windowRight: Infinity,
      keyBias: call.keyBias,
    });
    x = attn.o_proj.forward(a, { addend: x });
    const m = swigluChunk(mlp.up_gate.forward(this.post_attention_layernorm.forward(x)));
    return mlp.down_proj.forward(m, { addend: x });
  }
}

/** The text embedding, factorized into a table and a projection. */
class EmbedTokens extends nn.Module {
  /** [vocab, rank] on the CPU; a prompt reads a few dozen rows. */
  readonly table: nn.Embedding;
  readonly project: nn.Linear;
  constructor({ vocab, embeddingRank, hidden }: ReverbConfig) {
    super();
    this.table = new nn.Embedding(vocab, embeddingRank, { device: 'cpu' });
    this.project = projection(embeddingRank, hidden);
  }
}

class LanguageModel extends nn.Module {
  readonly embed_tokens: EmbedTokens;
  readonly layers: nn.ModuleList<Layer>;
  readonly norm: nn.RMSNorm;
  constructor(cfg: ReverbConfig) {
    super();
    this.embed_tokens = new EmbedTokens(cfg);
    this.layers = new nn.ModuleList(Array.from({ length: cfg.layers }, () => new Layer(cfg)));
    this.norm = new nn.RMSNorm(cfg.hidden, cfg.eps);
  }
}

/** The checkpoint's audio heads, split at load into codebooks 0–1 and 2–7,
 *  rows centered per codebook, outputs padded with zeros to a multiple of 4
 *  for the vec4 matmul. */
class AudioHeads extends nn.Module {
  readonly coarse: nn.Linear;
  readonly fine: nn.Linear;
  constructor({ hidden, codebooks, codes }: ReverbConfig) {
    super();
    const head = (books: number) => projection(hidden, headRows(books, codes));
    this.coarse = head(COARSE_CODEBOOKS);
    this.fine = head(codebooks - COARSE_CODEBOOKS);
  }
}

export class ReverbModel extends nn.Module<[Value, ReverbCall], Value> {
  readonly llm: LanguageModel;
  /** [codebooks·codes, hidden]; a frame sums one row per codebook. */
  readonly audio_embeddings: nn.Embedding;
  readonly audio_heads: AudioHeads;

  constructor(readonly cfg: ReverbConfig = REVERB) {
    super();
    this.llm = new LanguageModel(cfg);
    this.audio_embeddings = new nn.Embedding(cfg.codebooks * cfg.codes, cfg.hidden);
    this.audio_heads = new AudioHeads(cfg);
  }

  override transformStateDict(sd: LazyStateDict, prefix: string): void {
    transformReverbStateDict(sd, prefix, this.cfg);
  }

  /** Table rows for codes [codebooks, frames], frame-major: frame t's
   *  codebook c at t·codebooks + c. */
  audioIds(codes: Int32Array, frames: number): Float32Array {
    const { codebooks, codes: perBook } = this.cfg;
    const ids = new Float32Array(codebooks * frames);
    for (let t = 0; t < frames; t++) {
      for (let c = 0; c < codebooks; c++) {
        ids[t * codebooks + c] = c * perBook + codes[c * frames + t]!;
      }
    }
    return ids;
  }

  /** Frame rows [frames, hidden] for audioIds: each sums one row per codebook. */
  audioRows(ids: Value): Value {
    return gatherRowsFrom(this.audio_embeddings.weight.value, ids, this.cfg.codebooks);
  }

  /** Rows x [T, hidden] through every layer, keys and values into the cache
   *  from call.start. Returns the normed hidden rows. */
  override forward(x: Value, call: ReverbCall): Value {
    this.llm.layers.items.forEach((layer, i) => {
      x = layer.forward(x, call, i);
    });
    return this.llm.norm.forward(x);
  }
}
