/** Reverb: a bidirectional Qwen3 backbone that fills in masked Higgs codec
 *  tokens.
 *
 *  The sequence is a prefix (prompt tokens, then the reference frames) and
 *  the target frames. The prefix never attends to the target, so it runs once
 *  into a key/value cache. Each later pass runs only the target rows.
 *
 *  A layer is RMSNorm, attention, residual, RMSNorm, SwiGLU MLP, residual.
 *  Attention RMS-norms q and k per head and ropes them. Residuals ride the o
 *  and down matmuls. The checkpoint maps on at load (stateDictHooks). */

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

class Layer extends nn.Module<[Value, ReverbCall, number], Value> {
  readonly input_norm: nn.RMSNorm;
  /** q heads then k heads, rows permuted for the interleaved rope. */
  readonly qk: nn.Linear;
  /** q_norm·d^-½ tiled over the q heads, then k_norm over the kv heads. */
  readonly qk_norm: nn.Parameter;
  readonly v: nn.Linear;
  readonly o: nn.Linear;
  readonly post_norm: nn.RMSNorm;
  /** up then gate. */
  readonly up_gate: nn.Linear;
  readonly down: nn.Linear;

  constructor(private readonly cfg: ReverbConfig) {
    super();
    const { hidden, ffn, heads, kvHeads, headDim: d, eps } = cfg;
    const qkWidth = (heads + kvHeads) * d;
    this.input_norm = new nn.RMSNorm(hidden, eps);
    this.qk = projection(hidden, qkWidth);
    this.qk_norm = new nn.Parameter({ elems: qkWidth, dtype: 'f32', dims: [qkWidth] });
    this.v = projection(hidden, kvHeads * d);
    this.o = projection(heads * d, hidden);
    this.post_norm = new nn.RMSNorm(hidden, eps);
    this.up_gate = projection(hidden, 2 * ffn);
    this.down = projection(ffn, hidden);
  }

  override forward(x: Value, call: ReverbCall, index: number): Value {
    const { heads, kvHeads, headDim: d, eps } = this.cfg;
    const h = this.input_norm.forward(x);
    const qk = rmsNorm(this.qk.forward(h), this.qk_norm.value, eps, { group: d });
    const roped = (start: number, width: number) =>
      rope(qk, call.cos, call.sin, { headDim: d, srcStart: start, width });
    const q = roped(0, heads * d);
    const cache = call.cache[index]!;
    const k = writeRows(cache.k, roped(heads * d, kvHeads * d), call.start);
    const v = writeRows(cache.v, this.v.forward(h), call.start);
    call.written?.push(k, v);
    const a = sdpa(q, k, v, {
      qHeads: heads,
      kvHeads,
      headDim: d,
      windowLeft: Infinity,
      windowRight: Infinity,
      keyBias: call.keyBias,
    });
    x = this.o.forward(a, { addend: x });
    const m = swigluChunk(this.up_gate.forward(this.post_norm.forward(x)));
    return this.down.forward(m, { addend: x });
  }
}

export class ReverbModel extends nn.Module<[Value, ReverbCall], Value> {
  /** [vocab, rank] on the CPU; a prompt reads a few dozen rows. */
  readonly text_table: nn.Embedding;
  readonly text_project: nn.Linear;
  /** [codebooks·codes, hidden]; a frame sums one row per codebook. */
  readonly audio_table: nn.Embedding;
  readonly layers: nn.ModuleList<Layer>;
  readonly norm: nn.RMSNorm;
  /** Codebooks 0–1 and 2–7, rows centered per codebook, outputs padded
   *  with zeros to a multiple of 4 for the vec4 matmul. */
  readonly heads_coarse: nn.Linear;
  readonly heads_fine: nn.Linear;

  constructor(readonly cfg: ReverbConfig = REVERB) {
    super();
    const { vocab, embeddingRank, hidden, codebooks, codes } = cfg;
    this.text_table = new nn.Embedding(vocab, embeddingRank, { device: 'cpu' });
    this.text_project = projection(embeddingRank, hidden);
    this.audio_table = new nn.Embedding(codebooks * codes, hidden);
    this.layers = new nn.ModuleList(Array.from({ length: cfg.layers }, () => new Layer(cfg)));
    this.norm = new nn.RMSNorm(hidden, cfg.eps);
    const head = (books: number) => projection(hidden, headRows(books, codes));
    this.heads_coarse = head(COARSE_CODEBOOKS);
    this.heads_fine = head(codebooks - COARSE_CODEBOOKS);
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
    return gatherRowsFrom(this.audio_table.weight.value, ids, this.cfg.codebooks);
  }

  /** Rows x [T, hidden] through every layer, keys and values into the cache
   *  from call.start. Returns the normed hidden rows. */
  override forward(x: Value, call: ReverbCall): Value {
    this.layers.items.forEach((layer, i) => {
      x = layer.forward(x, call, i);
    });
    return this.norm.forward(x);
  }
}
