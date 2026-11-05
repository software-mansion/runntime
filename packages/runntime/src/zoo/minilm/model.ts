/** all-MiniLM-L6-v2: a post-LN BERT encoder built from core nn blocks, so
 *  LayerNorm runs after each residual add and every one carries a bias.
 *
 *  Attention is one fused qkv matmul sliced back into q/k/v, with an unbounded
 *  window both ways. The score scale is folded into the q and k projections at
 *  load, so the kernel applies none. Positions are a learned table with the
 *  constant token-type row folded in. Module fields are named like the
 *  checkpoint (BertModel), so `LayerNorm` is capitalized. */

import {
  add,
  materialized,
  matrix,
  nn,
  uploadF32,
  type LazyStateDict,
  type Value,
} from '../../core/index.ts';
import { MINILM_L6, type MinilmConfig } from './config.ts';

import { transformMinilmStateDict } from './stateDictHooks.ts';

class MinilmEmbeddings extends nn.Module<
  [readonly number[] | Value, (readonly number[] | Value)?],
  Value
> {
  readonly word_embeddings: nn.Embedding;
  /** The checkpoint's table plus its token-type row 0, summed at load. */
  readonly position_embeddings: nn.Embedding;
  readonly LayerNorm: nn.LayerNorm;
  constructor(cfg: MinilmConfig) {
    super();
    this.word_embeddings = new nn.Embedding(cfg.vocab, cfg.hidden);
    this.position_embeddings = new nn.Embedding(cfg.maxPositions, cfg.hidden);
    this.LayerNorm = new nn.LayerNorm(cfg.hidden, cfg.eps, { bias: true });
  }
  forward(ids: readonly number[], lengths?: readonly number[]): Value;
  forward(ids: Value, positions: Value): Value;
  forward(ids: readonly number[] | Value, second?: readonly number[] | Value): Value {
    if (!Array.isArray(ids)) {
      return this.LayerNorm.forward(
        add(
          this.word_embeddings.forward(ids as Value),
          this.position_embeddings.forward(second as Value),
        ),
      );
    }
    const lengths = second as readonly number[] | undefined;
    const positions = lengths
      ? lengths.flatMap((len) => Array.from({ length: len }, (_, i) => i))
      : Array.from({ length: ids.length }, (_, i) => i);
    return this.LayerNorm.forward(
      add(this.word_embeddings.forward(ids), this.position_embeddings.forward(positions)),
    );
  }
}

/** BertSelfAttention: q, k and v as one projection, scores through `mha`. */
class SelfAttention extends nn.Module {
  /** q | k | v rows, fused at load with the score scale split over q and k. */
  readonly qkv: nn.Linear;
  readonly mha: nn.MultiHeadAttention;
  constructor(cfg: MinilmConfig) {
    super();
    const qw = cfg.heads * cfg.headDim;
    this.qkv = new nn.Linear(cfg.hidden, 3 * qw);
    this.mha = new nn.MultiHeadAttention({
      qHeads: cfg.heads,
      kvHeads: cfg.heads,
      headDim: cfg.headDim,
      windowLeft: Infinity,
      windowRight: Infinity,
    });
  }
}

/** A projection and the post-LN after its residual add. */
class Output extends nn.Module<[Value, Value], Value> {
  readonly dense: nn.Linear;
  readonly LayerNorm: nn.LayerNorm;
  constructor(cfg: MinilmConfig, inFeatures: number) {
    super();
    this.dense = new nn.Linear(inFeatures, cfg.hidden);
    this.LayerNorm = new nn.LayerNorm(cfg.hidden, cfg.eps, { bias: true });
  }
  forward(x: Value, residual: Value): Value {
    return this.LayerNorm.forward(this.dense.forward(x, { addend: residual }));
  }
}

class Attention extends nn.Module {
  readonly self: SelfAttention;
  readonly output: Output;
  constructor(cfg: MinilmConfig) {
    super();
    this.self = new SelfAttention(cfg);
    this.output = new Output(cfg, cfg.heads * cfg.headDim);
  }
}

class Intermediate extends nn.Module {
  readonly dense: nn.Linear;
  constructor(cfg: MinilmConfig) {
    super();
    this.dense = new nn.Linear(cfg.hidden, cfg.ffn, { activation: 'gelu' });
  }
}

class MinilmLayer extends nn.Module<[Value, Value?, number?], Value> {
  readonly attention: Attention;
  readonly intermediate: Intermediate;
  readonly output: Output;
  constructor(cfg: MinilmConfig) {
    super();
    this.attention = new Attention(cfg);
    this.intermediate = new Intermediate(cfg);
    this.output = new Output(cfg, cfg.ffn);
  }
  override forward(x: Value, segments?: Value, maxSegment?: number): Value {
    const { self, output } = this.attention;
    const qkv = self.qkv.forward(x); // [T, 3·qw], columns q|k|v read in place
    const h = output.forward(self.mha.forwardPacked(qkv, { segments, maxSegment }), x);
    return this.output.forward(this.intermediate.dense.forward(h), h);
  }
}

class MinilmEncoder extends nn.Module {
  readonly layer: nn.ModuleList<MinilmLayer>;
  constructor(cfg: MinilmConfig) {
    super();
    this.layer = new nn.ModuleList(Array.from({ length: cfg.layers }, () => new MinilmLayer(cfg)));
  }
}

export interface MinilmReplayInputs {
  positions: Value;
  segments: Value;
  maxSegment: number;
}

export class MinilmModel extends nn.Module<
  [readonly number[] | Value, (readonly number[] | MinilmReplayInputs)?],
  Value
> {
  readonly embeddings: MinilmEmbeddings;
  readonly encoder: MinilmEncoder;
  constructor(readonly cfg: MinilmConfig = MINILM_L6) {
    super();
    this.embeddings = new MinilmEmbeddings(cfg);
    this.encoder = new MinilmEncoder(cfg);
  }
  override forward(ids: readonly number[], lengths?: readonly number[]): Value;
  override forward(ids: Value, replay: MinilmReplayInputs): Value;
  override forward(
    ids: readonly number[] | Value,
    second?: readonly number[] | MinilmReplayInputs,
  ): Value {
    if (!Array.isArray(ids)) {
      const { positions, segments, maxSegment } = second as MinilmReplayInputs;
      let h: Value = this.embeddings.forward(ids as Value, positions);
      for (const layer of this.encoder.layer.items) h = layer.forward(h, segments, maxSegment);
      return h;
    }
    const lengths = second as readonly number[] | undefined;
    let segments: Value | undefined;
    let maxSegment: number | undefined;
    if (lengths) {
      const total = lengths.reduce((s, l) => s + l, 0);
      if (total !== ids.length) {
        throw new Error(`minilm: lengths sum ${total} != packed ids length ${ids.length}`);
      }
      const segs = new Float32Array(ids.length * 2);
      let start = 0;
      for (const len of lengths) {
        if (len < 1) throw new Error('minilm: every sentence needs at least one token');
        for (let i = 0; i < len; i++) {
          segs[2 * (start + i)] = start;
          segs[2 * (start + i) + 1] = start + len;
        }
        start += len;
      }
      segments = materialized(matrix(ids.length, 2), uploadF32(segs));
      // The longest sentence, not the pack length, drives the key split.
      maxSegment = Math.max(...lengths);
    }
    let h: Value = this.embeddings.forward(ids, lengths);
    for (const layer of this.encoder.layer.items) h = layer.forward(h, segments, maxSegment);
    return h;
  }
  override transformStateDict(sd: LazyStateDict, prefix: string): void {
    transformMinilmStateDict(sd, prefix, this.cfg);
  }
}
