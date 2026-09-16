/** all-MiniLM-L6-v2: a post-LN BERT encoder built from core nn blocks, so
 *  LayerNorm runs after each residual add and every one carries a bias.
 *
 *  Attention is one fused qkv matmul sliced back into q/k/v, with an unbounded
 *  window both ways. The score scale is folded into the q and k projections at
 *  load, so the kernel applies none. Positions are a learned table with the
 *  constant token-type row folded in. */

import {
  add,
  gelu,
  materialized,
  matrix,
  nn,
  slice,
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
  readonly word: nn.Embedding;
  readonly position: nn.Embedding;
  readonly norm: nn.LayerNorm;
  constructor(cfg: MinilmConfig) {
    super();
    this.word = new nn.Embedding(cfg.vocab, cfg.hidden);
    this.position = new nn.Embedding(cfg.maxPositions, cfg.hidden);
    this.norm = new nn.LayerNorm(cfg.hidden, cfg.eps, { bias: true });
  }
  forward(ids: readonly number[], lengths?: readonly number[]): Value;
  forward(ids: Value, positions: Value): Value;
  forward(ids: readonly number[] | Value, second?: readonly number[] | Value): Value {
    if (!Array.isArray(ids)) {
      return this.norm.forward(
        add(this.word.forward(ids as Value), this.position.forward(second as Value)),
      );
    }
    const lengths = second as readonly number[] | undefined;
    const positions = lengths
      ? lengths.flatMap((len) => Array.from({ length: len }, (_, i) => i))
      : Array.from({ length: ids.length }, (_, i) => i);
    return this.norm.forward(add(this.word.forward(ids), this.position.forward(positions)));
  }
}

class MinilmLayer extends nn.Module<[Value, Value?, number?], Value> {
  readonly qkv: nn.Linear;
  readonly mha: nn.MultiHeadAttention;
  readonly out: nn.Linear;
  readonly attnNorm: nn.LayerNorm;
  readonly up: nn.Linear;
  readonly down: nn.Linear;
  readonly ffnNorm: nn.LayerNorm;
  constructor(private readonly cfg: MinilmConfig) {
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
    this.out = new nn.Linear(qw, cfg.hidden);
    this.attnNorm = new nn.LayerNorm(cfg.hidden, cfg.eps, { bias: true });
    this.up = new nn.Linear(cfg.hidden, cfg.ffn, { activation: 'gelu' });
    this.down = new nn.Linear(cfg.ffn, cfg.hidden);
    this.ffnNorm = new nn.LayerNorm(cfg.hidden, cfg.eps, { bias: true });
  }
  override forward(x: Value, segments?: Value, maxSegment?: number): Value {
    const qkv = this.qkv.forward(x); // [T, 3·qw], columns q|k|v read in place
    const attn = this.mha.forwardPacked(qkv, { segments, maxSegment });
    const h = this.attnNorm.forward(this.out.forward(attn, { addend: x }));
    return this.ffnNorm.forward(this.down.forward(this.up.forward(h), { addend: h }));
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
  readonly layers: nn.ModuleList<MinilmLayer>;
  constructor(readonly cfg: MinilmConfig = MINILM_L6) {
    super();
    this.embeddings = new MinilmEmbeddings(cfg);
    this.layers = new nn.ModuleList(Array.from({ length: cfg.layers }, () => new MinilmLayer(cfg)));
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
      for (const layer of this.layers.items) h = layer.forward(h, segments, maxSegment);
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
    for (const layer of this.layers.items) h = layer.forward(h, segments, maxSegment);
    return h;
  }
  override transformStateDict(sd: LazyStateDict, prefix: string): void {
    transformMinilmStateDict(sd, prefix, this.cfg);
  }
}
