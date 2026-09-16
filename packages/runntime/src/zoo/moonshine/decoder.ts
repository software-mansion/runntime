/** Moonshine decoder: pre-norm layers with three sublayers each.
 *
 *      LN -> causal MHA self-attention (partial rope) -> residual
 *      LN -> cross-attention over the encoder output  -> residual
 *      LN -> SwiGLU MLP                               -> residual
 *
 *  followed by a final weight-only LN.
 *
 *  Two drive modes: forward() runs all t tokens at once for teacher forcing and
 *  parity tests, while forwardStep() feeds one token against the per-layer
 *  cache, so per-step cost is independent of the sequence length.
 *
 *  Score scales are folded into the weights and rope tables, so none appears
 *  here. Cross K and V are computed once per utterance and reused by every
 *  decode step. */

import { chunk, nn, rope, slice, swigluChunk, writeRows, type Value } from '../../core/index.ts';
import type { MoonshineConfig } from './config.ts';

export interface CrossKV {
  k: Value;
  v: Value;
}

export interface SelfKV {
  k: Value;
  v: Value;
}

class DecoderSelfAttn extends nn.Module<[Value, Value, Value, Value?], Value> {
  readonly qkv: nn.Linear;
  readonly mha: nn.MultiHeadAttention;
  readonly out: nn.Linear;
  constructor(private readonly cfg: MoonshineConfig) {
    super();
    const qw = cfg.heads * cfg.headDim;
    this.qkv = new nn.Linear(cfg.dModel, 3 * qw, { bias: false });
    this.mha = new nn.MultiHeadAttention({
      qHeads: cfg.heads,
      kvHeads: cfg.heads,
      headDim: cfg.headDim,
      windowLeft: Infinity,
      windowRight: 0, // causal
    });
    this.out = new nn.Linear(qw, cfg.dModel, { bias: false });
  }
  forward(x: Value, cos: Value, sin: Value, residual?: Value): Value {
    const { heads, headDim } = this.cfg;
    const qw = heads * headDim;
    const qkv = this.qkv.forward(x); // [t, 3·qw], columns q|k|v
    const q = rope(qkv, cos, sin, { headDim, srcStart: 0, width: qw });
    const k = rope(qkv, cos, sin, { headDim, srcStart: qw, width: qw });
    const v = slice(qkv, 1, 2 * qw, 3 * qw);
    return this.out.forward(this.mha.forward(q, k, v), { addend: residual });
  }
  forwardStep(
    x1: Value,
    cosRow: Value,
    sinRow: Value,
    cache: SelfKV,
    t: number,
    residual?: Value,
  ): Value {
    const { heads, headDim } = this.cfg;
    const qw = heads * headDim;
    const qkv = this.qkv.forward(x1); // [1, 3·qw]
    const q = rope(qkv, cosRow, sinRow, { headDim, srcStart: 0, width: qw });
    const kNew = rope(qkv, cosRow, sinRow, { headDim, srcStart: qw, width: qw });
    const vNew = slice(qkv, 1, 2 * qw, 3 * qw);
    const k = writeRows(cache.k, kNew, t); // [t+1, qw] view of the cache buffer
    const v = writeRows(cache.v, vNew, t);
    const ctx = this.mha.forward(q, k, v, { qPosOffset: t });
    return this.out.forward(ctx, { addend: residual });
  }
}

class DecoderCrossAttn extends nn.Module<[Value, CrossKV, Value?], Value> {
  readonly q: nn.Linear;
  readonly kv: nn.Linear;
  readonly mha: nn.MultiHeadAttention;
  readonly out: nn.Linear;
  constructor(cfg: MoonshineConfig) {
    super();
    const qw = cfg.heads * cfg.headDim;
    this.q = new nn.Linear(cfg.dModel, qw, { bias: false });
    this.kv = new nn.Linear(cfg.dModel, 2 * qw, { bias: false });
    this.mha = new nn.MultiHeadAttention({
      qHeads: cfg.heads,
      kvHeads: cfg.heads,
      headDim: cfg.headDim,
      windowLeft: Infinity,
      windowRight: Infinity,
    });
    this.out = new nn.Linear(qw, cfg.dModel, { bias: false });
  }
  precompute(encOut: Value): CrossKV {
    const fused = this.kv.forward(encOut); // [kvLen, 2·qw], columns k|v
    const [k, v] = chunk(fused, 2, 1);
    return { k: k!, v: v! };
  }
  forward(x: Value, cross: CrossKV, residual?: Value): Value {
    return this.out.forward(this.mha.forward(this.q.forward(x), cross.k, cross.v), {
      addend: residual,
    });
  }
}

class DecoderMlp extends nn.Module {
  readonly fc1: nn.Linear;
  readonly fc2: nn.Linear;
  constructor(ffn: number, dModel: number) {
    super();
    this.fc1 = new nn.Linear(dModel, 2 * ffn);
    this.fc2 = new nn.Linear(ffn, dModel);
  }
  forward(x: Value, residual?: Value): Value {
    return this.fc2.forward(swigluChunk(this.fc1.forward(x)), { addend: residual });
  }
}

export class DecoderLayer extends nn.Module<[Value, Value, Value, CrossKV], Value> {
  readonly norm1: nn.LayerNorm;
  readonly selfAttn: DecoderSelfAttn;
  readonly norm2: nn.LayerNorm;
  readonly crossAttn: DecoderCrossAttn;
  readonly norm3: nn.LayerNorm;
  readonly mlp: DecoderMlp;
  constructor(cfg: MoonshineConfig) {
    super();
    this.norm1 = new nn.LayerNorm(cfg.dModel);
    this.selfAttn = new DecoderSelfAttn(cfg);
    this.norm2 = new nn.LayerNorm(cfg.dModel);
    this.crossAttn = new DecoderCrossAttn(cfg);
    this.norm3 = new nn.LayerNorm(cfg.dModel);
    this.mlp = new DecoderMlp(cfg.ffn, cfg.dModel);
  }
  forward(x: Value, cos: Value, sin: Value, cross: CrossKV): Value {
    const h1 = this.selfAttn.forward(this.norm1.forward(x), cos, sin, x);
    const h2 = this.crossAttn.forward(this.norm2.forward(h1), cross, h1);
    return this.mlp.forward(this.norm3.forward(h2), h2);
  }
  forwardStep(
    x1: Value,
    cosRow: Value,
    sinRow: Value,
    cross: CrossKV,
    cache: SelfKV,
    t: number,
  ): Value {
    const h1 = this.selfAttn.forwardStep(this.norm1.forward(x1), cosRow, sinRow, cache, t, x1);
    const h2 = this.crossAttn.forward(this.norm2.forward(h1), cross, h1);
    return this.mlp.forward(this.norm3.forward(h2), h2);
  }
}

export class MoonshineDecoder extends nn.Module<[Value, Value, Value, readonly CrossKV[]], Value> {
  readonly layers: nn.ModuleList<DecoderLayer>;
  readonly norm: nn.LayerNorm;
  constructor(cfg: MoonshineConfig) {
    super();
    this.layers = new nn.ModuleList(
      Array.from({ length: cfg.decLayers }, () => new DecoderLayer(cfg)),
    );
    this.norm = new nn.LayerNorm(cfg.dModel);
  }
  precomputeCrossKV(encOut: Value): CrossKV[] {
    return this.layers.items.map((layer) => layer.crossAttn.precompute(encOut));
  }
  forward(x: Value, cos: Value, sin: Value, cross: readonly CrossKV[]): Value {
    if (cross.length !== this.layers.items.length) {
      throw new Error(
        `MoonshineDecoder: ${cross.length} cross K/V entries for ${this.layers.items.length} layers`,
      );
    }
    let h = x;
    this.layers.items.forEach((layer, i) => {
      h = layer.forward(h, cos, sin, cross[i]!);
    });
    return this.norm.forward(h);
  }
  forwardStep(
    x1: Value,
    cosRow: Value,
    sinRow: Value,
    cross: readonly CrossKV[],
    caches: readonly SelfKV[],
    t: number,
  ): Value {
    const n = this.layers.items.length;
    if (cross.length !== n || caches.length !== n) {
      throw new Error(
        `MoonshineDecoder.forwardStep: got ${cross.length} cross / ${caches.length} cache entries for ${n} layers`,
      );
    }
    let h = x1;
    this.layers.items.forEach((layer, i) => {
      h = layer.forwardStep(h, cosRow, sinRow, cross[i]!, caches[i]!, t);
    });
    return this.norm.forward(h);
  }
}
