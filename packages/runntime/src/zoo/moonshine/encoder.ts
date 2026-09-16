/** Moonshine encoder: pre-norm transformer layers over the conv-stem frames,
 *  each running LN, bidirectional MHA with partial rope, and an MLP, with a
 *  final weight-only LN.
 *
 *  Attention is one fused qkv matmul with q and k roped in one dispatch each,
 *  then a fully unbounded window. The rope tables carry the score scale, so the
 *  kernel needs none. */

import { gelu, nn, rope, slice, type Value } from '../../core/index.ts';
import type { MoonshineConfig } from './config.ts';

class EncoderAttn extends nn.Module<[Value, Value, Value, Value?], Value> {
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
      windowRight: Infinity,
    });
    this.out = new nn.Linear(qw, cfg.dModel, { bias: false });
  }
  forward(x: Value, cos: Value, sin: Value, residual?: Value): Value {
    const { heads, headDim } = this.cfg;
    const qw = heads * headDim;
    const qkv = this.qkv.forward(x); // [T, 3·qw], columns q|k|v
    const q = rope(qkv, cos, sin, { headDim, srcStart: 0, width: qw });
    const k = rope(qkv, cos, sin, { headDim, srcStart: qw, width: qw });
    const v = slice(qkv, 1, 2 * qw, 3 * qw);
    return this.out.forward(this.mha.forward(q, k, v), { addend: residual });
  }
}

class EncoderMlp extends nn.Module {
  readonly fc1: nn.Linear;
  readonly fc2: nn.Linear;
  constructor(cfg: MoonshineConfig) {
    super();
    this.fc1 = new nn.Linear(cfg.dModel, cfg.ffn);
    this.fc2 = new nn.Linear(cfg.ffn, cfg.dModel);
  }
  forward(x: Value, residual?: Value): Value {
    return this.fc2.forward(gelu(this.fc1.forward(x)), { addend: residual });
  }
}

export class EncoderLayer extends nn.Module<[Value, Value, Value], Value> {
  readonly norm1: nn.LayerNorm;
  readonly attn: EncoderAttn;
  readonly norm2: nn.LayerNorm;
  readonly mlp: EncoderMlp;
  constructor(cfg: MoonshineConfig) {
    super();
    this.norm1 = new nn.LayerNorm(cfg.dModel);
    this.attn = new EncoderAttn(cfg);
    this.norm2 = new nn.LayerNorm(cfg.dModel);
    this.mlp = new EncoderMlp(cfg);
  }
  forward(x: Value, cos: Value, sin: Value): Value {
    const h = this.attn.forward(this.norm1.forward(x), cos, sin, x);
    return this.mlp.forward(this.norm2.forward(h), h);
  }
}

export class MoonshineEncoder extends nn.Module<[Value, Value, Value], Value> {
  readonly layers: nn.ModuleList<EncoderLayer>;
  readonly norm: nn.LayerNorm;
  constructor(cfg: MoonshineConfig) {
    super();
    this.layers = new nn.ModuleList(
      Array.from({ length: cfg.encLayers }, () => new EncoderLayer(cfg)),
    );
    this.norm = new nn.LayerNorm(cfg.dModel);
  }
  forward(x: Value, cos: Value, sin: Value): Value {
    let h = x;
    for (const layer of this.layers.items) h = layer.forward(h, cos, sin);
    return this.norm.forward(h);
  }
}
