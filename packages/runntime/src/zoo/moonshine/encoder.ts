/** Moonshine encoder: the conv stem over raw audio, then pre-norm
 *  transformer layers, each running LN, bidirectional MHA with partial rope,
 *  and an MLP, with a final weight-only LN.
 *
 *  The stem is conv1 k=127 s=64 (no bias) → tanh → nn.GroupNorm(1, d) → conv2
 *  k=7 s=3 (+bias) → gelu → conv3 k=3 s=2 (+bias) → gelu, valid padding
 *  throughout. Each stage's bias and activation ride the conv epilogue.
 *
 *  Attention is one fused qkv matmul with q and k roped in one dispatch each,
 *  then a fully unbounded window. The rope tables carry the score scale, so the
 *  kernel needs none. Module fields are named like the checkpoint. */

import { gelu, nn, rope, slice, type Value } from '../../core/index.ts';
import type { MoonshineConfig } from './config.ts';
import { K1, K2, K3, S1, S2, S3 } from './convStem.ts';

class EncoderAttn extends nn.Module<[Value, Value, Value, Value?], Value> {
  /** q | k | v rows, fused at load. */
  readonly qkv: nn.Linear;
  readonly mha: nn.MultiHeadAttention;
  readonly o_proj: nn.Linear;
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
    this.o_proj = new nn.Linear(qw, cfg.dModel, { bias: false });
  }
  forward(x: Value, cos: Value, sin: Value, residual?: Value): Value {
    const { heads, headDim } = this.cfg;
    const qw = heads * headDim;
    const qkv = this.qkv.forward(x); // [T, 3·qw], columns q|k|v
    const q = rope(qkv, cos, sin, { headDim, srcStart: 0, width: qw });
    const k = rope(qkv, cos, sin, { headDim, srcStart: qw, width: qw });
    const v = slice(qkv, 1, 2 * qw, 3 * qw);
    return this.o_proj.forward(this.mha.forward(q, k, v), { addend: residual });
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
  readonly input_layernorm: nn.LayerNorm;
  readonly self_attn: EncoderAttn;
  readonly post_attention_layernorm: nn.LayerNorm;
  readonly mlp: EncoderMlp;
  constructor(cfg: MoonshineConfig) {
    super();
    this.input_layernorm = new nn.LayerNorm(cfg.dModel);
    this.self_attn = new EncoderAttn(cfg);
    this.post_attention_layernorm = new nn.LayerNorm(cfg.dModel);
    this.mlp = new EncoderMlp(cfg);
  }
  forward(x: Value, cos: Value, sin: Value): Value {
    const h = this.self_attn.forward(this.input_layernorm.forward(x), cos, sin, x);
    return this.mlp.forward(this.post_attention_layernorm.forward(h), h);
  }
}

export class MoonshineEncoder extends nn.Module<[Value, Value, Value], Value> {
  readonly conv1: nn.Conv1d;
  readonly groupnorm: nn.GroupNorm;
  readonly conv2: nn.Conv1d;
  readonly conv3: nn.Conv1d;
  readonly layers: nn.ModuleList<EncoderLayer>;
  readonly layer_norm: nn.LayerNorm;
  constructor(cfg: MoonshineConfig) {
    super();
    const d = cfg.dModel;
    this.conv1 = new nn.Conv1d(1, d, K1, { stride: S1, bias: false, activation: 'tanh' });
    this.groupnorm = new nn.GroupNorm(1, d);
    this.conv2 = new nn.Conv1d(d, 2 * d, K2, { stride: S2, activation: 'gelu' });
    this.conv3 = new nn.Conv1d(2 * d, d, K3, { stride: S3, activation: 'gelu' });
    this.layers = new nn.ModuleList(
      Array.from({ length: cfg.encLayers }, () => new EncoderLayer(cfg)),
    );
    this.layer_norm = new nn.LayerNorm(cfg.dModel);
  }
  /** Raw audio [L, 1] to encoder frames [T, d]; the rope tables cover T. */
  forward(x: Value, cos: Value, sin: Value): Value {
    let h = this.conv3.forward(this.conv2.forward(this.groupnorm.forward(this.conv1.forward(x))));
    for (const layer of this.layers.items) h = layer.forward(h, cos, sin);
    return this.layer_norm.forward(h);
  }
}
