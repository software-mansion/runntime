/** Moonshine v2 encoder: the embedder, then pre-norm transformer layers
 *  carrying no positional information. Locality comes entirely from
 *  per-layer asymmetric sliding-window attention, with windows converted from
 *  HF's strict bounds to the kernel's inclusive form.
 *
 *  There is no rope, so the score scale folds into the q columns at load. The
 *  loader also adds 1 back to the checkpoint's offset norm gamma, so plain
 *  weight-only LayerNorms serve here. Module fields are named like the
 *  checkpoint. */

import { chunk, gelu, nn, type Value } from '../../core/index.ts';
import type { MoonshineStreamingConfig } from './config.ts';
import { StreamingEmbedder } from './frontend.ts';

class EncoderAttn extends nn.Module {
  /** q | k | v rows, fused at load. */
  readonly qkv: nn.Linear;
  readonly mha: nn.MultiHeadAttention;
  readonly o_proj: nn.Linear;
  constructor(
    private readonly cfg: MoonshineStreamingConfig,
    window: readonly [number, number],
  ) {
    super();
    const { hidden, heads, headDim } = cfg.enc;
    const qw = heads * headDim;
    this.qkv = new nn.Linear(hidden, 3 * qw, { bias: false });
    this.mha = new nn.MultiHeadAttention({
      qHeads: heads,
      kvHeads: heads,
      headDim,
      windowLeft: window[0] - 1,
      windowRight: Math.max(0, window[1] - 1),
    });
    this.o_proj = new nn.Linear(qw, hidden, { bias: false });
  }
  forward(x: Value, residual?: Value): Value {
    const qkv = this.qkv.forward(x); // [T, 3·qw], columns q|k|v
    const [q, k, v] = chunk(qkv, 3, 1) as [Value, Value, Value];
    return this.o_proj.forward(this.mha.forward(q, k, v), { addend: residual });
  }
}

class EncoderMlp extends nn.Module {
  readonly fc1: nn.Linear;
  readonly fc2: nn.Linear;
  constructor(cfg: MoonshineStreamingConfig) {
    super();
    this.fc1 = new nn.Linear(cfg.enc.hidden, cfg.enc.ffn);
    this.fc2 = new nn.Linear(cfg.enc.ffn, cfg.enc.hidden);
  }
  forward(x: Value, residual?: Value): Value {
    return this.fc2.forward(gelu(this.fc1.forward(x)), { addend: residual });
  }
}

export class EncoderLayer extends nn.Module {
  readonly input_layernorm: nn.LayerNorm;
  readonly self_attn: EncoderAttn;
  readonly post_attention_layernorm: nn.LayerNorm;
  readonly mlp: EncoderMlp;
  constructor(cfg: MoonshineStreamingConfig, window: readonly [number, number]) {
    super();
    this.input_layernorm = new nn.LayerNorm(cfg.enc.hidden);
    this.self_attn = new EncoderAttn(cfg, window);
    this.post_attention_layernorm = new nn.LayerNorm(cfg.enc.hidden);
    this.mlp = new EncoderMlp(cfg);
  }
  forward(x: Value): Value {
    const h = this.self_attn.forward(this.input_layernorm.forward(x), x);
    return this.mlp.forward(this.post_attention_layernorm.forward(h), h);
  }
}

export class StreamingEncoder extends nn.Module {
  readonly embedder: StreamingEmbedder;
  readonly layers: nn.ModuleList<EncoderLayer>;
  readonly final_norm: nn.LayerNorm;
  constructor(cfg: MoonshineStreamingConfig) {
    super();
    this.embedder = new StreamingEmbedder(cfg);
    this.layers = new nn.ModuleList(cfg.enc.windows.map((w) => new EncoderLayer(cfg, w)));
    this.final_norm = new nn.LayerNorm(cfg.enc.hidden);
  }
  /** Audio frames [T, frameLen] to encoder frames [T/4, enc.hidden]. */
  forward(x: Value): Value {
    let h = this.embedder.forward(x);
    for (const layer of this.layers.items) h = layer.forward(h);
    return this.final_norm.forward(h);
  }
}
