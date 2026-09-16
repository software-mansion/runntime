/** Moonshine v2 encoder: pre-norm transformer layers carrying no positional
 *  information. Locality comes entirely from per-layer asymmetric
 *  sliding-window attention, with windows converted from HF's strict bounds to
 *  the kernel's inclusive form.
 *
 *  There is no rope, so the score scale folds into the q columns at load. The
 *  loader also adds 1 back to the checkpoint's offset norm gamma, so plain
 *  weight-only LayerNorms serve here.
 *
 *  The adapter bridges encoder to decoder, and is where positions enter. */

import { add, chunk, gelu, nn, slice, type Value } from '../../core/index.ts';
import type { MoonshineStreamingConfig } from './config.ts';

class EncoderAttn extends nn.Module {
  readonly qkv: nn.Linear;
  readonly mha: nn.MultiHeadAttention;
  readonly out: nn.Linear;
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
    this.out = new nn.Linear(qw, hidden, { bias: false });
  }
  forward(x: Value, residual?: Value): Value {
    const qkv = this.qkv.forward(x); // [T, 3·qw], columns q|k|v
    const [q, k, v] = chunk(qkv, 3, 1) as [Value, Value, Value];
    return this.out.forward(this.mha.forward(q, k, v), { addend: residual });
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
  readonly norm1: nn.LayerNorm;
  readonly attn: EncoderAttn;
  readonly norm2: nn.LayerNorm;
  readonly mlp: EncoderMlp;
  constructor(cfg: MoonshineStreamingConfig, window: readonly [number, number]) {
    super();
    this.norm1 = new nn.LayerNorm(cfg.enc.hidden);
    this.attn = new EncoderAttn(cfg, window);
    this.norm2 = new nn.LayerNorm(cfg.enc.hidden);
    this.mlp = new EncoderMlp(cfg);
  }
  forward(x: Value): Value {
    const h = this.attn.forward(this.norm1.forward(x), x);
    return this.mlp.forward(this.norm2.forward(h), h);
  }
}

export class StreamingEncoder extends nn.Module {
  readonly layers: nn.ModuleList<EncoderLayer>;
  readonly norm: nn.LayerNorm;
  constructor(cfg: MoonshineStreamingConfig) {
    super();
    this.layers = new nn.ModuleList(cfg.enc.windows.map((w) => new EncoderLayer(cfg, w)));
    this.norm = new nn.LayerNorm(cfg.enc.hidden);
  }
  forward(x: Value): Value {
    let h = x;
    for (const layer of this.layers.items) h = layer.forward(h);
    return this.norm.forward(h);
  }
}

export class StreamingAdapter extends nn.Module {
  readonly posEmb: nn.Parameter;
  readonly proj?: nn.Linear;
  constructor(cfg: MoonshineStreamingConfig) {
    super();
    const [eh, dh] = [cfg.enc.hidden, cfg.dec.hidden];
    this.posEmb = new nn.Parameter({
      elems: cfg.maxPositions * eh,
      dtype: 'f32',
      dims: [cfg.maxPositions, eh],
    });
    this.proj = eh === dh ? undefined : new nn.Linear(eh, dh, { bias: false });
  }
  forward(x: Value): Value {
    const t = x.shape.dims![0]!;
    const positioned = add(x, slice(this.posEmb.value, 0, 0, t));
    return this.proj ? this.proj.forward(positioned) : positioned;
  }
}
