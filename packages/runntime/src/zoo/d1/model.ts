/** d1-omni-600M on core ops: the bidirectional LFM2 trunk, the decision head
 *  and the SigLIP2 vision tower with its projector. Audio is not ported.
 *
 *  Parameter names follow the checkpoint after stateDictHooks.ts reshapes it:
 *  RMSNorm weights are `scale`, the MLP's w3 and w1 are row-stacked into
 *  `w13`, attention q/k rows are permuted so rotate-half RoPE becomes the
 *  kernel's interleaved pairs, and every softmax scale is folded into q. */

import {
  add,
  cat,
  clamp,
  gatherRows,
  mul,
  nn,
  reshape,
  rope,
  sdpa,
  sdpaPacked,
  slice,
  swigluChunk,
  tanh,
  type LazyStateDict,
  type Value,
} from '../../core/index.ts';
import { transformD1StateDict } from './stateDictHooks.ts';

export const D1 = {
  vocab: 65536,
  hidden: 1024,
  ffn: 4608,
  heads: 16,
  kvHeads: 8,
  headDim: 64,
  eps: 1e-5,
  ropeTheta: 1e6,
  layerTypes: 'ccaccaccacacacac',
  headLayers: 2,
  headFfn: 4096,
  visionHidden: 768,
  visionFfn: 3072,
  visionHeads: 12,
  visionLayers: 12,
  visionEps: 1e-6,
  /** SigLIP2's position grid side; resized to each crop's patch grid. */
  posGrid: 16,
  projectorHidden: 2048,
} as const;

class ShortConv extends nn.Module<[Value, Value, Value | undefined, Value], Value> {
  readonly in_proj = new nn.Linear(D1.hidden, 3 * D1.hidden, { bias: false });
  readonly out_proj = new nn.Linear(D1.hidden, D1.hidden, { bias: false });
  /** The depthwise kernel's three taps, [hidden] each: t−1, t, t+1. */
  readonly tap0 = new nn.Parameter({ elems: D1.hidden, dtype: 'f32', dims: [D1.hidden] });
  readonly tap1 = new nn.Parameter({ elems: D1.hidden, dtype: 'f32', dims: [D1.hidden] });
  readonly tap2 = new nn.Parameter({ elems: D1.hidden, dtype: 'f32', dims: [D1.hidden] });

  /** out(C · conv(B · x)) with a centred 3-tap conv; `keepRight` [T,1] zeroes
   *  the t+1 tap on the last media row so media never reads the text. */
  override forward(x: Value, residual: Value, keepRight: Value | undefined, zeroRow: Value): Value {
    const t = x.shape.dims![0]!;
    const bcx = this.in_proj.forward(x);
    const b = slice(bcx, 1, 0, D1.hidden);
    const c = slice(bcx, 1, D1.hidden, 2 * D1.hidden);
    const u = slice(bcx, 1, 2 * D1.hidden, 3 * D1.hidden);
    const bx = mul(b, u);
    const prev = cat([zeroRow, slice(bx, 0, 0, t - 1)], 0);
    let next = cat([slice(bx, 0, 1, t), zeroRow], 0);
    if (keepRight) next = mul(next, keepRight);
    const y = add(
      add(mul(prev, this.tap0.value), mul(bx, this.tap1.value)),
      mul(next, this.tap2.value),
    );
    return this.out_proj.forward(mul(c, y), { addend: residual });
  }
}

class TrunkAttention extends nn.Module<[Value, Value, Value, Value, Value | undefined], Value> {
  readonly q_proj = new nn.Linear(D1.hidden, D1.heads * D1.headDim, { bias: false });
  readonly k_proj = new nn.Linear(D1.hidden, D1.kvHeads * D1.headDim, { bias: false });
  readonly v_proj = new nn.Linear(D1.hidden, D1.kvHeads * D1.headDim, { bias: false });
  readonly out_proj = new nn.Linear(D1.heads * D1.headDim, D1.hidden, { bias: false });
  readonly q_layernorm = new nn.RMSNorm(D1.headDim, D1.eps);
  readonly k_layernorm = new nn.RMSNorm(D1.headDim, D1.eps);

  override forward(x: Value, residual: Value, cos: Value, sin: Value, segments?: Value): Value {
    const t = x.shape.dims![0]!;
    const perHead = (v: Value, heads: number, norm: nn.RMSNorm) =>
      reshape(norm.forward(reshape(v, [t * heads, D1.headDim])), [t, heads * D1.headDim]);
    const q = perHead(this.q_proj.forward(x), D1.heads, this.q_layernorm);
    const k = perHead(this.k_proj.forward(x), D1.kvHeads, this.k_layernorm);
    const a = sdpa(
      rope(q, cos, sin, { headDim: D1.headDim }),
      rope(k, cos, sin, { headDim: D1.headDim }),
      this.v_proj.forward(x),
      {
        qHeads: D1.heads,
        kvHeads: D1.kvHeads,
        headDim: D1.headDim,
        windowLeft: Infinity,
        windowRight: Infinity,
        segments,
      },
    );
    return this.out_proj.forward(a, { addend: residual });
  }
}

class SwiGlu extends nn.Module {
  /** w3 rows then w1 rows: swigluChunk reads hidden first, gate second. */
  readonly w13 = new nn.Linear(D1.hidden, 2 * D1.ffn, { bias: false });
  readonly w2 = new nn.Linear(D1.ffn, D1.hidden, { bias: false });
  forward(x: Value, residual?: Value): Value {
    return this.w2.forward(swigluChunk(this.w13.forward(x)), { addend: residual });
  }
}

class TrunkLayer extends nn.Module {
  readonly operator_norm = new nn.RMSNorm(D1.hidden, D1.eps);
  readonly ffn_norm = new nn.RMSNorm(D1.hidden, D1.eps);
  readonly feed_forward = new SwiGlu();
  readonly conv?: ShortConv;
  readonly self_attn?: TrunkAttention;
  constructor(attention: boolean) {
    super();
    if (attention) this.self_attn = new TrunkAttention();
    else this.conv = new ShortConv();
  }
}

/** Per-call trunk inputs: rope tables, the attention key ranges and the conv
 *  mask that keep the media prefix blind to the text, and a zero row. */
export interface TrunkInputs {
  cos: Value;
  sin: Value;
  segments?: Value;
  keepRight?: Value;
  zeroRow: Value;
}

export class Trunk extends nn.Module<[Value, TrunkInputs], Value> {
  readonly embed_tokens = new nn.Embedding(D1.vocab, D1.hidden, { device: 'cpu' });
  readonly layers = new nn.ModuleList(
    [...D1.layerTypes].map((kind) => new TrunkLayer(kind === 'a')),
  );
  readonly embedding_norm = new nn.RMSNorm(D1.hidden, D1.eps);

  /** Embeddings [T, hidden] (media prefix first) to hidden states. */
  forward(h: Value, inp: TrunkInputs): Value {
    for (const layer of this.layers.items) {
      const x = layer.operator_norm.forward(h);
      h = layer.self_attn
        ? layer.self_attn.forward(x, h, inp.cos, inp.sin, inp.segments)
        : layer.conv!.forward(x, h, inp.keepRight, inp.zeroRow);
      h = layer.feed_forward.forward(layer.ffn_norm.forward(h), h);
    }
    return this.embedding_norm.forward(h);
  }
}

/** One nn.TransformerEncoderLayer (norm_first, ReLU) over [L, hidden]. */
class HeadLayer extends nn.Module {
  /** Packed q‖k‖v, q pre-scaled by 1/√headDim. */
  readonly in_proj = new nn.Linear(D1.hidden, 3 * D1.hidden);
  readonly out_proj = new nn.Linear(D1.hidden, D1.hidden);
  readonly linear1 = new nn.Linear(D1.hidden, D1.headFfn, { activation: 'relu' });
  readonly linear2 = new nn.Linear(D1.headFfn, D1.hidden);
  readonly norm1 = new nn.LayerNorm(D1.hidden, 1e-5, { bias: true });
  readonly norm2 = new nn.LayerNorm(D1.hidden, 1e-5, { bias: true });

  forward(h: Value): Value {
    const heads = D1.hidden / 64;
    const a = sdpaPacked(this.in_proj.forward(this.norm1.forward(h)), {
      qHeads: heads,
      kvHeads: heads,
      headDim: 64,
      windowLeft: Infinity,
      windowRight: Infinity,
    });
    h = this.out_proj.forward(a, { addend: h });
    return this.linear2.forward(this.linear1.forward(this.norm2.forward(h)), { addend: h });
  }
}

export class DecisionHead extends nn.Module<[Value, Value, readonly number[]], Value> {
  readonly type_emb = new nn.Embedding(3, D1.hidden, { device: 'cpu' });
  readonly layers = new nn.ModuleList(Array.from({ length: D1.headLayers }, () => new HeadLayer()));
  readonly scorer_norm = new nn.LayerNorm(D1.hidden, 1e-5, { bias: true });
  readonly scorer_fc = new nn.Linear(D1.hidden, D1.hidden, { activation: 'gelu' });
  readonly scorer_out = new nn.Linear(D1.hidden, 1);

  /** Text hidden states [L, hidden] and the question-type row [1, hidden] to
   *  one logit per marker, [markers, 1]. */
  override forward(text: Value, typeRow: Value, markers: readonly number[]): Value {
    let h = add(text, typeRow);
    for (const layer of this.layers.items) h = layer.forward(h);
    const g = gatherRows(h, markers);
    return this.scorer_out.forward(this.scorer_fc.forward(this.scorer_norm.forward(g)));
  }
}

/** gelu_pytorch_tanh: 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))). */
function geluTanh(x: Value): Value {
  const inner = mul(mul(x, add(mul(mul(x, x), 0.044715), 1)), Math.sqrt(2 / Math.PI));
  return mul(mul(x, add(tanh(clamp(inner, -10, 10)), 1)), 0.5);
}

class VisionLayer extends nn.Module {
  readonly layer_norm1 = new nn.LayerNorm(D1.visionHidden, D1.visionEps, { bias: true });
  readonly layer_norm2 = new nn.LayerNorm(D1.visionHidden, D1.visionEps, { bias: true });
  /** Packed q‖k‖v, q pre-scaled by 1/√headDim. */
  readonly qkv = new nn.Linear(D1.visionHidden, 3 * D1.visionHidden);
  readonly out_proj = new nn.Linear(D1.visionHidden, D1.visionHidden);
  readonly fc1 = new nn.Linear(D1.visionHidden, D1.visionFfn);
  readonly fc2 = new nn.Linear(D1.visionFfn, D1.visionHidden);

  forward(x: Value): Value {
    const a = sdpaPacked(this.qkv.forward(this.layer_norm1.forward(x)), {
      qHeads: D1.visionHeads,
      kvHeads: D1.visionHeads,
      headDim: 64,
      windowLeft: Infinity,
      windowRight: Infinity,
    });
    x = this.out_proj.forward(a, { addend: x });
    return this.fc2.forward(geluTanh(this.fc1.forward(this.layer_norm2.forward(x))), { addend: x });
  }
}

export class Vision extends nn.Module<[Value, Value, number, number], Value> {
  readonly patch_embedding = new nn.Linear(3 * 16 * 16, D1.visionHidden);
  /** [16·16, 768], resized on the CPU to each crop's patch grid. */
  readonly position_embedding = new nn.CpuParameter({
    elems: D1.posGrid * D1.posGrid * D1.visionHidden,
    dtype: 'f32',
    dims: [D1.posGrid * D1.posGrid, D1.visionHidden],
  });
  readonly layers = new nn.ModuleList(
    Array.from({ length: D1.visionLayers }, () => new VisionLayer()),
  );
  readonly post_layernorm = new nn.LayerNorm(D1.visionHidden, D1.visionEps, { bias: true });
  readonly linear_1 = new nn.Linear(4 * D1.visionHidden, D1.projectorHidden, {
    activation: 'gelu',
  });
  readonly linear_2 = new nn.Linear(D1.projectorHidden, D1.hidden);

  /** One crop's patches [ph·pw, 768] and resized positions to its prefix
   *  embeddings [(ph/2)·(pw/2), hidden]. */
  override forward(patches: Value, positions: Value, ph: number, pw: number): Value {
    let x = this.patch_embedding.forward(patches, { addend: positions });
    for (const layer of this.layers.items) x = layer.forward(x);
    x = this.post_layernorm.forward(x);
    // 2x2 pixel unshuffle: token (i, j) is patches (2i+m, 2j+k) side by side,
    // in (m, k) order.
    const quads = [0, 1].flatMap((m) =>
      [0, 1].map((k) => {
        const idx: number[] = [];
        for (let i = 0; i < ph / 2; i++) {
          for (let j = 0; j < pw / 2; j++) idx.push((2 * i + m) * pw + 2 * j + k);
        }
        return gatherRows(x, idx);
      }),
    );
    return this.linear_2.forward(this.linear_1.forward(cat(quads, 1)));
  }
}

export class D1Model extends nn.Module {
  readonly encoder = new Trunk();
  readonly head = new DecisionHead();
  readonly vision = new Vision();

  override transformStateDict(sd: LazyStateDict): void {
    transformD1StateDict(sd);
  }
}
