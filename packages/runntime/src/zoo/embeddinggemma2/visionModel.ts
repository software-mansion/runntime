/** EmbeddingGemma 2 vision tower: a bidirectional ViT over an image's 16×16
 *  patches, each 3×3 square of patches averaged into one image token and
 *  projected to the text model's width.
 *
 *  Layers are the text model's without PLE. Positions enter twice: x and y
 *  embeddings on the patches and a 2D rope on q and k. q, k and v are
 *  RMS-normed per head, so scores carry no 1/√d scale.
 *
 *  Module fields are named like the checkpoint: the tower under
 *  `vision_tower`, its projection to the text model under `embed_vision`. */

import {
  add,
  cat,
  matmul,
  mul,
  nn,
  rmsNorm,
  rope,
  sdpa,
  slice,
  softmax,
  transpose,
  type LazyStateDict,
  type Value,
} from '../../core/index.ts';
import {
  EMBEDDINGGEMMA2_TEXT,
  EMBEDDINGGEMMA2_VISION,
  type EmbeddingGemma2VisionConfig,
} from './config.ts';
import { transformEmbeddingGemma2VisionStateDict } from './stateDictHooks.ts';

/** Patch count from which attention runs on matmuls. */
const MATMUL_ATTENTION_MIN_PATCHES = 2560;
/** Most scores one matmul writes: 128 MiB of f32, WebGPU's default
 *  storage binding limit. Larger images split their queries into equal
 *  chunks. */
const MAX_SCORES = 2 ** 25;

/** One image's inputs: P patches in row-major order. */
export interface EmbeddingGemma2VisionInputs {
  /** Patch pixels [P, 3·patch²], each in [-1, 1], (row, column, channel)
   *  order inside a patch. */
  patches: Value;
  /** Each patch's column and row [P, 1], f32. */
  xs: Value;
  ys: Value;
  /** Rope rows [P, headDim], f32, from visionRopeTables. */
  cos: Value;
  sin: Value;
  /** [N, P] pooling weights: row n holds 1/pool² on the patches of image
   *  token n. */
  pool: Value;
}

/** A Linear stored under a `linear` child, as the tower's projections are. */
class WrappedLinear extends nn.Module {
  readonly linear: nn.Linear;
  constructor(inFeatures: number, outFeatures: number, opts?: { activation?: 'geluTanh' }) {
    super();
    this.linear = new nn.Linear(inFeatures, outFeatures, { bias: false, ...opts });
  }
  forward(x: Value): Value {
    return this.linear.forward(x);
  }
}

/** The attention projections and per-head norms, fused at load. */
class Attention extends nn.Module {
  /** q | k | v rows, with q and k permuted for the interleaved rope. */
  readonly qkv: nn.Linear;
  /** q_norm and k_norm tiled per head, ones for v. */
  readonly qkv_norm: nn.Parameter;
  readonly o_proj: WrappedLinear;
  constructor({ hidden, heads, headDim }: EmbeddingGemma2VisionConfig) {
    super();
    const width = 3 * heads * headDim;
    this.qkv = new nn.Linear(hidden, width, { bias: false });
    this.qkv_norm = new nn.Parameter({ elems: width, dtype: 'f32', dims: [width] });
    this.o_proj = new WrappedLinear(heads * headDim, hidden);
  }
}

/** GeGLU MLP, GELU in the gate matmul's epilogue. */
class Mlp extends nn.Module {
  readonly gate_proj: WrappedLinear;
  readonly up_proj: WrappedLinear;
  readonly down_proj: WrappedLinear;
  constructor({ hidden, ffn }: EmbeddingGemma2VisionConfig) {
    super();
    this.gate_proj = new WrappedLinear(hidden, ffn, { activation: 'geluTanh' });
    this.up_proj = new WrappedLinear(hidden, ffn);
    this.down_proj = new WrappedLinear(ffn, hidden);
  }
  forward(x: Value): Value {
    return this.down_proj.forward(mul(this.gate_proj.forward(x), this.up_proj.forward(x)));
  }
}

class VisionLayer extends nn.Module<[Value, EmbeddingGemma2VisionInputs], Value> {
  readonly input_layernorm: nn.RMSNorm;
  readonly self_attn: Attention;
  readonly post_attention_layernorm: nn.RMSNorm;
  readonly pre_feedforward_layernorm: nn.RMSNorm;
  readonly mlp: Mlp;
  readonly post_feedforward_layernorm: nn.RMSNorm;

  constructor(private readonly cfg: EmbeddingGemma2VisionConfig) {
    super();
    const norm = () => new nn.RMSNorm(cfg.hidden, cfg.eps);
    this.input_layernorm = norm();
    this.self_attn = new Attention(cfg);
    this.post_attention_layernorm = norm();
    this.pre_feedforward_layernorm = norm();
    this.mlp = new Mlp(cfg);
    this.post_feedforward_layernorm = norm();
  }

  override forward(x: Value, inp: EmbeddingGemma2VisionInputs): Value {
    const { hidden, heads, headDim: d, eps } = this.cfg;
    const w = heads * d;
    const postNorm = (norm: nn.RMSNorm, y: Value, residual: Value) =>
      rmsNorm(y, norm.weight.value, norm.eps, { residual });

    // One grouped norm for every head, then q and k roped in one dispatch.
    const { self_attn } = this;
    const qkv = self_attn.qkv.forward(this.input_layernorm.forward(x));
    const n = rmsNorm(qkv, self_attn.qkv_norm.value, eps, { group: d });
    const qk = rope(n, inp.cos, inp.sin, { headDim: d, width: 2 * w });
    const head = (v: Value, start: number) => slice(v, 1, start, start + d);
    const p = x.shape.elems / hidden;
    let a: Value;
    if (p >= MATMUL_ATTENTION_MIN_PATCHES) {
      // softmax(q·kᵀ) · v per head, the queries in chunks of rows.
      const kv = Array.from({ length: heads }, (_, h) => ({
        kT: transpose(head(qk, w + h * d)),
        v: head(n, 2 * w + h * d),
      }));
      const parts = Math.ceil((p * p) / MAX_SCORES);
      const step = Math.ceil(p / parts);
      const chunks: Value[] = [];
      for (let r = 0; r < p; r += step) {
        const q = slice(qk, 0, r, Math.min(r + step, p));
        const outs = kv.map(({ kT, v }, h) => matmul(softmax(matmul(head(q, h * d), kT)), v));
        chunks.push(cat(outs, 1));
      }
      a = cat(chunks, 0);
    } else {
      a = sdpa(slice(qk, 1, 0, w), slice(qk, 1, w, 2 * w), slice(n, 1, 2 * w, 3 * w), {
        qHeads: heads,
        kvHeads: heads,
        headDim: d,
        windowLeft: Infinity,
        windowRight: Infinity,
      });
    }
    x = postNorm(this.post_attention_layernorm, self_attn.o_proj.forward(a), x);

    const m = this.mlp.forward(this.pre_feedforward_layernorm.forward(x));
    return postNorm(this.post_feedforward_layernorm, m, x);
  }
}

/** Patches to the tower's width, plus their x and y position embeddings. */
class PatchEmbedder extends nn.Module<[Value, Value, Value], Value> {
  readonly input_proj: nn.Linear;
  /** The checkpoint's `position_embedding_table`, split at load. */
  readonly pos_x: nn.Embedding;
  readonly pos_y: nn.Embedding;
  constructor({ hidden, patch, positions }: EmbeddingGemma2VisionConfig) {
    super();
    this.input_proj = new nn.Linear(3 * patch * patch, hidden, { bias: false });
    this.pos_x = new nn.Embedding(positions, hidden);
    this.pos_y = new nn.Embedding(positions, hidden);
  }
  forward(patches: Value, xs: Value, ys: Value): Value {
    const addend = add(this.pos_x.forward(xs), this.pos_y.forward(ys));
    return this.input_proj.forward(patches, { addend });
  }
}

class VisionEncoder extends nn.Module {
  readonly layers: nn.ModuleList<VisionLayer>;
  constructor(cfg: EmbeddingGemma2VisionConfig) {
    super();
    this.layers = new nn.ModuleList(Array.from({ length: cfg.layers }, () => new VisionLayer(cfg)));
  }
}

class VisionTower extends nn.Module {
  readonly patch_embedder: PatchEmbedder;
  readonly encoder: VisionEncoder;
  constructor(cfg: EmbeddingGemma2VisionConfig) {
    super();
    this.patch_embedder = new PatchEmbedder(cfg);
    this.encoder = new VisionEncoder(cfg);
  }
}

/** The pooled tokens' projection to the text model's width. */
class EmbedVision extends nn.Module {
  /** The pooled tokens enter this weightless norm scaled by √hidden. The
   *  norm cancels the scale, so eps divided by hidden stands in for it. Its
   *  weight is ones, a Buffer since the checkpoint has none. */
  readonly embedding_pre_projection_norm: nn.Buffer;
  readonly embedding_projection: nn.Linear;
  private readonly eps: number;
  constructor({ hidden, eps }: EmbeddingGemma2VisionConfig) {
    super();
    this.eps = eps / hidden;
    this.embedding_pre_projection_norm = new nn.Buffer(new Float32Array(hidden).fill(1), {
      elems: hidden,
      dtype: 'f32',
      dims: [hidden],
    });
    this.embedding_projection = new nn.Linear(hidden, EMBEDDINGGEMMA2_TEXT.hidden, { bias: false });
  }
  forward(pooled: Value): Value {
    return this.embedding_projection.forward(
      rmsNorm(pooled, this.embedding_pre_projection_norm.value, this.eps),
    );
  }
}

export class EmbeddingGemma2VisionModel extends nn.Module<[EmbeddingGemma2VisionInputs], Value> {
  readonly vision_tower: VisionTower;
  readonly embed_vision: EmbedVision;

  constructor(readonly cfg: EmbeddingGemma2VisionConfig = EMBEDDINGGEMMA2_VISION) {
    super();
    this.vision_tower = new VisionTower(cfg);
    this.embed_vision = new EmbedVision(cfg);
  }

  /** One image to its tokens [N, text hidden], ready to stand in for N
   *  placeholder rows of the text model's input. */
  override forward(inp: EmbeddingGemma2VisionInputs): Value {
    const { patch_embedder, encoder } = this.vision_tower;
    let x = patch_embedder.forward(inp.patches, inp.xs, inp.ys);
    for (const layer of encoder.layers.items) x = layer.forward(x, inp);
    return this.embed_vision.forward(matmul(inp.pool, x));
  }

  override transformStateDict(sd: LazyStateDict, prefix: string): void {
    transformEmbeddingGemma2VisionStateDict(sd, prefix, this.cfg);
  }
}
