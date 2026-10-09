/** EmbeddingGemma 2 vision tower: Gemma 4 ViT, 16 layers of 768 with head 64,
 *  then 3×3 average pooling and a projection into the text model's 512.
 *
 *  An image becomes up to 2520 patches of 16×16, each tagged with its (x, y)
 *  grid position. Position enters twice: a learned x and y table added to the
 *  patch projection, and an axial rope that turns the first half of each head
 *  by x and the second half by y. The pooled tokens then run through the text
 *  model as soft tokens. Demo-grade: f32, one image per call. */

import {
  add,
  matmul,
  mul,
  nn,
  rope,
  slice,
  tensor,
  type LazyStateDict,
  type LazyTensor,
  type Value,
} from '../../core/index.ts';
import { astype, rsqrt } from '../../core/graph/ops.ts';
import { meanSquare, rmsNormFused } from '../../core/graph/ops/reduce.ts';
import { derivedTensor } from '../../core/index.ts';
import { headwiseAttention } from './model.ts';
import { interleaveHeads } from './stateDictHooks.ts';

const HIDDEN = 768;
const FFN = 3072;
const HEADS = 12;
const HEAD = 64;
const LAYERS = 16;
const POS_TABLE = 10240;
const EPS = 1e-6;
export const PATCH = 16;
export const POOL = 3;
export const MAX_SOFT_TOKENS = 280;
const THETA = 100;

const rmsNormPlain = (x: Value) =>
  mul(x, astype(rsqrt(add(meanSquare(x), EPS)), x.shape.dtype === 'f16' ? 'f16' : 'f32'));

/** Axial rope tables in the kernel's pair layout: pairs 0..15 turn by x,
 *  16..31 by y, each half with its own 16 frequencies. */
function axialRope(positions: Int32Array, n: number) {
  const cos = new Float32Array(n * HEAD);
  const sin = new Float32Array(n * HEAD);
  const quarter = HEAD / 4;
  for (let t = 0; t < n; t++) {
    for (let p = 0; p < HEAD / 2; p++) {
      const axis = p < quarter ? 0 : 1;
      const i = p % quarter;
      const angle = positions[2 * t + axis]! / THETA ** ((2 * i) / (HEAD / 2));
      cos[t * HEAD + 2 * p] = cos[t * HEAD + 2 * p + 1] = Math.cos(angle);
      sin[t * HEAD + 2 * p] = sin[t * HEAD + 2 * p + 1] = Math.sin(angle);
    }
  }
  return { cos, sin };
}

class VisionLayer extends nn.Module<[Value, Value, Value], Value> {
  readonly input_layernorm = new nn.RMSNorm(HIDDEN, EPS);
  /** q | k | v rows, q and k permuted per 32-wide rope half. */
  readonly qkv = new nn.Linear(HIDDEN, 3 * HIDDEN, { bias: false });
  /** q_norm and k_norm tiled per head, ones for v. */
  readonly qkv_norm = new nn.Parameter({ elems: 3 * HIDDEN, dtype: 'f32', dims: [3 * HIDDEN] });
  readonly o_proj = new nn.Linear(HIDDEN, HIDDEN, { bias: false });
  readonly post_attention_layernorm = new nn.RMSNorm(HIDDEN, EPS);
  readonly pre_feedforward_layernorm = new nn.RMSNorm(HIDDEN, EPS);
  readonly gate_proj = new nn.Linear(HIDDEN, FFN, { bias: false, activation: 'geluTanh' });
  readonly up_proj = new nn.Linear(HIDDEN, FFN, { bias: false });
  readonly down_proj = new nn.Linear(FFN, HIDDEN, { bias: false });
  readonly post_feedforward_layernorm = new nn.RMSNorm(HIDDEN, EPS);

  override forward(x: Value, cos: Value, sin: Value): Value {
    const qkv = this.qkv.forward(this.input_layernorm.forward(x));
    const n = rmsNormFused(qkv, this.qkv_norm.value, EPS, { group: HEAD });
    const q = rope(n, cos, sin, { headDim: HEAD, srcStart: 0, width: HIDDEN });
    const k = rope(n, cos, sin, { headDim: HEAD, srcStart: HIDDEN, width: HIDDEN });
    const v = slice(n, 1, 2 * HIDDEN, 3 * HIDDEN);
    // Per-head matmuls: at 2.4k patches they beat the fused sdpa kernel
    // (f32 1.5x, f16 ~4x on an M4 Pro).
    const a = headwiseAttention(q, k, v, HEADS, HEADS, HEAD);
    const post = (norm: nn.RMSNorm, y: Value, residual: Value) =>
      rmsNormFused(y, norm.scale.value, EPS, { residual });
    x = post(this.post_attention_layernorm, this.o_proj.forward(a), x);
    const f = this.pre_feedforward_layernorm.forward(x);
    const m = this.down_proj.forward(mul(this.gate_proj.forward(f), this.up_proj.forward(f)));
    return post(this.post_feedforward_layernorm, m, x);
  }
}

/** Patches ready for the tower: pixels already in [-1, 1], laid out
 *  [n, 16·16·3] row by row with channels last, plus (x, y) per patch. */
export interface ImagePatches {
  pixels: Float32Array;
  positions: Int32Array;
  n: number;
  /** Patch grid width; pooling needs it. */
  cols: number;
}

export class EmbeddingGemma2Vision extends nn.Module<[ImagePatches], Value> {
  readonly input_proj = new nn.Linear(PATCH * PATCH * 3, HIDDEN, { bias: false });
  /** x rows then y rows of the learned position table, gathered on the CPU. */
  readonly position_embedding = new nn.Embedding(2 * POS_TABLE, HIDDEN, { device: 'cpu' });
  readonly layers = new nn.ModuleList(Array.from({ length: LAYERS }, () => new VisionLayer()));
  readonly embedding_projection = new nn.Linear(HIDDEN, 512, { bias: false });

  /** Patches to soft tokens [n / 9, 512] for the text model. */
  override forward(img: ImagePatches): Value {
    const { n, positions, cols } = img;
    const xs = Array.from({ length: n }, (_, t) => positions[2 * t]!);
    const ys = Array.from({ length: n }, (_, t) => POS_TABLE + positions[2 * t + 1]!);
    const px = this.position_embedding.gather(xs);
    const py = this.position_embedding.gather(ys);
    for (let i = 0; i < px.length; i++) px[i]! += py[i]!;
    const dtype = this.input_proj.weight.shape.dtype === 'f16' ? 'f16' : 'f32';
    const act = (data: Float32Array, cols: number) =>
      tensor(data, { elems: data.length, dtype, dims: [data.length / cols, cols] });
    let x = this.input_proj.forward(act(img.pixels, PATCH * PATCH * 3), {
      addend: act(px, HIDDEN),
    });

    const { cos, sin } = axialRope(positions, n);
    const cosV = tensor(cos, [n, HEAD]);
    const sinV = tensor(sin, [n, HEAD]);
    for (const layer of this.layers.items) x = layer.forward(x, cosV, sinV);

    // 3×3 average pool as one matmul, with sqrt(hidden) folded into the weights.
    const out = n / (POOL * POOL);
    const poolCols = cols / POOL;
    const w = new Float32Array(out * n);
    const scale = Math.sqrt(HIDDEN) / (POOL * POOL);
    for (let t = 0; t < n; t++) {
      const cell =
        Math.floor(positions[2 * t]! / POOL) + poolCols * Math.floor(positions[2 * t + 1]! / POOL);
      w[cell * n + t] = scale;
    }
    // Pooling scales by sqrt(hidden), past f16 range: pool and norm in f32.
    const pooled = matmul(tensor(w, [out, n]), astype(x, 'f32'));
    const soft = this.embedding_projection.forward(astype(rmsNormPlain(pooled), dtype));
    return astype(soft, 'f32');
  }

  override transformStateDict(sd: LazyStateDict, prefix: string): void {
    const q = (name: string) => (prefix ? `${prefix}.${name}` : name);
    const src = new Map(sd.tensors);
    sd.tensors.clear();
    const take = (name: string) => {
      const t = src.get(name);
      if (!t) throw new Error(`embeddinggemma2 vision: missing tensor '${name}'`);
      return t;
    };
    const put = (name: string, t: LazyTensor) => sd.tensors.set(q(name), t);
    // Each head is two independent 32-wide rope halves.
    const halves = (x: Float32Array, rows: number, cols: number) =>
      interleaveHeads(x, rows, cols, HEAD / 2);
    put('embedding_projection.weight', take('embed_vision.embedding_projection.weight'));
    put('input_proj.weight', take('vision_tower.patch_embedder.input_proj.weight'));
    put('position_embedding.weight', take('vision_tower.patch_embedder.position_embedding_table'));
    for (let i = 0; i < LAYERS; i++) {
      const from = `vision_tower.encoder.layers.${i}`;
      const L = `layers.${i}`;
      const w = (n: string) => take(`${from}.${n}.linear.weight`);
      const [wq, wk, wv] = [w('self_attn.q_proj'), w('self_attn.k_proj'), w('self_attn.v_proj')];
      put(
        `${L}.qkv.weight`,
        derivedTensor(
          [3 * HIDDEN, HIDDEN],
          async () => {
            const out = new Float32Array(3 * HIDDEN * HIDDEN);
            out.set(halves(await wq.f32(), HIDDEN, HIDDEN), 0);
            out.set(halves(await wk.f32(), HIDDEN, HIDDEN), HIDDEN * HIDDEN);
            out.set(await wv.f32(), 2 * HIDDEN * HIDDEN);
            return out;
          },
          wq.byteLength + wk.byteLength + wv.byteLength,
        ),
      );
      const nq = take(`${from}.self_attn.q_norm.weight`);
      const nk = take(`${from}.self_attn.k_norm.weight`);
      put(
        `${L}.qkv_norm`,
        derivedTensor(
          [3 * HIDDEN],
          async () => {
            const out = new Float32Array(3 * HIDDEN).fill(1);
            const qn = halves(await nq.f32(), HEAD, 1);
            const kn = halves(await nk.f32(), HEAD, 1);
            for (let h = 0; h < HEADS; h++) {
              out.set(qn, h * HEAD);
              out.set(kn, HIDDEN + h * HEAD);
            }
            return out;
          },
          nq.byteLength + nk.byteLength,
        ),
      );
      put(`${L}.o_proj.weight`, w('self_attn.o_proj'));
      for (const n of ['gate_proj', 'up_proj', 'down_proj']) put(`${L}.${n}.weight`, w(`mlp.${n}`));
      for (const n of [
        'input_layernorm',
        'post_attention_layernorm',
        'pre_feedforward_layernorm',
        'post_feedforward_layernorm',
      ]) {
        put(`${L}.${n}.scale`, take(`${from}.${n}.weight`));
      }
    }
  }
}

/** Resized RGBA pixels (width and height multiples of 48) to patches. */
export function patchify(rgba: Uint8ClampedArray, width: number, height: number): ImagePatches {
  const cols = width / PATCH;
  const rows = height / PATCH;
  const n = cols * rows;
  const pixels = new Float32Array(n * PATCH * PATCH * 3);
  const positions = new Int32Array(n * 2);
  let o = 0;
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const t = r * cols + c;
      positions[2 * t] = c;
      positions[2 * t + 1] = r;
      for (let py = 0; py < PATCH; py++) {
        for (let px = 0; px < PATCH; px++) {
          const i = ((r * PATCH + py) * width + c * PATCH + px) * 4;
          for (let ch = 0; ch < 3; ch++) pixels[o++] = 2 * (rgba[i + ch]! / 255) - 1;
        }
      }
    }
  }
  return { pixels, positions, n, cols };
}

/** The processor's target size: largest multiple of 48 per side that keeps
 *  aspect ratio within the 2520-patch budget. */
export function targetSize(width: number, height: number): [number, number] {
  const side = POOL * PATCH;
  const factor = Math.sqrt((MAX_SOFT_TOKENS * POOL * POOL * PATCH * PATCH) / (width * height));
  const w = Math.max(side, Math.floor((factor * width) / side) * side);
  const h = Math.max(side, Math.floor((factor * height) / side) * side);
  return [w, h];
}
