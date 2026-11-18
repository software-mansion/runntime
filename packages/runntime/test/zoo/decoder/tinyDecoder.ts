/** A tiny decoder of either family with random bf16 weights, packed as a
 *  safetensors file, and an f64 reference forward of the same weights
 *  (HF Gemma3ForCausalLM / Qwen3ForCausalLM semantics) to check the engine
 *  against. */
import type { DecoderConfig } from '../../../src/zoo/decoder/config.ts';
import { gemma3ConfigFromJson } from '../../../src/zoo/gemma3/config.ts';
import { qwen3ConfigFromJson } from '../../../src/zoo/qwen3/config.ts';
import { randomF32 } from '../../gpu.ts';

/** 3 layers, the first two sliding with a window of 6 (Gemma) and a local
 *  RoPE base that differs from the global one, so a ring or table mix-up
 *  shows. GQA with 2 q heads per kv head. */
export function tinyConfig(arch: DecoderConfig['arch']): DecoderConfig {
  const common = {
    hidden_size: 64,
    intermediate_size: 96,
    num_hidden_layers: 3,
    num_attention_heads: 4,
    num_key_value_heads: 2,
    head_dim: 16,
    vocab_size: 200,
    rms_norm_eps: 1e-6,
    rope_theta: 10_000,
    max_position_embeddings: 512,
  };
  if (arch === 'qwen3') return qwen3ConfigFromJson({ ...common, tie_word_embeddings: false });
  return gemma3ConfigFromJson({
    ...common,
    query_pre_attn_scalar: 16,
    rope_local_base_freq: 1_000,
    sliding_window: 6,
    layer_types: ['sliding_attention', 'sliding_attention', 'full_attention'],
  });
}

/** f32 → bf16 bits, round to nearest even. */
function toBf16(f32: Float32Array): Uint16Array {
  const u = new Uint32Array(f32.buffer, f32.byteOffset, f32.length);
  const out = new Uint16Array(f32.length);
  for (let i = 0; i < u.length; i++) {
    const x = u[i]!;
    out[i] = (x + 0x7fff + ((x >> 16) & 1)) >>> 16;
  }
  return out;
}

function fromBf16(bits: Uint16Array): Float32Array {
  const out = new Float32Array(bits.length);
  const u = new Uint32Array(out.buffer);
  for (let i = 0; i < bits.length; i++) u[i] = bits[i]! << 16;
  return out;
}

export interface TinyDecoder {
  cfg: DecoderConfig;
  /** The safetensors file. */
  file: Uint8Array;
  /** Every tensor as the f32 the bf16 bits decode to. */
  weights: Map<string, { data: Float32Array; shape: number[] }>;
}

/** Random weights scaled so activations stay O(1) through the layers.
 *  `zeroRows` zeroes those LM-head rows (the embedding when tied), so their
 *  logit is exactly 0 and greedy decoding never picks them. */
export function tinyDecoder(cfg: DecoderConfig, zeroRows: readonly number[] = []): TinyDecoder {
  const { hidden, ffn, heads: H, kvHeads: KVH, headDim: D, vocab: V } = cfg;
  const gemma = cfg.arch === 'gemma3';
  const shapes: [string, number[], 'mat' | 'norm' | 'emb'][] = [
    ['model.embed_tokens.weight', [V, hidden], 'emb'],
    ['model.norm.weight', [hidden], 'norm'],
  ];
  if (!cfg.tieEmbeddings) shapes.push(['lm_head.weight', [V, hidden], 'emb']);
  for (let i = 0; i < cfg.layers; i++) {
    const p = `model.layers.${i}.`;
    shapes.push(
      [`${p}input_layernorm.weight`, [hidden], 'norm'],
      [`${p}post_attention_layernorm.weight`, [hidden], 'norm'],
      [`${p}self_attn.q_norm.weight`, [D], 'norm'],
      [`${p}self_attn.k_norm.weight`, [D], 'norm'],
      [`${p}self_attn.q_proj.weight`, [H * D, hidden], 'mat'],
      [`${p}self_attn.k_proj.weight`, [KVH * D, hidden], 'mat'],
      [`${p}self_attn.v_proj.weight`, [KVH * D, hidden], 'mat'],
      [`${p}self_attn.o_proj.weight`, [hidden, H * D], 'mat'],
      [`${p}mlp.gate_proj.weight`, [ffn, hidden], 'mat'],
      [`${p}mlp.up_proj.weight`, [ffn, hidden], 'mat'],
      [`${p}mlp.down_proj.weight`, [hidden, ffn], 'mat'],
    );
    if (gemma) {
      shapes.push(
        [`${p}pre_feedforward_layernorm.weight`, [hidden], 'norm'],
        [`${p}post_feedforward_layernorm.weight`, [hidden], 'norm'],
      );
    }
  }
  const weights = new Map<string, { data: Float32Array; shape: number[] }>();
  const header: Record<string, unknown> = {};
  const parts: Uint16Array[] = [];
  let offset = 0;
  shapes.forEach(([name, shape, kind], seed) => {
    const n = shape.reduce((a, b) => a * b, 1);
    const r = randomF32(n, seed + 1);
    const f32 = new Float32Array(n);
    for (let i = 0; i < n; i++) {
      if (kind === 'norm') f32[i] = (gemma ? 0 : 1) + 0.2 * r[i]!;
      else if (kind === 'emb') f32[i] = r[i]!;
      else f32[i] = (1.5 * r[i]!) / Math.sqrt(shape[1]!);
    }
    const head = cfg.tieEmbeddings || name === 'lm_head.weight';
    if (kind === 'emb' && head)
      for (const row of zeroRows) f32.fill(0, row * hidden, (row + 1) * hidden);
    const bits = toBf16(f32);
    weights.set(name, { data: fromBf16(bits), shape });
    header[name] = { dtype: 'BF16', shape, data_offsets: [offset, offset + bits.byteLength] };
    parts.push(bits);
    offset += bits.byteLength;
  });
  const json = new TextEncoder().encode(JSON.stringify(header));
  const file = new Uint8Array(8 + json.length + offset);
  new DataView(file.buffer).setBigUint64(0, BigInt(json.length), true);
  file.set(json, 8);
  let at = 8 + json.length;
  for (const p of parts) {
    file.set(new Uint8Array(p.buffer), at);
    at += p.byteLength;
  }
  return { cfg, file, weights };
}

/** Logits [T, vocab] of every position of `ids`, in f64. */
export function referenceLogits(
  { cfg, weights }: TinyDecoder,
  ids: readonly number[],
): Float64Array[] {
  const { hidden, ffn, heads: H, kvHeads: KVH, headDim: D, vocab: V, rmsEps } = cfg;
  const gemma = cfg.arch === 'gemma3';
  const T = ids.length;
  const w = (n: string) => weights.get(n)!.data;
  const rms = (x: Float64Array, n: number, scale: Float32Array) => {
    const out = new Float64Array(x.length);
    for (let r = 0; r < x.length / n; r++) {
      let ss = 0;
      for (let i = 0; i < n; i++) ss += x[r * n + i]! ** 2;
      const s = 1 / Math.sqrt(ss / n + rmsEps);
      for (let i = 0; i < n; i++)
        out[r * n + i] = x[r * n + i]! * s * ((gemma ? 1 : 0) + scale[i]!);
    }
    return out;
  };
  const lin = (x: Float64Array, name: string, N: number, K: number) => {
    const wd = w(name);
    const rows = x.length / K;
    const out = new Float64Array(rows * N);
    for (let r = 0; r < rows; r++) {
      for (let n = 0; n < N; n++) {
        let s = 0;
        for (let k = 0; k < K; k++) s += x[r * K + k]! * wd[n * K + k]!;
        out[r * N + n] = s;
      }
    }
    return out;
  };
  const emb = w('model.embed_tokens.weight');
  let x = new Float64Array(T * hidden);
  const embScale = gemma ? Math.sqrt(hidden) : 1;
  for (let t = 0; t < T; t++) {
    for (let i = 0; i < hidden; i++) x[t * hidden + i] = emb[ids[t]! * hidden + i]! * embScale;
  }
  const half = D / 2;
  const rope = (v: Float64Array, off: number, pos: number, theta: number) => {
    const o = new Float64Array(D);
    for (let i = 0; i < half; i++) {
      const a = pos / theta ** ((2 * i) / D);
      const c = Math.cos(a);
      const s = Math.sin(a);
      o[i] = v[off + i]! * c - v[off + i + half]! * s;
      o[i + half] = v[off + i + half]! * c + v[off + i]! * s;
    }
    for (let i = 0; i < D; i++) v[off + i] = o[i]!;
  };
  const gelu = (g: number) =>
    0.5 * g * (1 + Math.tanh(0.7978845608028654 * (g + 0.044715 * g * g * g)));
  const silu = (g: number) => g / (1 + Math.exp(-g));
  const qScale = cfg.queryPreAttnScalar ** -0.5;
  for (let l = 0; l < cfg.layers; l++) {
    const p = `model.layers.${l}.`;
    const sliding = cfg.sliding[l]!;
    const theta = sliding ? cfg.ropeLocalTheta : cfg.ropeTheta;
    const xn = rms(x, hidden, w(`${p}input_layernorm.weight`));
    const q = rms(
      lin(xn, `${p}self_attn.q_proj.weight`, H * D, hidden),
      D,
      w(`${p}self_attn.q_norm.weight`),
    );
    const k = rms(
      lin(xn, `${p}self_attn.k_proj.weight`, KVH * D, hidden),
      D,
      w(`${p}self_attn.k_norm.weight`),
    );
    const v = lin(xn, `${p}self_attn.v_proj.weight`, KVH * D, hidden);
    for (let t = 0; t < T; t++) {
      for (let h = 0; h < H; h++) rope(q, (t * H + h) * D, t, theta);
      for (let h = 0; h < KVH; h++) rope(k, (t * KVH + h) * D, t, theta);
    }
    const attn = new Float64Array(T * H * D);
    for (let t = 0; t < T; t++) {
      const j0 = sliding ? Math.max(0, t - cfg.slidingWindow + 1) : 0;
      for (let h = 0; h < H; h++) {
        const kvh = Math.floor(h / (H / KVH));
        const scores: number[] = [];
        for (let j = j0; j <= t; j++) {
          let s = 0;
          for (let d = 0; d < D; d++) s += q[(t * H + h) * D + d]! * k[(j * KVH + kvh) * D + d]!;
          scores.push(s * qScale);
        }
        const m = Math.max(...scores);
        const e = scores.map((s) => Math.exp(s - m));
        const z = e.reduce((a, b) => a + b, 0);
        for (let d = 0; d < D; d++) {
          let acc = 0;
          for (let j = j0; j <= t; j++) acc += (e[j - j0]! / z) * v[(j * KVH + kvh) * D + d]!;
          attn[(t * H + h) * D + d] = acc;
        }
      }
    }
    const o = lin(attn, `${p}self_attn.o_proj.weight`, hidden, H * D);
    const attnOut = gemma ? rms(o, hidden, w(`${p}post_attention_layernorm.weight`)) : o;
    const h1 = x.map((xv, i) => xv + attnOut[i]!);
    const n = rms(
      h1,
      hidden,
      w(gemma ? `${p}pre_feedforward_layernorm.weight` : `${p}post_attention_layernorm.weight`),
    );
    const gate = lin(n, `${p}mlp.gate_proj.weight`, ffn, hidden);
    const up = lin(n, `${p}mlp.up_proj.weight`, ffn, hidden);
    const act = gate.map((g, i) => (gemma ? gelu(g) : silu(g)) * up[i]!);
    const down = lin(act, `${p}mlp.down_proj.weight`, hidden, ffn);
    const mlpOut = gemma ? rms(down, hidden, w(`${p}post_feedforward_layernorm.weight`)) : down;
    x = h1.map((hv, i) => hv + mlpOut[i]!);
  }
  const head = cfg.tieEmbeddings ? 'model.embed_tokens.weight' : 'lm_head.weight';
  const logits = lin(rms(x, hidden, w('model.norm.weight')), head, V, hidden);
  return Array.from({ length: T }, (_, t) => logits.subarray(t * V, (t + 1) * V));
}

export const argmax = (a: ArrayLike<number>): number => {
  let best = 0;
  for (let i = 1; i < a.length; i++) if (a[i]! > a[best]!) best = i;
  return best;
};
