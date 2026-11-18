/** Architecture config of the decoder-only LLMs the shared engine runs
 *  (Gemma 3, Qwen 3), read from a Hugging Face config.json or given inline.
 *
 *  The families differ in a handful of switches:
 *    gemma3: RMSNorm scale (1 + w), embedding × sqrt(hidden), pre- and post-
 *            norms around attention and MLP, GeGLU (gelu_tanh), interleaved
 *            sliding-window layers with their own RoPE base, optional
 *            soft-capping;
 *    qwen3:  RMSNorm scale w, unscaled embedding, pre-norms only, SwiGLU,
 *            full attention everywhere.
 *  Both use GQA with per-head q/k RMSNorm and rotate-half RoPE. */

import { checkpointMismatch } from '../errors.ts';

export type DecoderArch = 'gemma3' | 'qwen3';

export interface DecoderConfig {
  readonly arch: DecoderArch;
  readonly hidden: number;
  readonly ffn: number;
  readonly layers: number;
  readonly heads: number;
  readonly kvHeads: number;
  readonly headDim: number;
  readonly vocab: number;
  readonly rmsEps: number;
  /** Scores are scaled by queryPreAttnScalar^-0.5. */
  readonly queryPreAttnScalar: number;
  /** RoPE base of the global (full attention) layers. */
  readonly ropeTheta: number;
  /** RoPE base of the sliding-window layers. */
  readonly ropeLocalTheta: number;
  /** Linear RoPE scaling factor of the global layers (1 = none). */
  readonly ropeScale: number;
  /** Window of the sliding layers; 0 when there are none. */
  readonly slidingWindow: number;
  /** true where layer i is a sliding-window layer. */
  readonly sliding: readonly boolean[];
  /** tanh soft-cap of the final logits; 0 = none. */
  readonly finalLogitSoftcap: number;
  /** tanh soft-cap of the attention scores; 0 = none. */
  readonly attnLogitSoftcap: number;
  /** The LM head is the embedding table (else `lm_head.weight`). */
  readonly tieEmbeddings: boolean;
  readonly maxPositions: number;
}

export type Json = Record<string, unknown>;

/** Reads numeric keys of a config.json, naming the family in errors. */
export function numReader(json: Json, family: string) {
  return (key: string, fallback?: number): number => {
    const v = json[key];
    if (typeof v === 'number') return v;
    if (fallback !== undefined) return fallback;
    throw checkpointMismatch(`${family} config`, `'${key}' missing or not a number`);
  };
}
