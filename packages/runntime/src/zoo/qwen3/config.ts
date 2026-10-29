/** Qwen 3 dense text-decoder architecture config, read from a Hugging Face
 *  config.json (`model_type: qwen3`) or given inline. */

import { checkpointMismatch } from '../errors.ts';
import { numReader, type DecoderConfig, type Json } from '../decoder/config.ts';

/** A DecoderConfig with `arch: 'qwen3'`. */
export type Qwen3Config = DecoderConfig;

/** Parses a qwen3 config.json. */
export function qwen3ConfigFromJson(json: Json): Qwen3Config {
  const num = numReader(json, 'qwen3');
  const act = json.hidden_act;
  if (act !== undefined && act !== 'silu') {
    throw checkpointMismatch('qwen3 config', `activation '${String(act)}' unsupported`);
  }
  if (json.attention_bias === true)
    throw checkpointMismatch('qwen3 config', 'attention_bias unsupported');
  if (json.use_sliding_window === true)
    throw checkpointMismatch('qwen3 config', 'sliding window unsupported');
  if (json.rope_scaling) throw checkpointMismatch('qwen3 config', 'rope_scaling unsupported');
  const layers = num('num_hidden_layers');
  const hidden = num('hidden_size');
  const heads = num('num_attention_heads');
  const headDim = num('head_dim', hidden / heads);
  const ropeTheta = num('rope_theta', 1_000_000);
  return Object.freeze({
    arch: 'qwen3',
    hidden,
    ffn: num('intermediate_size'),
    layers,
    heads,
    kvHeads: num('num_key_value_heads'),
    headDim,
    vocab: num('vocab_size'),
    rmsEps: num('rms_norm_eps', 1e-6),
    queryPreAttnScalar: headDim,
    ropeTheta,
    ropeLocalTheta: ropeTheta,
    ropeScale: 1,
    slidingWindow: 0,
    sliding: Object.freeze(Array.from({ length: layers }, () => false)),
    finalLogitSoftcap: 0,
    attnLogitSoftcap: 0,
    tieEmbeddings: json.tie_word_embeddings !== false,
    maxPositions: num('max_position_embeddings', 40_960),
  });
}

/** Qwen/Qwen3-0.6B. */
export const QWEN3_0_6B: Qwen3Config = qwen3ConfigFromJson({
  hidden_size: 1024,
  intermediate_size: 3072,
  num_hidden_layers: 28,
  num_attention_heads: 16,
  num_key_value_heads: 8,
  head_dim: 128,
  vocab_size: 151936,
  rms_norm_eps: 1e-6,
  rope_theta: 1_000_000,
  tie_word_embeddings: true,
  max_position_embeddings: 40_960,
});
