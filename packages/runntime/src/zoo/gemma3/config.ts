/** Gemma 3 text-decoder architecture config, read from a Hugging Face
 *  config.json (`model_type: gemma3_text`) or given inline. */

import { checkpointMismatch } from '../errors.ts';
import { numReader, type DecoderConfig, type Json } from '../decoder/config.ts';

/** A DecoderConfig with `arch: 'gemma3'`. */
export type Gemma3Config = DecoderConfig;

/** Parses a gemma3_text config.json (or a gemma3 multimodal config's
 *  `text_config`). */
export function gemma3ConfigFromJson(raw: Json): Gemma3Config {
  const json = (raw.text_config as Json | undefined) ?? raw;
  const num = numReader(json, 'gemma3');
  const act = json.hidden_activation ?? json.hidden_act;
  if (act !== undefined && act !== 'gelu_pytorch_tanh') {
    throw checkpointMismatch('gemma3 config', `activation '${String(act)}' unsupported`);
  }
  if (json.attention_bias === true)
    throw checkpointMismatch('gemma3 config', 'attention_bias unsupported');
  const layers = num('num_hidden_layers');
  const hidden = num('hidden_size');
  const heads = num('num_attention_heads');
  let sliding: boolean[];
  if (Array.isArray(json.layer_types)) {
    sliding = (json.layer_types as string[]).map((t) => t === 'sliding_attention');
    if (sliding.length !== layers)
      throw checkpointMismatch('gemma3 config', 'layer_types length mismatch');
  } else {
    const pattern = num('sliding_window_pattern', (json._sliding_window_pattern as number) ?? 6);
    sliding = Array.from({ length: layers }, (_, i) => (i + 1) % pattern !== 0);
  }
  const scaling = json.rope_scaling as Json | null | undefined;
  let ropeScale = 1;
  if (scaling) {
    const type = scaling.rope_type ?? scaling.type;
    if (type !== 'linear' && type !== 'default') {
      throw checkpointMismatch('gemma3 config', `rope_scaling type '${String(type)}' unsupported`);
    }
    if (type === 'linear') ropeScale = scaling.factor as number;
  }
  return Object.freeze({
    arch: 'gemma3',
    hidden,
    ffn: num('intermediate_size'),
    layers,
    heads,
    kvHeads: num('num_key_value_heads'),
    headDim: num('head_dim', hidden / heads),
    vocab: num('vocab_size'),
    rmsEps: num('rms_norm_eps', 1e-6),
    queryPreAttnScalar: num('query_pre_attn_scalar', num('head_dim', hidden / heads)),
    ropeTheta: num('rope_theta', 1_000_000),
    ropeLocalTheta: num('rope_local_base_freq', 10_000),
    ropeScale,
    slidingWindow: num('sliding_window', 512),
    sliding: Object.freeze(sliding),
    finalLogitSoftcap: (json.final_logit_softcapping as number | null) ?? 0,
    attnLogitSoftcap: (json.attn_logit_softcapping as number | null) ?? 0,
    tieEmbeddings: true,
    maxPositions: num('max_position_embeddings', 32_768),
  });
}

/** google/gemma-3-1b-it. */
export const GEMMA3_1B: Gemma3Config = gemma3ConfigFromJson({
  hidden_size: 1152,
  intermediate_size: 6912,
  num_hidden_layers: 26,
  num_attention_heads: 4,
  num_key_value_heads: 1,
  head_dim: 256,
  vocab_size: 262144,
  rms_norm_eps: 1e-6,
  query_pre_attn_scalar: 256,
  rope_theta: 1_000_000,
  rope_local_base_freq: 10_000,
  rope_scaling: null,
  sliding_window: 512,
  sliding_window_pattern: 6,
  final_logit_softcapping: null,
  attn_logit_softcapping: null,
  max_position_embeddings: 32_768,
});

/** google/gemma-3-270m-it. */
export const GEMMA3_270M: Gemma3Config = gemma3ConfigFromJson({
  hidden_size: 640,
  intermediate_size: 2048,
  num_hidden_layers: 18,
  num_attention_heads: 4,
  num_key_value_heads: 1,
  head_dim: 256,
  vocab_size: 262144,
  rms_norm_eps: 1e-6,
  query_pre_attn_scalar: 256,
  rope_theta: 1_000_000,
  rope_local_base_freq: 10_000,
  rope_scaling: null,
  sliding_window: 512,
  sliding_window_pattern: 6,
  final_logit_softcapping: null,
  attn_logit_softcapping: null,
  max_position_embeddings: 32_768,
});
