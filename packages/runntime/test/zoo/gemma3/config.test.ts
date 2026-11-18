import { describe, expect, it } from 'vitest';
import { GEMMA3_1B, gemma3ConfigFromJson } from '../../../src/zoo/gemma3/config.ts';

/** config.json of google/gemma-3-1b-it. */
const GEMMA3_1B_JSON = {
  architectures: ['Gemma3ForCausalLM'],
  attn_logit_softcapping: null,
  final_logit_softcapping: null,
  head_dim: 256,
  hidden_activation: 'gelu_pytorch_tanh',
  hidden_size: 1152,
  intermediate_size: 6912,
  max_position_embeddings: 32768,
  model_type: 'gemma3_text',
  num_attention_heads: 4,
  num_hidden_layers: 26,
  num_key_value_heads: 1,
  query_pre_attn_scalar: 256,
  rms_norm_eps: 1e-6,
  rope_local_base_freq: 10000,
  rope_scaling: null,
  rope_theta: 1000000,
  sliding_window: 512,
  sliding_window_pattern: 6,
  vocab_size: 262144,
};

describe('gemma3ConfigFromJson', () => {
  it('reads the 1B checkpoint config like the built-in one', () => {
    expect(gemma3ConfigFromJson(GEMMA3_1B_JSON)).toEqual(GEMMA3_1B);
  });

  it('makes every 6th layer global', () => {
    const global = GEMMA3_1B.sliding.flatMap((s, i) => (s ? [] : [i]));
    expect(global).toEqual([5, 11, 17, 23]);
  });

  it('prefers layer_types and reads a multimodal text_config', () => {
    const layer_types = Array.from({ length: 26 }, (_, i) =>
      i % 2 ? 'full_attention' : 'sliding_attention',
    );
    const cfg = gemma3ConfigFromJson({ text_config: { ...GEMMA3_1B_JSON, layer_types } });
    expect(cfg.sliding).toEqual(layer_types.map((t) => t === 'sliding_attention'));
  });

  it('reads linear RoPE scaling', () => {
    const rope_scaling = { rope_type: 'linear', factor: 8 };
    expect(gemma3ConfigFromJson({ ...GEMMA3_1B_JSON, rope_scaling }).ropeScale).toBe(8);
  });

  it.each([
    [{ hidden_activation: 'relu' }, /activation 'relu'/],
    [{ attention_bias: true }, /attention_bias/],
    [{ rope_scaling: { rope_type: 'yarn', factor: 2 } }, /rope_scaling type 'yarn'/],
    [{ layer_types: ['sliding_attention'] }, /layer_types length/],
    [{ hidden_size: undefined }, /'hidden_size' missing/],
  ])('refuses %j', (patch, error) => {
    expect(() => gemma3ConfigFromJson({ ...GEMMA3_1B_JSON, ...patch })).toThrow(error);
  });
});
