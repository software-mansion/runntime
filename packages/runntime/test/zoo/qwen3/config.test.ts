import { describe, expect, it } from 'vitest';
import { QWEN3_0_6B, qwen3ConfigFromJson } from '../../../src/zoo/qwen3/config.ts';
import { decoderConfigFromJson } from '../../../src/zoo/tasks/nlp/textGeneration.ts';

/** config.json of Qwen/Qwen3-0.6B. */
const QWEN3_0_6B_JSON = {
  architectures: ['Qwen3ForCausalLM'],
  attention_bias: false,
  head_dim: 128,
  hidden_act: 'silu',
  hidden_size: 1024,
  intermediate_size: 3072,
  max_position_embeddings: 40960,
  max_window_layers: 28,
  model_type: 'qwen3',
  num_attention_heads: 16,
  num_hidden_layers: 28,
  num_key_value_heads: 8,
  rms_norm_eps: 1e-6,
  rope_scaling: null,
  rope_theta: 1000000,
  sliding_window: null,
  tie_word_embeddings: true,
  use_sliding_window: false,
  vocab_size: 151936,
};

describe('qwen3ConfigFromJson', () => {
  it('reads the 0.6B checkpoint config like the built-in one', () => {
    expect(qwen3ConfigFromJson(QWEN3_0_6B_JSON)).toEqual(QWEN3_0_6B);
    expect(decoderConfigFromJson(QWEN3_0_6B_JSON)).toEqual(QWEN3_0_6B);
  });

  it('runs full attention everywhere, scaled by head_dim', () => {
    expect(QWEN3_0_6B.sliding.every((s) => !s)).toBe(true);
    expect(QWEN3_0_6B.queryPreAttnScalar).toBe(128);
  });

  it('reads an untied LM head', () => {
    const cfg = qwen3ConfigFromJson({ ...QWEN3_0_6B_JSON, tie_word_embeddings: false });
    expect(cfg.tieEmbeddings).toBe(false);
  });

  it.each([
    [{ hidden_act: 'gelu' }, /activation 'gelu'/],
    [{ attention_bias: true }, /attention_bias/],
    [{ use_sliding_window: true }, /sliding window/],
    [{ rope_scaling: { rope_type: 'yarn', factor: 4 } }, /rope_scaling/],
  ])('refuses %j', (patch, error) => {
    expect(() => qwen3ConfigFromJson({ ...QWEN3_0_6B_JSON, ...patch })).toThrow(error);
  });

  it('refuses another model_type', () => {
    expect(() => decoderConfigFromJson({ ...QWEN3_0_6B_JSON, model_type: 'llama' })).toThrow(
      /model_type 'llama'/,
    );
  });
});
