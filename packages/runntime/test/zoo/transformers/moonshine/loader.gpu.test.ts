/** The plugin's Moonshine loader: its load-time warm-up compiles the model's
 *  pipelines without blocking. */
import { PretrainedConfig } from '@huggingface/transformers';
import { describe, expect, it, onTestFinished, vi } from 'vitest';
import { MOONSHINE_TINY } from '../../../../src/zoo/moonshine/config.ts';
import { MoonshineModel } from '../../../../src/zoo/moonshine/model.ts';
import { moonshineLoader } from '../../../../src/zoo/transformers/moonshine/loader.ts';
import { freshDefaultRoot } from '../../../gpu.ts';
import { emptyWeightsUrl, loadRandomWeights } from '../../randomWeights.ts';

/** The hub config.json fields the loader reads, for moonshine-tiny. */
const TINY_CONFIG = {
  hidden_size: MOONSHINE_TINY.dModel,
  intermediate_size: MOONSHINE_TINY.ffn,
  encoder_num_hidden_layers: MOONSHINE_TINY.encLayers,
  decoder_num_hidden_layers: MOONSHINE_TINY.decLayers,
  encoder_num_attention_heads: MOONSHINE_TINY.heads,
  decoder_num_attention_heads: MOONSHINE_TINY.heads,
  attention_bias: false,
  rope_theta: MOONSHINE_TINY.ropeTheta,
  partial_rotary_factor: 0.9,
  vocab_size: MOONSHINE_TINY.vocab,
  max_position_embeddings: MOONSHINE_TINY.maxPositions,
  bos_token_id: MOONSHINE_TINY.bos,
  eos_token_id: MOONSHINE_TINY.eos,
};

describe('moonshineLoader', () => {
  it('compiles no pipeline on first use while it loads', async () => {
    // A device of its own, so no pipeline is compiled yet.
    const fresh = await freshDefaultRoot();
    // The loader builds the full model: bind random values instead of
    // downloading its weights.
    const load = vi.spyOn(MoonshineModel.prototype, 'loadStateDict').mockImplementation(function (
      this: MoonshineModel,
    ) {
      loadRandomWeights(this, fresh);
      return Promise.resolve();
    });
    onTestFinished(() => load.mockRestore());
    const blocking = vi.spyOn(fresh.device, 'createComputePipeline');
    const compiling = vi.spyOn(fresh.device, 'createComputePipelineAsync');

    const model = await moonshineLoader(emptyWeightsUrl()).load(
      '',
      new PretrainedConfig(TINY_CONFIG),
    );
    onTestFinished(async () => {
      await model.dispose();
    });

    expect(compiling).toHaveBeenCalled();
    expect(blocking).not.toHaveBeenCalled();
  });
});
