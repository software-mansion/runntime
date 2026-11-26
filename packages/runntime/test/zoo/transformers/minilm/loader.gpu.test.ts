/** The plugin's MiniLM loader: its load-time warm-up compiles the model's
 *  pipelines without blocking. */
import { PretrainedConfig } from '@huggingface/transformers';
import { describe, expect, it, onTestFinished, vi } from 'vitest';
import { MinilmModel } from '../../../../src/zoo/minilm/model.ts';
import { minilmLoader } from '../../../../src/zoo/transformers/minilm/loader.ts';
import { freshDefaultRoot } from '../../../gpu.ts';
import { emptyWeightsUrl, loadRandomWeights } from '../../randomWeights.ts';

describe('minilmLoader', () => {
  it('compiles no pipeline on first use while it loads', async () => {
    // A device of its own, so no pipeline is compiled yet.
    const fresh = await freshDefaultRoot();
    // The loader builds the full model: bind random values instead of
    // downloading its weights.
    const load = vi.spyOn(MinilmModel.prototype, 'loadStateDict').mockImplementation(function (
      this: MinilmModel,
    ) {
      loadRandomWeights(this, fresh);
      return Promise.resolve();
    });
    onTestFinished(() => load.mockRestore());
    const blocking = vi.spyOn(fresh.device, 'createComputePipeline');
    const compiling = vi.spyOn(fresh.device, 'createComputePipelineAsync');

    const model = await minilmLoader(emptyWeightsUrl()).load('', new PretrainedConfig({}));
    onTestFinished(async () => {
      await model.dispose();
    });

    expect(compiling).toHaveBeenCalled();
    expect(blocking).not.toHaveBeenCalled();
  });
});
