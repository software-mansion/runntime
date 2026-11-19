/** Reverb's speaker decodes with a quantized f16 codec as with the same codec
 *  loaded with the dequantized floats. */
import { describe, it, onTestFinished } from 'vitest';
import {
  bufferSource,
  fromSafetensors,
  type LazyStateDict,
} from '../../../src/core/weights/safetensors.ts';
import { ReverbModel } from '../../../src/zoo/reverb/model.ts';
import { createReverbSpeaker } from '../../../src/zoo/reverb/speaker.ts';
import { HiggsDecoder } from '../../../src/zoo/higgs/decoder.ts';
import { expectClose, gpu, requireFeature } from '../../gpu.ts';
import { checkpoint, QUANT_CFG, quantize, type Sd } from '../higgs/decoderCheckpoint.ts';

const { root } = await gpu();

/** Weights in ±0.3 keep every layer's output below 1, so f16 rounding stays
 *  small. */
const SPREAD = 0.3;

describe('createReverbSpeaker', () => {
  it('decodes in f16 with an int8 codec', async (ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const { file, dequantized } = quantize(checkpoint(QUANT_CFG, SPREAD).sd, 8);
    const frames = 16;
    const codes = Int32Array.from(
      { length: QUANT_CFG.codebooks * frames },
      (_, i) => (i * 3) % QUANT_CFG.codebookSize,
    );
    const decode = async (sd: LazyStateDict | Sd) => {
      const codec = new HiggsDecoder(QUANT_CFG).half(root);
      onTestFinished(() => codec.dispose());
      await codec.loadStateDict(sd, { root });
      // Made after the load, so the speaker sees the quantized weight's dtype.
      const speaker = createReverbSpeaker(root, new ReverbModel(), codec, { encode: () => [] });
      onTestFinished(() => speaker.dispose());
      return speaker.decode(codes, frames);
    };
    const got = await decode(await fromSafetensors(bufferSource(file), { chunkBytes: 0 }));
    const want = await decode(dequantized);
    const peak = want.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
    expectClose(got, want, { atol: 2e-3 * peak, rtol: 2e-3 });
  });
});
