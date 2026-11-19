/** The codec decoder loaded from a quantized checkpoint matches the same
 *  decoder loaded with the dequantized floats in torch's layout. */
import { describe, it, onTestFinished } from 'vitest';
import { toArray } from '../../../src/core/gpu/eval.ts';
import { matrix } from '../../../src/core/graph/value.ts';
import {
  bufferSource,
  fromSafetensors,
  type LazyStateDict,
} from '../../../src/core/weights/safetensors.ts';
import { HiggsDecoder } from '../../../src/zoo/higgs/decoder.ts';
import { expectClose, gpu, requireFeature, upload } from '../../gpu.ts';
import { checkpoint, QUANT_CFG, quantize, type Sd } from './decoderCheckpoint.ts';

const { root, ex } = await gpu();

/** Weights in ±0.3 keep every layer's output below 1, so f16 rounding stays
 *  small. */
const SPREAD = 0.3;

describe('HiggsDecoder from a quantized checkpoint', () => {
  it.for(
    (['f32', 'f16'] as const).flatMap((dtype) =>
      ([8, 4] as const).map((bits) => ({ dtype, bits })),
    ),
  )('$dtype int$bits matches the decoder loaded with the dequantized floats', async (c, ctx) => {
    if (c.dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    const { file, dequantized } = quantize(checkpoint(QUANT_CFG, SPREAD).sd, c.bits);
    const run = async (sd: LazyStateDict | Sd) => {
      const dec = new HiggsDecoder(QUANT_CFG);
      if (c.dtype === 'f16') dec.half(root);
      onTestFinished(() => dec.dispose());
      await dec.loadStateDict(sd, { root });
      const frames = 16;
      const codes = Int32Array.from(
        { length: QUANT_CFG.codebooks * frames },
        (_, i) => (i * 3) % QUANT_CFG.codebookSize,
      );
      const x = upload(root, dec.latents(codes, frames), {
        ...matrix(frames, QUANT_CFG.latent),
        dtype: c.dtype,
      });
      return toArray(dec.forward(x), ex);
    };
    const got = await run(await fromSafetensors(bufferSource(file), { chunkBytes: 0 }));
    const want = await run(dequantized);
    // Both sum in f32 in another order; f16 rounds every layer's output.
    const peak = want.reduce((m, v) => Math.max(m, Math.abs(v)), 0);
    const tol = c.dtype === 'f16' ? 2e-3 : 1e-5;
    expectClose(got, want, { atol: tol * peak, rtol: tol });
  });
});
