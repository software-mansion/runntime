import { describe, expect, it } from 'vitest';
import { pending, type ValueMeta } from '../../../src/core/graph/value.ts';
import {
  bufferSource,
  fromSafetensors,
  memoryStateDict,
  type LazyStateDict,
} from '../../../src/core/weights/safetensors.ts';
import type { HiggsDecoderConfig } from '../../../src/zoo/higgs/config.ts';
import { HiggsDecoder } from '../../../src/zoo/higgs/decoder.ts';
import { expectClose } from '../../gpu.ts';
import { CFG, checkpoint, DIM, QUANT_CFG, quantize, type Sd } from './decoderCheckpoint.ts';

/** Loads with stand-ins for the GPU uploads; the CPU table is real. */
async function load(sd: LazyStateDict | Sd, cfg: HiggsDecoderConfig = CFG): Promise<HiggsDecoder> {
  const dec = new HiggsDecoder(cfg);
  await dec.loadStateDict(sd, {
    upload: (_data, shape: ValueMeta) => pending(shape, 'reshape', []),
  });
  return dec;
}

describe('HiggsDecoder', () => {
  it('folds the codebooks, project_out and fc2 into one row per code', async () => {
    const { sd, latents } = checkpoint();
    const dec = await load(sd);
    const frames = 4;
    const codes = Int32Array.from(
      { length: CFG.codebooks * frames },
      (_, i) => (i * 3) % CFG.codebookSize,
    );
    expectClose(dec.latents(codes, frames), latents(codes, frames), { atol: 1e-6, rtol: 1e-5 });
  });

  it.for([8, 4] as const)(
    'folds an int%i project_out and fc2 as their dequantized values, with every conv quantized',
    async (bits) => {
      const { file, dequantized } = quantize(checkpoint(QUANT_CFG).sd, bits);
      const sd = await fromSafetensors(bufferSource(file), { chunkBytes: 0 });
      const quant = await load(sd, QUANT_CFG);
      const float = await load(dequantized, QUANT_CFG);
      const frames = 4;
      const codes = Int32Array.from(
        { length: QUANT_CFG.codebooks * frames },
        (_, i) => (i * 3) % QUANT_CFG.codebookSize,
      );
      // Both fold the same values in f64.
      expect(quant.latents(codes, frames)).toEqual(float.latents(codes, frames));
      const scales = [...quant.namedParameters()]
        .map(([name]) => name)
        .filter((name) => name.endsWith('weightScale'));
      const convs = Object.keys(dequantized)
        .filter((name) => name.startsWith('acoustic_decoder.') && name.endsWith('.weight'))
        .map((name) => name.replace(/weight$/, 'weightScale'));
      expect(scales.sort()).toEqual(convs.sort());
    },
  );

  it('loads a conv_t1 stored as its equivalent conv as its dequantized floats', async () => {
    const { file, dequantized } = quantize(checkpoint(QUANT_CFG).sd, 8, { equivalentConvT: true });
    const quantSd = await fromSafetensors(bufferSource(file), { chunkBytes: 0 });
    const floatSd = memoryStateDict(dequantized);
    const dec = await load(quantSd, QUANT_CFG);
    await load(floatSd, QUANT_CFG);
    const name = 'acoustic_decoder.block.0.conv_t1.weight';
    expect(dec.acoustic_decoder.block.items[0]!.conv_t1.weightScale).toBeUndefined();
    expect(await quantSd.tensors.get(name)!.f32()).toEqual(await floatSd.tensors.get(name)!.f32());
  });

  it('keeps only the decoder of the codec checkpoint', () => {
    const sd = memoryStateDict(checkpoint().sd);
    new HiggsDecoder(CFG).transformStateDict(sd, '');
    const names = [...sd.tensors.keys()];
    expect(names.filter((n) => !n.startsWith('acoustic_decoder.'))).toEqual([
      'codes.weight',
      'codes_bias',
    ]);
  });

  it('keeps the tensors outside its own prefix', () => {
    const raw = checkpoint().sd;
    const sd = memoryStateDict({
      ...Object.fromEntries(Object.entries(raw).map(([name, t]) => [`codec.${name}`, t])),
      'lm.weight': { data: new Float32Array(4), shape: [4] },
    });
    new HiggsDecoder(CFG).transformStateDict(sd, 'codec');
    const names = [...sd.tensors.keys()];
    expect(names).toContain('lm.weight');
    expect(names).toContain('codec.codes.weight');
    expect(names).not.toContain('codec.fc2.weight');
  });

  it.each([
    {
      name: 'a checkpoint without a project_out',
      edit: (sd: Sd) => delete sd['quantizer.quantizers.1.project_out.bias'],
      error: /missing 'quantizer\.quantizers\.1\.project_out\.bias'/,
    },
    {
      name: 'a codebook of another size',
      edit: (sd: Sd) => {
        sd['quantizer.quantizers.0.codebook.embed'] = {
          data: new Float32Array(6 * DIM),
          shape: [6, DIM],
        };
      },
      error: /a codebook embed has shape \[6,4\], expected \[5,4\]/,
    },
  ])('refuses $name before reading any tensor', ({ edit, error }) => {
    const raw = checkpoint().sd;
    edit(raw);
    const sd = memoryStateDict(raw);
    expect(() => new HiggsDecoder(CFG).transformStateDict(sd, '')).toThrow(
      expect.objectContaining({
        code: 'CHECKPOINT_MISMATCH',
        message: expect.stringMatching(error),
      }),
    );
  });

  it('rejects codes out of range and a wrong code count', async () => {
    const dec = await load(checkpoint().sd);
    const codes = new Int32Array(CFG.codebooks * 2);
    expect(() => dec.latents(codes, 3)).toThrow(/6 codes for 3 codebooks × 3 frames/);
    codes[1] = CFG.codebookSize;
    expect(() => dec.latents(codes, 2)).toThrow(/code 5 out of range \[0, 5\)/);
    codes[1] = -1;
    expect(() => dec.latents(codes, 2)).toThrow(/code -1 out of range/);
  });
});
