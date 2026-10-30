import { describe, expect, it } from 'vitest';
import { pending, type ValueMeta } from '../../../src/core/graph/value.ts';
import { memoryStateDict } from '../../../src/core/weights/safetensors.ts';
import type { HiggsDecoderConfig } from '../../../src/zoo/higgs/config.ts';
import { HiggsDecoder } from '../../../src/zoo/higgs/decoder.ts';
import { expectClose, randomF32 } from '../../gpu.ts';

// The real decoder's layout, small.
const CFG: HiggsDecoderConfig = {
  codebooks: 3,
  codebookSize: 5,
  latent: 8,
  channels: 8,
  strides: [2],
  kernel: 3,
};
/** The quantizer's codebook width and its project_out width, which only the
 *  load reads. */
const DIM = 4;
const HIDDEN = 6;

type Sd = Record<string, { data: Float32Array; shape: number[] }>;

/** Random weights under the codec checkpoint's names, and the decoder input
 *  they give for codes [codebooks, frames]: each codebook's row through its
 *  project_out, summed, then fc2, in f64. */
function checkpoint() {
  const sd: Sd = {};
  let seed = 1;
  const param = (name: string, shape: number[]) => {
    const data = randomF32(
      shape.reduce((a, b) => a * b, 1),
      seed++,
    );
    sd[name] = { data, shape };
    return data;
  };
  const { codebooks, codebookSize, latent, channels, kernel: k } = CFG;
  const books = Array.from({ length: codebooks }, (_, i) => {
    const q = `quantizer.quantizers.${i}`;
    param(`${q}.codebook.embed_avg`, [codebookSize, DIM]);
    param(`${q}.project_in.weight`, [DIM, HIDDEN]);
    return {
      embed: param(`${q}.codebook.embed`, [codebookSize, DIM]),
      w: param(`${q}.project_out.weight`, [HIDDEN, DIM]),
      b: param(`${q}.project_out.bias`, [HIDDEN]),
    };
  });
  const fc2 = { w: param('fc2.weight', [latent, HIDDEN]), b: param('fc2.bias', [latent]) };
  param('fc1.weight', [3, HIDDEN]);
  param('acoustic_encoder.conv1.weight', [4, 1, 7]);
  param('semantic_model.feature_projection.weight', [3, 3]);

  const D = 'acoustic_decoder';
  const conv = (name: string, cIn: number, cOut: number, kk: number) => {
    param(`${name}.weight`, [cOut, cIn, kk]);
    param(`${name}.bias`, [cOut]);
  };
  const snake = (name: string, c: number) => param(`${name}.alpha`, [1, c, 1]);
  conv(`${D}.conv1`, latent, channels, k);
  const out = channels / 2;
  snake(`${D}.block.0.snake1`, channels);
  param(`${D}.block.0.conv_t1.weight`, [channels, out, 4]);
  param(`${D}.block.0.conv_t1.bias`, [out]);
  for (const j of [1, 2, 3]) {
    const R = `${D}.block.0.res_unit${j}`;
    snake(`${R}.snake1`, out);
    conv(`${R}.conv1`, out, out, k);
    snake(`${R}.snake2`, out);
    conv(`${R}.conv2`, out, out, 1);
  }
  snake(`${D}.snake1`, out);
  conv(`${D}.conv2`, out, 1, k);

  const latents = (codes: Int32Array, frames: number): Float64Array => {
    const rows = new Float64Array(frames * latent);
    for (let t = 0; t < frames; t++) {
      const h = new Float64Array(HIDDEN);
      books.forEach((book, c) => {
        const code = codes[c * frames + t]!;
        for (let j = 0; j < HIDDEN; j++) {
          let v = book.b[j]!;
          for (let a = 0; a < DIM; a++) v += book.embed[code * DIM + a]! * book.w[j * DIM + a]!;
          h[j]! += v;
        }
      });
      for (let o = 0; o < latent; o++) {
        let v = fc2.b[o]!;
        for (let j = 0; j < HIDDEN; j++) v += fc2.w[o * HIDDEN + j]! * h[j]!;
        rows[t * latent + o] = v;
      }
    }
    return rows;
  };
  return { sd, latents };
}

/** Loads with stand-ins for the GPU uploads; the CPU table is real. */
async function load(sd: Sd): Promise<HiggsDecoder> {
  const dec = new HiggsDecoder(CFG);
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
