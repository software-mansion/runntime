import type { HiggsDecoderConfig } from '../../../src/zoo/higgs/config.ts';
import { randomF32 } from '../../gpu.ts';
import {
  dequantizeWeight,
  mixedQuantFile,
  quantizeWeight,
  type BitsQuantEntry,
} from '../../core/weights/checkpoint.ts';

/** The real decoder's layout, small. */
export const CFG: HiggsDecoderConfig = {
  codebooks: 3,
  codebookSize: 5,
  latent: 8,
  channels: 8,
  strides: [2],
  kernel: 3,
};
/** The quantizer's codebook width and its project_out width, which only the
 *  load reads. */
export const DIM = 4;
const HIDDEN = 6;

export type Sd = Record<string, { data: Float32Array; shape: number[] }>;

/** Random weights in [-spread, spread) under the codec checkpoint's names,
 *  and the decoder input they give for codes [codebooks, frames]: each
 *  codebook's row through its project_out, summed, then fc2, in f64. */
export function checkpoint(cfg: HiggsDecoderConfig = CFG, spread = 1) {
  const sd: Sd = {};
  let seed = 1;
  const param = (name: string, shape: number[]) => {
    const data = randomF32(
      shape.reduce((a, b) => a * b, 1),
      seed++,
    ).map((v) => v * spread);
    sd[name] = { data, shape };
    return data;
  };
  const { codebooks, codebookSize, latent, channels, strides, kernel: k } = cfg;
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
  strides.forEach((stride, i) => {
    const B = `${D}.block.${i}`;
    const cIn = channels / 2 ** i;
    const out = cIn / 2;
    snake(`${B}.snake1`, cIn);
    param(`${B}.conv_t1.weight`, [cIn, out, 2 * stride]);
    param(`${B}.conv_t1.bias`, [out]);
    for (const j of [1, 2, 3]) {
      const R = `${B}.res_unit${j}`;
      snake(`${R}.snake1`, out);
      conv(`${R}.conv1`, out, out, k);
      snake(`${R}.snake2`, out);
      conv(`${R}.conv2`, out, out, 1);
    }
  });
  const out = channels / 2 ** strides.length;
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

/** CFG with every conv's input channels a multiple of CONV_GROUP, so each
 *  conv loads quantized. */
export const QUANT_CFG: HiggsDecoderConfig = { ...CFG, channels: 16 };

/** Input channels per scale group of a conv. */
const CONV_GROUP = 8;
/** Values per scale group of project_out and fc2, which the load folds. */
const FOLD_GROUP = 2;

/** A conv weight's file rows [cOut, cIn·kernel]: for each value, its index
 *  in torch's layout. A transposed conv is [in, out, kernel] in torch and
 *  its equivalent conv in the file, with the kernel reversed. */
function convRows(name: string, shape: number[]) {
  const transposed = /\.conv_t\d+\.weight$/.test(name);
  const [a, b, kernel] = shape as [number, number, number];
  const [cOut, cIn] = transposed ? [b, a] : [a, b];
  const index = new Int32Array(cOut * cIn * kernel);
  for (let o = 0; o < cOut; o++) {
    for (let c = 0; c < cIn; c++) {
      for (let t = 0; t < kernel; t++) {
        index[(o * cIn + c) * kernel + t] = transposed
          ? (c * cOut + o) * kernel + kernel - 1 - t
          : (o * cIn + c) * kernel + t;
      }
    }
  }
  return { index, cOut, kernel };
}

/** `sd` as a quantized file with every conv weight, project_out and fc2
 *  quantized, and `sd` with each of those replaced by its dequantized values.
 *  Codebook tables, biases and Snake alphas stay float. */
export function quantize(sd: Sd, bits: 4 | 8): { file: Uint8Array; dequantized: Sd } {
  const entries: Record<string, BitsQuantEntry> = {};
  const floats: Sd = {};
  const dequantized: Sd = { ...sd };
  for (const [name, { data, shape }] of Object.entries(sd)) {
    if (name === 'fc2.weight' || name.endsWith('.project_out.weight')) {
      const entry = quantizeWeight(data, shape as [number, number], bits, FOLD_GROUP);
      entries[name] = { ...entry, bits };
      dequantized[name] = { data: dequantizeWeight(entry), shape };
    } else if (name.startsWith('acoustic_decoder.') && name.endsWith('.weight')) {
      const { index, cOut, kernel } = convRows(name, shape);
      const rows = Float32Array.from(index, (i) => data[i]!);
      const entry = quantizeWeight(rows, [cOut, rows.length / cOut], bits, CONV_GROUP * kernel);
      entries[name] = { ...entry, bits };
      const values = dequantizeWeight(entry);
      const back = new Float32Array(data.length);
      index.forEach((i, r) => (back[i] = values[r]!));
      dequantized[name] = { data: back, shape };
    } else {
      floats[name] = { data, shape };
    }
  }
  return { file: mixedQuantFile(entries, floats), dequantized };
}
