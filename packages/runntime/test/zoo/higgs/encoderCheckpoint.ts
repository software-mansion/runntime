import type { HiggsEncoderConfig } from '../../../src/zoo/higgs/config.ts';
import { randomF32 } from '../../gpu.ts';
import {
  dequantizeWeight,
  mixedQuantFile,
  quantizeWeight,
  type BitsQuantEntry,
} from '../../core/weights/checkpoint.ts';

/** The real encoder's layout, small. */
export const CFG: HiggsEncoderConfig = {
  sampleRate: 24_000,
  semanticRate: 16_000,
  acoustic: { channels: 4, strides: [3], kernel: 3, outKernel: 3, latent: 4 },
  hubert: {
    convDim: 8,
    convKernels: [4, 2],
    convStrides: [2, 2],
    hidden: 8,
    heads: 2,
    layers: 2,
    ffn: 12,
    posKernel: 4,
    posGroups: 2,
    pad: 2,
    frameStride: 2,
  },
  semanticBlocks: 1,
  semanticDilations: [1, 1],
  semanticKernel: 3,
  codebooks: 3,
  codebookSize: 5,
  codebookDim: 4,
};

export type Sd = Record<string, { data: Float32Array; shape: number[] }>;

/** Random weights under the codec checkpoint's names, decoder and training
 *  state included. */
export function checkpoint(cfg: HiggsEncoderConfig = CFG): Sd {
  const sd: Sd = {};
  let seed = 1;
  const param = (name: string, shape: number[]) => {
    const n = shape.reduce((a, b) => a * b, 1);
    sd[name] = { data: randomF32(n, seed++), shape };
  };
  const conv = (name: string, cIn: number, cOut: number, k: number, bias = true) => {
    param(`${name}.weight`, [cOut, cIn, k]);
    if (bias) param(`${name}.bias`, [cOut]);
  };
  const norm = (name: string, n: number) => {
    param(`${name}.weight`, [n]);
    param(`${name}.bias`, [n]);
  };
  const linear = (name: string, cIn: number, cOut: number) => {
    param(`${name}.weight`, [cOut, cIn]);
    param(`${name}.bias`, [cOut]);
  };
  const snake = (name: string, c: number) => param(`${name}.alpha`, [1, c, 1]);

  const { acoustic: ac, hubert: hu } = cfg;
  const A = 'acoustic_encoder';
  conv(`${A}.conv1`, 1, ac.channels, ac.kernel);
  ac.strides.forEach((s, i) => {
    const c = ac.channels * 2 ** i;
    for (const j of [1, 2, 3]) {
      const R = `${A}.block.${i}.res_unit${j}`;
      snake(`${R}.snake1`, c);
      conv(`${R}.conv1`, c, c, ac.kernel);
      snake(`${R}.snake2`, c);
      conv(`${R}.conv2`, c, c, 1);
    }
    snake(`${A}.block.${i}.snake1`, c);
    conv(`${A}.block.${i}.conv1`, c, 2 * c, 2 * s);
  });
  const last = ac.channels * 2 ** ac.strides.length;
  snake(`${A}.snake1`, last);
  conv(`${A}.conv2`, last, ac.latent, ac.outKernel);

  const H = 'semantic_model';
  hu.convKernels.forEach((k, i) => {
    conv(
      `${H}.feature_extractor.conv_layers.${i}.conv`,
      i === 0 ? 1 : hu.convDim,
      hu.convDim,
      k,
      false,
    );
  });
  norm(`${H}.feature_extractor.conv_layers.0.layer_norm`, hu.convDim);
  norm(`${H}.feature_projection.layer_norm`, hu.convDim);
  linear(`${H}.feature_projection.projection`, hu.convDim, hu.hidden);
  const P = `${H}.encoder.pos_conv_embed.conv`;
  param(`${P}.parametrizations.weight.original0`, [1, 1, hu.posKernel]);
  param(`${P}.parametrizations.weight.original1`, [
    hu.hidden,
    hu.hidden / hu.posGroups,
    hu.posKernel,
  ]);
  param(`${P}.bias`, [hu.hidden]);
  norm(`${H}.encoder.layer_norm`, hu.hidden);
  for (let i = 0; i < hu.layers; i++) {
    const L = `${H}.encoder.layers.${i}`;
    for (const p of ['q', 'k', 'v', 'out'])
      linear(`${L}.attention.${p}_proj`, hu.hidden, hu.hidden);
    norm(`${L}.layer_norm`, hu.hidden);
    linear(`${L}.feed_forward.intermediate_dense`, hu.hidden, hu.ffn);
    linear(`${L}.feed_forward.output_dense`, hu.ffn, hu.hidden);
    norm(`${L}.final_layer_norm`, hu.hidden);
  }

  const S = 'encoder_semantic';
  const k = cfg.semanticKernel;
  conv(`${S}.conv`, hu.hidden, hu.hidden, k, false);
  for (let i = 0; i < cfg.semanticBlocks; i++) {
    cfg.semanticDilations.forEach((_, j) => {
      conv(`${S}.conv_blocks.${i}.res_units.${j}.conv1`, hu.hidden, hu.hidden, k, false);
      conv(`${S}.conv_blocks.${i}.res_units.${j}.conv2`, hu.hidden, hu.hidden, 1, false);
    });
    conv(`${S}.conv_blocks.${i}.conv`, hu.hidden, hu.hidden, k);
  }

  const hidden = ac.latent + hu.hidden;
  linear('fc', hidden, hidden);
  for (let i = 0; i < cfg.codebooks; i++) {
    const Q = `quantizer.quantizers.${i}`;
    linear(`${Q}.project_in`, hidden, cfg.codebookDim);
    linear(`${Q}.project_out`, cfg.codebookDim, hidden);
    param(`${Q}.codebook.embed`, [cfg.codebookSize, cfg.codebookDim]);
    param(`${Q}.codebook.embed_avg`, [cfg.codebookSize, cfg.codebookDim]);
    param(`${Q}.codebook.cluster_size`, [cfg.codebookSize]);
    param(`${Q}.codebook.inited`, [1]);
  }

  // Read by the decoder only.
  linear('fc1', hidden, hu.hidden);
  linear('fc2', hidden, ac.latent);
  conv('acoustic_decoder.conv1', ac.latent, last, ac.kernel);
  conv('decoder_semantic.conv1', hu.hidden, hu.hidden, k, false);
  return sd;
}

/** CFG with every conv but the two on one input channel, and every Linear,
 *  loadable quantized. */
export const QUANT_CFG: HiggsEncoderConfig = {
  ...CFG,
  acoustic: { ...CFG.acoustic, channels: 8, latent: 8 },
  hubert: { ...CFG.hubert, ffn: 16 },
};

/** Input channels per scale group of an acoustic conv. */
const CONV_GROUP = 4;
/** Values per scale group of a Linear. */
const LINEAR_GROUP = 4;

const POS = 'semantic_model.encoder.pos_conv_embed.conv';

/** The position conv's weight norm folded, [hidden, width, kernel], in f64. */
function foldedPositionWeight(sd: Sd): { data: Float32Array; shape: number[] } {
  const g = sd[`${POS}.parametrizations.weight.original0`]!.data;
  const { data: v, shape } = sd[`${POS}.parametrizations.weight.original1`]!;
  const kernel = shape[2]!;
  const norm = new Float64Array(kernel);
  v.forEach((x, i) => (norm[i % kernel]! += x * x));
  return { data: v.map((x, i) => (x * g[i % kernel]!) / Math.sqrt(norm[i % kernel]!)), shape };
}

/** `sd` as a quantized file the way the codec's int files hold it: the
 *  position conv as its folded weight, the convs in int8, the acoustic ones
 *  in groups of input channels and HuBERT's and the semantic ones with one
 *  scale per output channel, and every Linear but the quantizer's in `bits`,
 *  in small groups. Also `sd` with each of those replaced by its dequantized
 *  values. The quantizer, norms, biases and Snake alphas stay float. */
export function quantize(sd: Sd, bits: 4 | 8): { file: Uint8Array; dequantized: Sd } {
  const source: Sd = { ...sd, [`${POS}.weight`]: foldedPositionWeight(sd) };
  delete source[`${POS}.parametrizations.weight.original0`];
  delete source[`${POS}.parametrizations.weight.original1`];
  const entries: Record<string, BitsQuantEntry> = {};
  const floats: Sd = {};
  const dequantized: Sd = { ...source };
  for (const [name, { data, shape }] of Object.entries(source)) {
    const encoder = /^(acoustic_encoder|semantic_model|encoder_semantic|fc)\./.test(name);
    if (!encoder || !name.endsWith('.weight') || shape.length === 1) {
      floats[name] = { data, shape };
      continue;
    }
    // torch [out, in, kernel] is [out, in·kernel] in the file, the same order.
    const [cOut, cIn, kernel = 1] = shape as [number, number, number?];
    const cols = cIn * kernel;
    const groupSize =
      shape.length === 2
        ? LINEAR_GROUP
        : name.startsWith('acoustic_encoder.') && cIn % CONV_GROUP === 0
          ? CONV_GROUP * kernel
          : cols;
    const entryBits = shape.length === 2 ? bits : 8;
    const entry = quantizeWeight(data, [cOut, cols], entryBits, groupSize);
    entries[name] = { ...entry, bits: entryBits };
    dequantized[name] = { data: dequantizeWeight(entry), shape };
  }
  return { file: mixedQuantFile(entries, floats), dequantized };
}
