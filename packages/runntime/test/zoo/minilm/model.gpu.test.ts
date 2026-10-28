/** MiniLM loaded from a quantized checkpoint, through the same state dict
 *  hooks and embed core createEmbedder uses. The checkpoint has MiniLM's real
 *  widths, two layers, a small vocab and seeded random weights in HF names. */
import { describe, expect, it, onTestFinished } from 'vitest';
import { bufferSource, fromSafetensors } from '../../../src/core/weights/safetensors.ts';
import { MINILM_L6 } from '../../../src/zoo/minilm/config.ts';
import { createMinilmEmbedCore } from '../../../src/zoo/minilm/embedCore.ts';
import { MinilmModel } from '../../../src/zoo/minilm/model.ts';
import { expectClose, gpu, randomF32, requireFeature } from '../../gpu.ts';
import { quantFile, quantizeWeight, safetensorsFile } from '../../core/weights/checkpoint.ts';

const { root } = await gpu();

/** Real widths; the vocab only sizes the float word embedding table. */
const CFG = { ...MINILM_L6, layers: 2, vocab: 64 };

/** Half-width of the uniform random weights, a standard deviation of 0.029
 *  near BERT's 0.02 init, so attention does not saturate and a small weight
 *  change moves the output smoothly. */
const WEIGHT_SPREAD = 0.05;

/** One text for the replay path, two for the packed eager path. */
const TEXT = [2, 7, 1, 8, 28, 18, 28, 45, 9, 4, 52, 3];
const BATCH = [
  [3, 14, 15, 9, 26],
  [5, 35, 8, 9, 7, 9, 32, 38, 46],
];

type Tensors = Record<string, { data: Float32Array; shape: number[] }>;
/** A torch-layout Linear weight, [out, in]. */
type LinearWeight = { data: Float32Array; shape: [number, number] };

/** A BertModel checkpoint in HF names, split into the Linear weights the
 *  quantized file quantizes and every other tensor. */
function bertCheckpoint(): { linear: Record<string, LinearWeight>; other: Tensors } {
  const { hidden, ffn, layers, vocab, maxPositions } = CFG;
  let seed = 1;
  const random = (n: number, center = 0) =>
    randomF32(n, seed++).map((v) => center + WEIGHT_SPREAD * v);
  const other: Tensors = {
    'embeddings.word_embeddings.weight': { data: random(vocab * hidden), shape: [vocab, hidden] },
    'embeddings.position_embeddings.weight': {
      data: random(maxPositions * hidden),
      shape: [maxPositions, hidden],
    },
    'embeddings.token_type_embeddings.weight': { data: random(2 * hidden), shape: [2, hidden] },
    'embeddings.LayerNorm.weight': { data: random(hidden, 1), shape: [hidden] },
    'embeddings.LayerNorm.bias': { data: random(hidden), shape: [hidden] },
  };
  const linear: Record<string, LinearWeight> = {};
  for (let i = 0; i < layers; i++) {
    const hf = (name: string) => `encoder.layer.${i}.${name}`;
    const dense = (name: string, out: number, inF: number) => {
      linear[hf(`${name}.weight`)] = { data: random(out * inF), shape: [out, inF] };
      other[hf(`${name}.bias`)] = { data: random(out), shape: [out] };
    };
    const norm = (name: string) => {
      other[hf(`${name}.weight`)] = { data: random(hidden, 1), shape: [hidden] };
      other[hf(`${name}.bias`)] = { data: random(hidden), shape: [hidden] };
    };
    for (const p of ['query', 'key', 'value']) dense(`attention.self.${p}`, hidden, hidden);
    dense('attention.output.dense', hidden, hidden);
    norm('attention.output.LayerNorm');
    dense('intermediate.dense', ffn, hidden);
    dense('output.dense', hidden, ffn);
    norm('output.LayerNorm');
  }
  return { linear, other };
}

/** The model loaded from `file`, and its embeddings of TEXT and BATCH. */
async function embed(file: Uint8Array, dtype: 'f32' | 'f16') {
  const model = new MinilmModel(CFG);
  onTestFinished(() => model.dispose());
  if (dtype === 'f16') model.half(root);
  await model.loadStateDict(await fromSafetensors(bufferSource(file), { chunkBytes: 0 }), {
    root,
  });
  const core = createMinilmEmbedCore(model);
  onTestFinished(() => core.dispose());
  return { model, embeddings: [await core.embedPooled(TEXT), await core.embedPooledBatch(BATCH)] };
}

const cosine = (a: Float32Array, b: Float32Array) => {
  let dot = 0;
  let aa = 0;
  let bb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i]! * b[i]!;
    aa += a[i]! ** 2;
    bb += b[i]! ** 2;
  }
  return dot / Math.sqrt(aa * bb);
};

/** Tolerance against the same ints dequantized on the CPU and loaded as
 *  floats, which isolates packing and kernel error from quantization error.
 *  Measured worst case: 6e-7 in f32, where only the summation order differs,
 *  and one f16 step (2^-10) in f16, on outputs up to 1.93. f16 allows two
 *  steps at that size. */
const KERNEL_TOLERANCE = {
  f32: { atol: 5e-6, rtol: 0 },
  f16: { atol: 2 ** -9, rtol: 0 },
};

/** Least cosine similarity to the model with the original float weights,
 *  set from the measured worst case across dtypes and group sizes: int8
 *  0.999993 (1 - cos = 7e-6), int4 0.99782 (2.2e-3). Each bound allows
 *  about 5 to 15 times the measured 1 - cos. */
const LEAST_COSINE = { 8: 0.9999, 4: 0.99 };

describe('MiniLM from a quantized checkpoint', () => {
  it.for(
    (['f32', 'f16'] as const).flatMap((dtype) =>
      ([8, 4] as const).flatMap((bits) =>
        [32, 128].map((groupSize) => ({ dtype, bits, groupSize })),
      ),
    ),
  )('$dtype int$bits, groupSize $groupSize', async ({ dtype, bits, groupSize }, ctx) => {
    if (dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    const { linear, other } = bertCheckpoint();
    const quantized = Object.fromEntries(
      Object.entries(linear).map(([name, { data, shape }]) => [
        name,
        quantizeWeight(data, shape, bits, groupSize),
      ]),
    );
    // The same ints times the same f16 scales, as float weights.
    const dequantized: Tensors = Object.fromEntries(
      Object.entries(quantized).map(([name, { ints, scales, shape }]) => [
        name,
        { data: Float32Array.from(ints, (q, i) => q * scales[Math.floor(i / groupSize)]!), shape },
      ]),
    );

    const { model, embeddings: got } = await embed(
      quantFile(bits, groupSize, quantized, other),
      dtype,
    );
    // Every Linear loaded quantized, so no layer took the float matmul.
    const scaleNames = [...model.namedParameters()]
      .map(([name]) => name)
      .filter((name) => name.endsWith('weightScale'));
    expect(scaleNames).toEqual(
      Array.from({ length: CFG.layers }, (_, i) =>
        ['qkv', 'out', 'up', 'down'].map((m) => `layers.${i}.${m}.weightScale`),
      ).flat(),
    );

    const reference = await embed(safetensorsFile({ ...other, ...dequantized }), dtype);
    const float = await embed(safetensorsFile({ ...other, ...linear }), dtype);
    for (let i = 0; i < got.length; i++) {
      expectClose(got[i]!, reference.embeddings[i]!, KERNEL_TOLERANCE[dtype]);
      expect(cosine(got[i]!, float.embeddings[i]!)).toBeGreaterThan(LEAST_COSINE[bits]);
    }
  });
});
