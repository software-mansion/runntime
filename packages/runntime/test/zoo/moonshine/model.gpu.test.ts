/** Moonshine loaded from a quantized checkpoint, through the state dict hooks
 *  the transcriber uses: the quantized q|k|v row stack, the scaled
 *  cross-attention q, and every Linear on the quantized matmul. The
 *  checkpoint has Tiny's real widths, one layer each, a small vocab and seeded
 *  random weights in HF names. */
import { describe, expect, it, onTestFinished } from 'vitest';
import { tensor, toArray, type Value } from '../../../src/core/index.ts';
import { bufferSource, fromSafetensors } from '../../../src/core/weights/safetensors.ts';
import { MOONSHINE_TINY } from '../../../src/zoo/moonshine/config.ts';
import { MoonshineModel } from '../../../src/zoo/moonshine/model.ts';
import { buildRopeTables } from '../../../src/zoo/moonshine/rope.ts';
import { stemFrames } from '../../../src/zoo/moonshine/convStem.ts';
import { expectClose, gpu, randomF32, requireFeature, upload } from '../../gpu.ts';
import { quantFile, quantizeWeight, safetensorsFile } from '../../core/weights/checkpoint.ts';

const { root, ex } = await gpu();

/** Real widths; the vocab only sizes the float embedding table. */
const CFG = { ...MOONSHINE_TINY, encLayers: 1, decLayers: 1, vocab: 64 };

/** Half-width of the uniform random weights, a standard deviation of 0.029
 *  near the usual 0.02 init, so attention does not saturate and a small
 *  weight change moves the output smoothly. */
const WEIGHT_SPREAD = 0.05;

/** One second of audio, 41 encoder frames. */
const AUDIO_SAMPLES = 16_000;
/** The decoder runs these tokens at once: BOS and two more. */
const TOKENS = [1, 7, 42];

type Tensors = Record<string, { data: Float32Array; shape: number[] }>;
/** A torch-layout Linear weight, [out, in]. */
type LinearWeight = { data: Float32Array; shape: [number, number] };

/** A Moonshine checkpoint in HF names, split into the Linear weights the
 *  quantized file quantizes and every other tensor. */
function moonshineCheckpoint(): { linear: Record<string, LinearWeight>; other: Tensors } {
  const { dModel: d, ffn, vocab } = CFG;
  let seed = 1;
  const random = (n: number, center = 0) =>
    randomF32(n, seed++).map((v) => center + WEIGHT_SPREAD * v);
  const other: Tensors = {
    'model.decoder.embed_tokens.weight': { data: random(vocab * d), shape: [vocab, d] },
    'model.encoder.conv1.weight': { data: random(d * 127), shape: [d, 1, 127] },
    'model.encoder.groupnorm.weight': { data: random(d, 1), shape: [d] },
    'model.encoder.groupnorm.bias': { data: random(d), shape: [d] },
    'model.encoder.conv2.weight': { data: random(2 * d * d * 7), shape: [2 * d, d, 7] },
    'model.encoder.conv2.bias': { data: random(2 * d), shape: [2 * d] },
    'model.encoder.conv3.weight': { data: random(d * 2 * d * 3), shape: [d, 2 * d, 3] },
    'model.encoder.conv3.bias': { data: random(d), shape: [d] },
    'model.encoder.layer_norm.weight': { data: random(d, 1), shape: [d] },
    'model.decoder.norm.weight': { data: random(d, 1), shape: [d] },
  };
  const linear: Record<string, LinearWeight> = {};
  const dense = (name: string, out: number, inF: number, bias = false) => {
    linear[`${name}.weight`] = { data: random(out * inF), shape: [out, inF] };
    if (bias) other[`${name}.bias`] = { data: random(out), shape: [out] };
  };
  const norm = (name: string) => {
    other[`${name}.weight`] = { data: random(d, 1), shape: [d] };
  };
  const attention = (name: string) => {
    for (const p of ['q', 'k', 'v', 'o']) dense(`${name}.${p}_proj`, d, d);
  };
  const enc = 'model.encoder.layers.0';
  norm(`${enc}.input_layernorm`);
  attention(`${enc}.self_attn`);
  norm(`${enc}.post_attention_layernorm`);
  dense(`${enc}.mlp.fc1`, ffn, d, true);
  dense(`${enc}.mlp.fc2`, d, ffn, true);
  const dec = 'model.decoder.layers.0';
  norm(`${dec}.input_layernorm`);
  attention(`${dec}.self_attn`);
  norm(`${dec}.post_attention_layernorm`);
  attention(`${dec}.encoder_attn`);
  norm(`${dec}.final_layernorm`);
  dense(`${dec}.mlp.fc1`, 2 * ffn, d, true);
  dense(`${dec}.mlp.fc2`, d, ffn, true);
  return { linear, other };
}

/** The model loaded from `file`, and its decoder output [TOKENS, dModel] for
 *  one second of seeded audio. */
async function decode(file: Uint8Array, dtype: 'f32' | 'f16') {
  const model = new MoonshineModel(CFG);
  onTestFinished(() => model.dispose());
  if (dtype === 'f16') model.half(root);
  await model.loadStateDict(await fromSafetensors(bufferSource(file), { chunkBytes: 0 }), {
    root,
  });

  const audio = randomF32(AUDIO_SAMPLES, 99).map((v) => 0.3 * v);
  const x = tensor(root, audio, { elems: audio.length, dtype, dims: [audio.length, 1] });
  onTestFinished(() => x.buffer.destroy());
  const table = (positions: number): [Value, Value] => {
    const { cos, sin } = buildRopeTables(positions, CFG);
    return [
      upload(root, cos, [positions, CFG.headDim]),
      upload(root, sin, [positions, CFG.headDim]),
    ];
  };
  const encOut = model.encoder.forward(
    model.stem.forward(x),
    ...table(stemFrames(audio.length).conv3),
  );
  const cross = model.decoder.precomputeCrossKV(encOut);
  const h = model.decoder.forward(model.embed(TOKENS), ...table(TOKENS.length), cross);
  return { model, output: await toArray(h, ex) };
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
 *  Measured worst case: 4e-6 in f32, where only the summation order differs,
 *  and 2^-8 in f16 on outputs up to 3.6, where the reference rounds each
 *  int times scale to f16 before the stem, encoder and decoder stack up the
 *  difference. f16 allows two f16 steps at that size on top. */
const KERNEL_TOLERANCE = {
  f32: { atol: 5e-6, rtol: 0 },
  f16: { atol: 2 ** -7, rtol: 0 },
};

/** Least cosine similarity to the model with the original float weights,
 *  set from the measured worst case across dtypes: int8 0.99997
 *  (1 - cos = 2.7e-5), int4 0.9921 (7.9e-3). Each bound allows about 2.5 to
 *  4 times the measured 1 - cos. */
const LEAST_COSINE = { 8: 0.9999, 4: 0.98 };

describe('Moonshine from a quantized checkpoint', () => {
  it.for(
    (['f32', 'f16'] as const).flatMap((dtype) =>
      ([8, 4] as const).map((bits) => ({ dtype, bits, groupSize: 32 })),
    ),
  )('$dtype int$bits, groupSize $groupSize', async ({ dtype, bits, groupSize }, ctx) => {
    if (dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    const { linear, other } = moonshineCheckpoint();
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

    const { model, output: got } = await decode(
      quantFile(bits, groupSize, quantized, other),
      dtype,
    );
    // Every Linear loaded quantized, so no layer took the float matmul.
    const scaleNames = [...model.namedParameters()]
      .map(([name]) => name)
      .filter((name) => name.endsWith('weightScale'))
      .sort();
    expect(scaleNames).toEqual(
      [
        ...['attn.qkv', 'attn.out', 'mlp.fc1', 'mlp.fc2'].map((m) => `encoder.layers.0.${m}`),
        ...['selfAttn.qkv', 'selfAttn.out', 'crossAttn.q', 'crossAttn.kv', 'crossAttn.out'].map(
          (m) => `decoder.layers.0.${m}`,
        ),
        ...['mlp.fc1', 'mlp.fc2'].map((m) => `decoder.layers.0.${m}`),
      ]
        .map((m) => `${m}.weightScale`)
        .sort(),
    );

    const reference = await decode(safetensorsFile({ ...other, ...dequantized }), dtype);
    const float = await decode(safetensorsFile({ ...other, ...linear }), dtype);
    expectClose(got, reference.output, KERNEL_TOLERANCE[dtype]);
    expect(cosine(got, float.output)).toBeGreaterThan(LEAST_COSINE[bits]);
  });
});
