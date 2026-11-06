/** EmbeddingGemma 2's text model and its vision and audio towers loaded from
 *  a quantized checkpoint, through the state dict hooks and encoders
 *  createEmbedder uses: the quantized q|k|v row stacks, the scaled PLE
 *  projection, and every Linear on the quantized matmul. Each checkpoint has
 *  the real widths, one or two layers, a small vocab or position table and
 *  seeded random weights in HF names. */
import { describe, expect, it, onTestFinished } from 'vitest';
import { defaultRoot, gpuExecutor, type nn } from '../../../src/core/index.ts';
import { bufferSource, fromSafetensors } from '../../../src/core/weights/safetensors.ts';
import { encodeAudio } from '../../../src/zoo/embeddinggemma2/audioEncoder.ts';
import { EmbeddingGemma2AudioModel } from '../../../src/zoo/embeddinggemma2/audioModel.ts';
import {
  EMBEDDINGGEMMA2_AUDIO,
  EMBEDDINGGEMMA2_TEXT,
  EMBEDDINGGEMMA2_VISION,
  type EmbeddingGemma2Config,
} from '../../../src/zoo/embeddinggemma2/config.ts';
import { createEmbeddingGemma2EmbedCore } from '../../../src/zoo/embeddinggemma2/embedCore.ts';
import { EmbeddingGemma2Model } from '../../../src/zoo/embeddinggemma2/model.ts';
import { encodeImage } from '../../../src/zoo/embeddinggemma2/visionEncoder.ts';
import { EmbeddingGemma2VisionModel } from '../../../src/zoo/embeddinggemma2/visionModel.ts';
import { imageBuffer } from '../../../src/zoo/tasks/cv/image.ts';
import { expectClose, gpu, randomF32, requireFeature } from '../../gpu.ts';
import { quantFile, quantizeWeight, safetensorsFile } from '../../core/weights/checkpoint.ts';

const { root } = await gpu();

/** Real widths; one sliding and one full layer, a vocab that only sizes the
 *  CPU embedding table. */
const TEXT: EmbeddingGemma2Config = {
  ...EMBEDDINGGEMMA2_TEXT,
  layers: 2,
  layerTypes: ['sliding', 'full'],
  vocab: 64,
};
/** Real widths, one layer, a position table that covers the test image. */
const VISION = { ...EMBEDDINGGEMMA2_VISION, layers: 1, positions: 8 };
/** Real widths but the feed-forward, one layer. The four feed-forward
 *  weights are two thirds of a real layer, and building, quantizing and
 *  loading them three times is what a case's time goes on. */
const AUDIO = { ...EMBEDDINGGEMMA2_AUDIO, layers: 1, ffn: 1024 };

/** Half-width of the uniform random weights, a standard deviation of 0.029
 *  near the usual 0.02 init, so attention does not saturate and a small
 *  weight change moves the output smoothly. */
const WEIGHT_SPREAD = 0.05;
/** k_norm's scale: q and k are RMS-normed per head and the scores carry no
 *  1/√d, so a k scale near 1 gives scores of about √d and a softmax that
 *  picks one key, where the smallest weight change flips the pick. The
 *  checkpoint's k_norm scales are one constant per layer, a few times this:
 *  0.120–0.132 for the text sliding layers (d = 256, about 2/√d),
 *  0.061–0.065 for the full layers (d = 512, 1.4/√d) and 0.48–0.51 in the
 *  vision tower (d = 64, 4/√d). 1/√d is a softer stand-in that keeps the
 *  scores of order 1. */
const K_NORM_SCALE = (headDim: number) => 1 / Math.sqrt(headDim);

/** One text for the replay path, two for the packed eager path. */
const TOKENS = [2, 7, 1, 8, 28, 18, 28, 45, 9, 4, 52, 3];
const BATCH = [
  [3, 14, 15, 9, 26],
  [5, 35, 8, 9, 7, 9, 32, 38, 46],
];
/** 96×48 pixels at a budget of 2 tokens: 6×3 patches, two pooled tokens. */
const IMAGE = { width: 96, height: 48, tokens: 2 };
/** One second of audio, 25 tokens. */
const AUDIO_SAMPLES = 16_000;
/** Clipped linears clamp their input and output to these bounds. */
const CLIP_BOUNDS = [-4, 4, -4, 4];

type Tensors = Record<string, { data: Float32Array; shape: number[] }>;
/** A torch-layout Linear weight, [out, in]. */
type LinearWeight = { data: Float32Array; shape: [number, number] };
/** A checkpoint in HF names, split into the Linear weights the quantized
 *  file quantizes and every other tensor. */
type Checkpoint = { linear: Record<string, LinearWeight>; other: Tensors };

function builder() {
  let seed = 1;
  const other: Tensors = {};
  const linear: Record<string, LinearWeight> = {};
  const random = (n: number, center = 0) =>
    randomF32(n, seed++).map((v) => center + WEIGHT_SPREAD * v);
  return {
    other,
    linear,
    random,
    dense: (name: string, out: number, inF: number) => {
      linear[`${name}.weight`] = { data: random(out * inF), shape: [out, inF] };
    },
    /** A norm's scale near 1, or one of `n` values of any other tensor. */
    vector: (name: string, n: number, center = 0) => {
      other[name] = { data: random(n, center), shape: [n] };
    },
  };
}

function textCheckpoint(): Checkpoint {
  const { hidden, ffn, pleDim, vocab, heads, embeddingDim } = TEXT;
  const b = builder();
  const S = 'language_model.';
  b.other[`${S}embed_tokens.weight`] = { data: b.random(vocab * hidden), shape: [vocab, hidden] };
  TEXT.layerTypes.forEach((type, i) => {
    const L = `${S}layers.${i}`;
    const { headDim: d, kvHeads } = TEXT.attention[type];
    b.dense(`${L}.self_attn.q_proj`, heads * d, hidden);
    b.dense(`${L}.self_attn.k_proj`, kvHeads * d, hidden);
    b.dense(`${L}.self_attn.v_proj`, kvHeads * d, hidden);
    b.dense(`${L}.self_attn.o_proj`, hidden, heads * d);
    b.vector(`${L}.self_attn.q_norm.weight`, d, 1);
    b.vector(`${L}.self_attn.k_norm.weight`, d, K_NORM_SCALE(d));
    b.dense(`${L}.mlp.gate_proj`, ffn, hidden);
    b.dense(`${L}.mlp.up_proj`, ffn, hidden);
    b.dense(`${L}.mlp.down_proj`, hidden, ffn);
    b.dense(`${L}.ple_block.per_layer_input_gate`, pleDim, hidden);
    b.dense(`${L}.ple_block.per_layer_projection`, hidden, pleDim);
    for (const n of [
      'input_layernorm',
      'post_attention_layernorm',
      'pre_feedforward_layernorm',
      'post_feedforward_layernorm',
      'ple_block.post_per_layer_input_norm',
    ]) {
      b.vector(`${L}.${n}.weight`, hidden, 1);
    }
    b.other[`${L}.layer_scalar`] = { data: Float32Array.of(1), shape: [1] };
  });
  b.dense(`${S}ple.per_layer_model_projection`, TEXT.layers * pleDim, hidden);
  b.vector(`${S}ple.per_layer_projection_norm.weight`, pleDim, 1);
  b.vector(`${S}norm.weight`, hidden, 1);
  b.dense(`${S}embedding_projection`, embeddingDim, hidden);
  return b;
}

function visionCheckpoint(): Checkpoint {
  const { hidden, ffn, heads, headDim: d, patch, positions } = VISION;
  const b = builder();
  const P = 'vision_tower.patch_embedder';
  b.dense(`${P}.input_proj`, hidden, 3 * patch * patch);
  b.other[`${P}.position_embedding_table`] = {
    data: b.random(2 * positions * hidden),
    shape: [2, positions, hidden],
  };
  const L = 'vision_tower.encoder.layers.0';
  for (const p of ['q', 'k', 'v', 'o'])
    b.dense(`${L}.self_attn.${p}_proj.linear`, heads * d, hidden);
  b.vector(`${L}.self_attn.q_norm.weight`, d, 1);
  b.vector(`${L}.self_attn.k_norm.weight`, d, K_NORM_SCALE(d));
  b.dense(`${L}.mlp.gate_proj.linear`, ffn, hidden);
  b.dense(`${L}.mlp.up_proj.linear`, ffn, hidden);
  b.dense(`${L}.mlp.down_proj.linear`, hidden, ffn);
  for (const n of [
    'input_layernorm',
    'post_attention_layernorm',
    'pre_feedforward_layernorm',
    'post_feedforward_layernorm',
  ]) {
    b.vector(`${L}.${n}.weight`, hidden, 1);
  }
  b.dense('embed_vision.embedding_projection', EMBEDDINGGEMMA2_TEXT.hidden, hidden);
  return b;
}

function audioCheckpoint(): Checkpoint {
  const { hidden, ffn, headDim, mels, convKernel, outputDim } = AUDIO;
  const [c0, c1] = AUDIO.convChannels;
  const b = builder();
  const clipped = (name: string, out: number, inF: number) => {
    b.dense(`${name}.linear`, out, inF);
    ['input_min', 'input_max', 'output_min', 'output_max'].forEach((bound, i) => {
      b.other[`${name}.${bound}`] = { data: Float32Array.of(CLIP_BOUNDS[i]!), shape: [1] };
    });
  };
  const sub = 'audio_tower.subsample_conv_projection';
  b.other[`${sub}.layer0.conv.weight`] = { data: b.random(c0 * 9), shape: [c0, 1, 3, 3] };
  b.vector(`${sub}.layer0.norm.weight`, c0, 1);
  b.other[`${sub}.layer1.conv.weight`] = { data: b.random(c1 * c0 * 9), shape: [c1, c0, 3, 3] };
  b.vector(`${sub}.layer1.norm.weight`, c1, 1);
  b.dense(`${sub}.input_proj_linear`, hidden, (mels / 4) * c1);
  const L = 'audio_tower.layers.0';
  for (const ff of ['feed_forward1', 'feed_forward2']) {
    b.vector(`${L}.${ff}.pre_layer_norm.weight`, hidden, 1);
    clipped(`${L}.${ff}.ffw_layer_1`, ffn, hidden);
    clipped(`${L}.${ff}.ffw_layer_2`, hidden, ffn);
    b.vector(`${L}.${ff}.post_layer_norm.weight`, hidden, 1);
  }
  const A = `${L}.self_attn`;
  for (const p of ['q_proj', 'k_proj', 'v_proj', 'post']) clipped(`${A}.${p}`, hidden, hidden);
  // Read on the CPU into the distance bias rows, so it stays float.
  b.other[`${A}.relative_k_proj.weight`] = {
    data: b.random(hidden * hidden),
    shape: [hidden, hidden],
  };
  b.other[`${A}.per_dim_scale`] = { data: new Float32Array(headDim), shape: [headDim] };
  const C = `${L}.lconv1d`;
  b.vector(`${C}.pre_layer_norm.weight`, hidden, 1);
  clipped(`${C}.linear_start`, 2 * hidden, hidden);
  b.other[`${C}.depthwise_conv1d.weight`] = {
    data: b.random(hidden * convKernel),
    shape: [hidden, 1, convKernel],
  };
  b.vector(`${C}.conv_norm.weight`, hidden, 1);
  clipped(`${C}.linear_end`, hidden, hidden);
  for (const n of ['norm_pre_attn', 'norm_post_attn', 'norm_out'])
    b.vector(`${L}.${n}.weight`, hidden, 1);
  b.dense('audio_tower.output_proj', outputDim, hidden);
  b.vector('audio_tower.output_proj.bias', outputDim);
  b.dense('embed_audio.embedding_projection', EMBEDDINGGEMMA2_TEXT.hidden, outputDim);
  return b;
}

/** A model of `make` loaded from `file`, disposed when the test ends. */
async function load<M extends nn.Module<never[], unknown>>(
  make: () => M,
  file: Uint8Array,
  dtype: 'f32' | 'f16',
): Promise<M> {
  const model = make();
  onTestFinished(() => model.dispose());
  if (dtype === 'f16') model.half(root);
  await model.loadStateDict(await fromSafetensors(bufferSource(file), { chunkBytes: 0 }), {
    root,
  });
  return model;
}

/** A tower: how to build its checkpoint, load it and run it, and which
 *  Linears the quantized file must have put on the quantized matmul. */
interface Tower {
  checkpoint: () => Checkpoint;
  run: (
    file: Uint8Array,
    dtype: 'f32' | 'f16',
  ) => Promise<{ scaleNames: string[]; outputs: Float32Array[] }>;
  quantized: string[];
}

const scaleNamesOf = (model: nn.Module<never[], unknown>) =>
  [...model.namedParameters()]
    .map(([name]) => name)
    .filter((name) => name.endsWith('.weightScale'))
    .sort();

const textLayer = (i: number) =>
  [
    'qkv',
    'o_proj',
    'gate_proj',
    'up_proj',
    'down_proj',
    'per_layer_input_gate',
    'per_layer_projection',
  ].map((m) => `layers.${i}.${m}`);

const TOWERS: Record<string, Tower> = {
  text: {
    checkpoint: textCheckpoint,
    async run(file, dtype) {
      const model = await load(() => new EmbeddingGemma2Model(TEXT), file, dtype);
      const core = createEmbeddingGemma2EmbedCore(model);
      onTestFinished(() => core.dispose());
      const outputs = [
        await core.embedPooled({ ids: TOKENS, media: [] }),
        await core.embedPooledBatch(BATCH.map((ids) => ({ ids, media: [] }))),
      ];
      return { scaleNames: scaleNamesOf(model), outputs };
    },
    quantized: [
      ...textLayer(0),
      ...textLayer(1),
      'ple.per_layer_model_projection',
      'embedding_projection',
    ],
  },
  vision: {
    checkpoint: visionCheckpoint,
    async run(file, dtype) {
      const model = await load(() => new EmbeddingGemma2VisionModel(VISION), file, dtype);
      const ex = gpuExecutor(defaultRoot());
      onTestFinished(() => ex.dispose());
      const { width, height, tokens } = IMAGE;
      const pixels = randomF32(width * height * 4, 7).map((v) => Math.floor(128 + 127 * v));
      const image = imageBuffer(Uint8Array.from(pixels), width, height);
      return {
        scaleNames: scaleNamesOf(model),
        outputs: [await encodeImage(model, image, tokens, ex)],
      };
    },
    quantized: [
      'input_proj',
      ...['qkv', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'].map((m) => `layers.0.${m}`),
      'embedding_projection',
    ],
  },
  audio: {
    checkpoint: audioCheckpoint,
    async run(file, dtype) {
      const model = await load(() => new EmbeddingGemma2AudioModel(AUDIO), file, dtype);
      const ex = gpuExecutor(defaultRoot());
      onTestFinished(() => ex.dispose());
      const samples = randomF32(AUDIO_SAMPLES, 99).map((v) => 0.3 * v);
      return { scaleNames: scaleNamesOf(model), outputs: [await encodeAudio(model, samples, ex)] };
    },
    quantized: [
      'input_proj',
      ...[
        'feed_forward1.ffw_layer_1',
        'feed_forward1.ffw_layer_2',
        'qkv',
        'post',
        'lconv.linear_start',
        'lconv.linear_end',
        'feed_forward2.ffw_layer_1',
        'feed_forward2.ffw_layer_2',
      ].map((m) => `layers.0.${m}`),
      'output_proj',
      'embedding_projection',
    ],
  },
};

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
 *  Measured worst case over the three towers, the same on three runs: 7.2e-6
 *  in f32, where only the summation order differs, and 4.9e-3 in f16 on
 *  outputs up to 4.7 (the audio tower), where the reference rounds each int
 *  times scale to f16 before a tower's layers stack up the difference. f16
 *  allows four f16 steps at that size. */
const KERNEL_TOLERANCE = {
  f32: { atol: 2e-5, rtol: 0 },
  f16: { atol: 2 ** -6, rtol: 0 },
};

/** Least cosine similarity to the model with the original float weights,
 *  set from the measured worst case across towers and dtypes, both the text
 *  model's packed batch: int8 0.99994 (1 - cos = 6.4e-5), int4 0.978
 *  (2.2e-2). Each bound allows about 2.5 to 3 times the measured 1 - cos. */
const LEAST_COSINE = { 8: 0.9998, 4: 0.95 };

describe('EmbeddingGemma 2 from a quantized checkpoint', () => {
  it.for(
    Object.keys(TOWERS).flatMap((tower) =>
      (['f32', 'f16'] as const).flatMap((dtype) =>
        ([8, 4] as const).map((bits) => ({ tower, dtype, bits, groupSize: 32 })),
      ),
    ),
  )(
    '$tower $dtype int$bits, groupSize $groupSize',
    async ({ tower, dtype, bits, groupSize }, ctx) => {
      if (dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
      const { checkpoint, run, quantized: expectedQuantized } = TOWERS[tower]!;
      const { linear, other } = checkpoint();
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
          {
            data: Float32Array.from(ints, (q, i) => q * scales[Math.floor(i / groupSize)]!),
            shape,
          },
        ]),
      );

      const got = await run(quantFile(bits, groupSize, quantized, other), dtype);
      // Every Linear loaded quantized, so no layer took the float matmul.
      expect(got.scaleNames).toEqual(expectedQuantized.map((m) => `${m}.weightScale`).sort());

      const reference = await run(safetensorsFile({ ...other, ...dequantized }), dtype);
      const float = await run(safetensorsFile({ ...other, ...linear }), dtype);
      got.outputs.forEach((out, i) => {
        expectClose(out, reference.outputs[i]!, KERNEL_TOLERANCE[dtype]);
        expect(cosine(out, float.outputs[i]!)).toBeGreaterThan(LEAST_COSINE[bits]);
      });
    },
  );
});
