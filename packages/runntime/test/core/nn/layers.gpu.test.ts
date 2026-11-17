import { describe, expect, it, onTestFinished, vi } from 'vitest';
import { toArray } from '../../../src/core/gpu/eval.ts';
import type { SlotAct } from '../../../src/core/graph/ops.ts';
import { matrix } from '../../../src/core/graph/value.ts';
import {
  Conv1d,
  ConvTranspose1d,
  Embedding,
  FusedLinear,
  Linear,
} from '../../../src/core/nn/layers.ts';
import { Module } from '../../../src/core/nn/module.ts';
import {
  bufferSource,
  fromSafetensors,
  memoryStateDict,
  type LazyStateDict,
} from '../../../src/core/weights/safetensors.ts';
import { expectClose, gpu, randomF32, requireFeature, roundF16, upload } from '../../gpu.ts';
import { ACTIVATION_REF } from '../kernels/matmul/checkMatmul.ts';
import {
  mixedQuantFile,
  quantFile,
  safetensorsFile,
  type BitsQuantEntry,
} from '../weights/checkpoint.ts';

const { root, ex } = await gpu();

interface Case {
  m: number;
  inF: number;
  outF: number;
  groupSize: number;
  bias: boolean;
  addend?: boolean;
  activation?: SlotAct;
  /** Takes the tiled kernel, which stages the dequantized weight in f16
   *  workgroup memory for an f16 input. */
  tiled?: boolean;
  /** The bits at which the quantized matmul cannot take the weight, so it
   *  loads as its dequantized floats in the layer's dtype. */
  float?: readonly (4 | 8)[];
}

/** Watches console.warn for the rest of the test. Returns a reader of the
 *  loadStateDict warnings so far; TypeGPU warns there too. */
function loadWarnings(): () => unknown[] {
  const warn = vi.spyOn(console, 'warn');
  onTestFinished(() => warn.mockRestore());
  return () =>
    warn.mock.calls
      .map(([message]) => message as unknown)
      .filter((message) => typeof message === 'string' && message.startsWith('loadStateDict:'));
}

const LOADED_AS_FLOAT = /^loadStateDict: quantized weights loaded as float.*: weight$/;

/** A Linear loaded from a quantized file matches a CPU loop of
 *  act(x·W + bias) + addend in f64, with W dequantized from the same ints as
 *  int · f16 scale. */
async function checkQuantizedLinear(c: Case, bits: 4 | 8, dtype: 'f32' | 'f16') {
  const { m, inF, outF, groupSize } = c;
  const half = 2 ** (bits - 1);
  // torch [out, in] ints over the full signed range, positive scales
  const ints = Array.from(randomF32(outF * inF, 1), (v) =>
    Math.min(half - 1, Math.floor(v * half)),
  );
  const scales = roundF16(
    randomF32((outF * inF) / groupSize, 2).map((v) => (Math.abs(v) + 0.1) / half),
  );
  // Rounded to f16 for both dtypes, so the f16 upload stores them exactly.
  const bias = roundF16(randomF32(outF, 3));
  const xs = roundF16(randomF32(m * inF, 4));
  const addend = roundF16(randomF32(m * outF, 5));

  const layer = new Linear(inF, outF, { bias: c.bias, activation: c.activation });
  if (dtype === 'f16') layer.half(root);
  onTestFinished(() => layer.dispose());
  const file = quantFile(bits, groupSize, {
    weight: { ints, scales: [...scales], shape: [outF, inF] },
  });
  const sd = await fromSafetensors(bufferSource(file), { chunkBytes: 0 });
  if (c.bias) sd.tensors.set('bias', memoryStateDict({ bias }).tensors.get('bias')!);
  const warned = loadWarnings();
  await layer.loadStateDict(sd, { root });
  const float = c.float?.includes(bits) ?? false;
  expect(layer.weight.shape.dtype).toBe(float ? dtype : 'quantW');
  expect(warned()).toEqual(float ? [expect.stringMatching(LOADED_AS_FLOAT)] : []);

  const x = upload(root, xs, { ...matrix(m, inF), dtype });
  const call = c.addend ? { addend: upload(root, addend, { ...matrix(m, outF), dtype }) } : {};
  const got = await toArray(layer.forward(x, call), ex);

  const dequant = Float32Array.from(ints, (q, i) => q * scales[Math.floor(i / groupSize)]!);
  const weight = dtype === 'f16' && (c.tiled || float) ? roundF16(dequant) : dequant;
  const want = new Float64Array(m * outF);
  for (let i = 0; i < m; i++) {
    for (let o = 0; o < outF; o++) {
      let sum = 0;
      for (let p = 0; p < inF; p++) sum += xs[i * inF + p]! * weight[o * inF + p]!;
      if (c.bias) sum += bias[o]!;
      if (c.activation) sum = ACTIVATION_REF[c.activation](sum);
      if (c.addend) sum += addend[i * outF + o]!;
      want[i * outF + o] = sum;
    }
  }
  // The kernel sums in f32, so its error grows with inF; f16 also rounds the
  // output to half precision, 2^-11 relative.
  const tol = dtype === 'f16' ? { atol: 1e-5 * inF, rtol: 1e-3 } : { atol: 1e-6 * inF, rtol: 1e-5 };
  expectClose(got, want, tol);
  if (!float) return;

  // The fallback is the float layer: a Linear loaded from the dequantized
  // weight takes the same route and gives the same bits.
  const reference = new Linear(inF, outF, { bias: c.bias, activation: c.activation });
  if (dtype === 'f16') reference.half(root);
  onTestFinished(() => reference.dispose());
  await reference.loadStateDict(
    memoryStateDict({
      weight: { data: dequant, shape: [outF, inF] },
      ...(c.bias ? { bias } : {}),
    }),
    { root },
  );
  expect(reference.weight.shape.dtype).toBe(dtype);
  expectClose(await toArray(reference.forward(x, call), ex), got, { atol: 0, rtol: 0 });
}

describe('Linear with a quantized weight', () => {
  // A quantized Linear takes matmulQuantW below 48·1024 output values (m ·
  // outF) and the tiled kernel from there, for either dtype and any epilogue.
  const CASES: (Case & { name: string; dtypes: ('f32' | 'f16')[] })[] = [
    // matmulQuantW: m · outF < 48·1024
    { name: '1 row', m: 1, inF: 64, outF: 48, groupSize: 16, bias: false, dtypes: ['f32', 'f16'] },
    { name: '8 rows', m: 8, inF: 64, outF: 48, groupSize: 16, bias: false, dtypes: ['f32'] },
    { name: 'bias, 7 rows', m: 7, inF: 64, outF: 48, groupSize: 16, bias: true, dtypes: ['f32'] },
    {
      name: 'activation, 3 rows',
      m: 3,
      inF: 64,
      outF: 48,
      groupSize: 16,
      bias: false,
      activation: 'gelu',
      dtypes: ['f32', 'f16'],
    },
    {
      name: 'addend, 2 rows',
      m: 2,
      inF: 64,
      outF: 48,
      groupSize: 16,
      bias: false,
      addend: true,
      dtypes: ['f32', 'f16'],
    },
    {
      name: 'bias, 5 rows',
      m: 5,
      inF: 64,
      outF: 48,
      groupSize: 16,
      bias: true,
      dtypes: ['f16'],
    },
    {
      name: 'bias, gelu and addend, 37 rows',
      m: 37,
      inF: 256,
      outF: 136,
      groupSize: 32,
      bias: true,
      addend: true,
      activation: 'gelu',
      dtypes: ['f32', 'f16'],
    },
    {
      name: 'one group per row, 767 rows',
      m: 767,
      inF: 128,
      outF: 64,
      groupSize: 128,
      bias: false,
      dtypes: ['f32', 'f16'],
    },
    // tiled: m · outF >= 48·1024
    {
      name: 'one group per row, 768 rows',
      m: 768,
      inF: 128,
      outF: 64,
      groupSize: 128,
      bias: false,
      tiled: true,
      dtypes: ['f32', 'f16'],
    },
    {
      name: 'partial tiles with bias, gelu and addend, 400 rows',
      m: 400,
      inF: 256,
      outF: 136,
      groupSize: 32,
      bias: true,
      addend: true,
      activation: 'gelu',
      tiled: true,
      dtypes: ['f32', 'f16'],
    },
  ];
  it.for(
    CASES.flatMap(({ dtypes, ...c }) =>
      dtypes.flatMap((dtype) => ([4, 8] as const).map((bits) => ({ ...c, dtype, bits }))),
    ),
  )('$dtype int$bits $name: [$m, $inF] → $outF, groupSize $groupSize', async (c, ctx) => {
    if (c.dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    await checkQuantizedLinear(c, c.bits, c.dtype);
  });
});

describe('Linear with a quantized weight at the limits of the quantized matmul', () => {
  // It takes in features a multiple of 8 and of the group size, out features
  // a multiple of 4 (int8) or 8 (int4), and a group size a multiple of 4.
  // Anything else runs the float matmul on the dequantized weight.
  const base = { m: 3, bias: true, dtypes: ['f32', 'f16'] } as const;
  const CASES: (Case & { name: string; dtypes: readonly ('f32' | 'f16')[] })[] = [
    // quantized: in 8 % 8 == 0, out 8 % 8 == 0, groupSize 4 % 4 == 0
    { ...base, name: 'the smallest that fits', inF: 8, outF: 8, groupSize: 4 },
    // float: in 12 % 8 != 0
    { ...base, name: 'in off the 8-row tile', inF: 12, outF: 8, groupSize: 4, float: [4, 8] },
    // float: in 4 % 8 != 0
    { ...base, name: 'in under one tile', inF: 4, outF: 8, groupSize: 4, float: [4, 8] },
    // int8 quantized: out 4 % 4 == 0. int4 float: out 4 % 8 != 0
    { ...base, name: 'out of one int8 word', inF: 8, outF: 4, groupSize: 4, float: [4] },
    // int8 quantized: out 12 % 4 == 0. int4 float: out 12 % 8 != 0
    { ...base, name: 'out off the int4 word', inF: 16, outF: 12, groupSize: 8, float: [4] },
    // float: out 6 % 4 != 0
    { ...base, name: 'out off the int8 word', inF: 16, outF: 6, groupSize: 8, float: [4, 8] },
    // float: out 1
    { ...base, name: 'one output', inF: 16, outF: 1, groupSize: 8, float: [4, 8] },
    // quantized: groupSize 12 % 4 == 0 and 24 % 12 == 0
    { ...base, name: 'groups of 12', inF: 24, outF: 8, groupSize: 12 },
    // float: groupSize 6 % 4 != 0
    { ...base, name: 'groups of 6', inF: 24, outF: 8, groupSize: 6, float: [4, 8] },
    // float: groupSize 2 % 4 != 0
    { ...base, name: 'groups of 2', inF: 16, outF: 8, groupSize: 2, float: [4, 8] },
    // float: groupSize 1
    { ...base, name: 'a scale per value', inF: 16, outF: 8, groupSize: 1, float: [4, 8] },
    // float, with the epilogue the float matmul folds
    {
      ...base,
      name: 'in off the tile with gelu and addend',
      m: 37,
      inF: 100,
      outF: 30,
      groupSize: 20,
      addend: true,
      activation: 'gelu',
      float: [4, 8],
    },
  ];
  it.for(
    CASES.flatMap(({ dtypes, ...c }) =>
      dtypes.flatMap((dtype) => ([4, 8] as const).map((bits) => ({ ...c, dtype, bits }))),
    ),
  )('$dtype int$bits $name: [$m, $inF] → $outF, groupSize $groupSize', async (c, ctx) => {
    if (c.dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    await checkQuantizedLinear(c, c.bits, c.dtype);
  });
});

/** A part of a FusedLinear test: its out features, how its weight sits in
 *  the file, and whether it has a bias. */
type FusedPart = { out: number; bits?: 4 | 8; groupSize?: number; bias: boolean };

/** Loads `parts` through a FusedLinear and through one Linear over their
 *  hand-stacked rows. The FusedLinear output matches a CPU loop of each
 *  part's x·W + bias side by side in f64, and the Linear's output bit for
 *  bit, since both run the same matmul on the same weight. */
async function checkFusedLinear(
  c: { m: number; inF: number; parts: readonly FusedPart[]; quantized: boolean },
  dtype: 'f32' | 'f16',
) {
  const { m, inF, parts } = c;
  const outF = parts.reduce((sum, part) => sum + part.out, 0);
  const xs = roundF16(randomF32(m * inF, 9));

  // Each part's torch [out, in] weight: floats rounded to f16, or ints over
  // the signed range with positive f16 scales.
  const floats: Record<string, { data: Float32Array; shape: number[] }> = {};
  const quant: Record<string, BitsQuantEntry> = {};
  const weights: Float32Array[] = [];
  const biases: Float32Array[] = [];
  for (const [p, part] of parts.entries()) {
    const { out, bits, groupSize = 1 } = part;
    if (bits) {
      const half = 2 ** (bits - 1);
      const ints = Array.from(randomF32(out * inF, 10 + p), (v) =>
        Math.min(half - 1, Math.floor(v * half)),
      );
      const scales = roundF16(
        randomF32((out * inF) / groupSize, 20 + p).map((v) => (Math.abs(v) + 0.1) / half),
      );
      quant[`part${p}.weight`] = { ints, scales: [...scales], shape: [out, inF], bits };
      weights.push(Float32Array.from(ints, (q, i) => q * scales[Math.floor(i / groupSize)]!));
    } else {
      const data = roundF16(randomF32(out * inF, 10 + p));
      floats[`part${p}.weight`] = { data, shape: [out, inF] };
      weights.push(data);
    }
    const bias = part.bias ? roundF16(randomF32(out, 30 + p)) : new Float32Array(out);
    if (part.bias) floats[`part${p}.bias`] = { data: bias, shape: [out] };
    biases.push(bias);
  }
  const hasBias = parts.some((part) => part.bias);

  class Fused extends Module {
    readonly fused: FusedLinear;
    constructor() {
      super();
      const linears = parts.map((part) => new Linear(inF, part.out, { bias: part.bias }));
      linears.forEach((linear, p) => {
        (this as Record<string, unknown>)[`part${p}`] = linear;
      });
      this.fused = new FusedLinear(linears);
    }
  }
  const model = new Fused();
  if (dtype === 'f16') model.half(root);
  onTestFinished(() => model.dispose());
  const warned = loadWarnings();
  await model.loadStateDict(
    await fromSafetensors(bufferSource(mixedQuantFile(quant, floats)), { chunkBytes: 0 }),
    { root, strict: true },
  );
  const differ = parts.map(({ bits, groupSize }, p) => {
    return `part${p}.weight is ${bits ? `int${bits} in groups of ${groupSize}` : 'float'}`;
  });
  // A mix of float and quantized parts, or of bits or group sizes.
  expect(warned()).toEqual(
    c.quantized || Object.keys(quant).length === 0
      ? []
      : [
          'loadStateDict: fused layers loaded as float, their parts are not quantized alike: ' +
            `fused: ${differ.join(', ')}`,
        ],
  );

  const x = upload(root, xs, { ...matrix(m, inF), dtype });
  const got = await toArray(model.fused.forward(x), ex);

  // A float weight in an f16 model is stored rounded to f16.
  const stored = weights.map((w) => (dtype === 'f16' && !c.quantized ? roundF16(w) : w));
  const want = new Float64Array(m * outF);
  for (let i = 0; i < m; i++) {
    let column = 0;
    for (const [p, part] of parts.entries()) {
      for (let o = 0; o < part.out; o++) {
        let sum = biases[p]![o]!;
        for (let k = 0; k < inF; k++) sum += xs[i * inF + k]! * stored[p]![o * inF + k]!;
        want[i * outF + column++] = sum;
      }
    }
  }
  const tol = dtype === 'f16' ? { atol: 1e-5 * inF, rtol: 1e-3 } : { atol: 1e-6 * inF, rtol: 1e-5 };
  expectClose(got, want, tol);

  // One Linear over the stacked rows: the same file layout a hand-written
  // fuse produced.
  const single = new Linear(inF, outF, { bias: hasBias });
  if (dtype === 'f16') single.half(root);
  onTestFinished(() => single.dispose());
  const stackedBias = { data: Float32Array.from(biases.flatMap((b) => [...b])), shape: [outF] };
  const first = Object.values(quant)[0];
  const file = c.quantized
    ? quantFile(
        first!.bits,
        parts[0]!.groupSize!,
        {
          weight: {
            ints: Object.values(quant).flatMap((entry) => entry.ints),
            scales: Object.values(quant).flatMap((entry) => entry.scales),
            shape: [outF, inF],
          },
        },
        hasBias ? { bias: stackedBias } : {},
      )
    : safetensorsFile({
        weight: { data: Float32Array.from(weights.flatMap((w) => [...w])), shape: [outF, inF] },
        ...(hasBias ? { bias: stackedBias } : {}),
      });
  await single.loadStateDict(await fromSafetensors(bufferSource(file), { chunkBytes: 0 }), {
    root,
  });
  expect(single.weight.shape.dtype).toBe(c.quantized ? 'quantW' : dtype);
  expect([...got]).toEqual([...(await toArray(single.forward(x), ex))]);
}

describe('FusedLinear', () => {
  const float = (out: number, bias = true): FusedPart => ({ out, bias });
  const int8 = (out: number, groupSize = 16, bias = true): FusedPart => ({
    out,
    bits: 8,
    groupSize,
    bias,
  });
  const int4 = (out: number): FusedPart => ({ out, bits: 4, groupSize: 16, bias: true });
  const CASES: [name: string, m: number, inF: number, parts: FusedPart[], quantized: boolean][] = [
    ['1 float part', 3, 24, [float(10)], false],
    ['2 float parts', 5, 24, [float(10), float(7)], false],
    [
      '3 float parts, the middle one without a bias',
      7,
      64,
      [float(48), float(16, false), float(16)],
      false,
    ],
    ['5 float parts without a bias', 2, 16, [1, 4, 2, 7, 3].map((out) => float(out, false)), false],
    // quantized: in 64 % 8 == 0, stacked out 80 % 4 == 0, groups of 16
    ['3 int8 parts', 7, 64, [int8(48), int8(16), int8(16)], true],
    // quantized: stacked out 12 % 4 == 0, though no part is a multiple of 4
    [
      '5 int8 parts of sizes off the int8 word',
      4,
      32,
      [1, 2, 3, 1, 5].map((out) => int8(out, 8, false)),
      true,
    ],
    // quantized: stacked out 8 % 8 == 0, each part 4 % 8 != 0
    ['2 int4 parts', 3, 32, [int4(4), int4(4)], true],
    // float: a float part cannot stack with ints
    ['a float part between two int8 parts', 3, 32, [int8(8), float(5), int8(4)], false],
    // float: int8 and int4 cannot share packed words
    ['int8 next to int4', 3, 32, [int8(8), int4(8)], false],
    // float: two group sizes cannot share a scale grid
    ['int8 in groups of 16 next to groups of 8', 3, 32, [int8(8, 16), int8(8, 8)], false],
  ];
  it.for(
    CASES.flatMap(([name, m, inF, parts, quantized]) =>
      (['f32', 'f16'] as const).map((dtype) => ({ name, m, inF, parts, quantized, dtype })),
    ),
  )('$dtype $name: [$m, $inF]', async (c, ctx) => {
    if (c.dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    await checkFusedLinear(c, c.dtype);
  });
});

describe('Embedding with a quantized table', () => {
  // No kernel reads a quantized lookup table, so it loads as its dequantized
  // floats, without a warning: on the GPU in the model's dtype, on the CPU as
  // f32.
  const [vocab, dim, groupSize] = [10, 6, 3];
  const ids = [9, 0, 4, 4];
  it.for(
    (['gpu', 'cpu'] as const).flatMap((device) =>
      ([4, 8] as const).flatMap((bits) =>
        (['f32', 'f16'] as const).map((dtype) => ({ device, bits, dtype })),
      ),
    ),
  )('$device table, int$bits, $dtype', async ({ device, bits, dtype }, ctx) => {
    if (dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    const half = 2 ** (bits - 1);
    const ints = Array.from(randomF32(vocab * dim, 1), (v) =>
      Math.min(half - 1, Math.floor(v * half)),
    );
    const scales = roundF16(
      randomF32((vocab * dim) / groupSize, 2).map((v) => (Math.abs(v) + 0.1) / half),
    );
    const table = new Embedding(vocab, dim, { device });
    if (dtype === 'f16') table.half(root);
    onTestFinished(() => table.dispose());
    const file = quantFile(bits, groupSize, {
      weight: { ints, scales: [...scales], shape: [vocab, dim] },
    });
    const warned = loadWarnings();
    await table.loadStateDict(await fromSafetensors(bufferSource(file), { chunkBytes: 0 }), {
      root,
    });
    expect(warned()).toEqual([]);
    const got = await toArray(table.forward(ids), ex);

    const dequant = Float32Array.from(ints, (q, i) => q * scales[Math.floor(i / groupSize)]!);
    // A GPU table is stored in the model's dtype; a CPU one stays f32 and its
    // gathered rows round on upload.
    const stored = dtype === 'f16' ? roundF16(dequant) : dequant;
    const want = ids.flatMap((id) => [...stored.subarray(id * dim, (id + 1) * dim)]);
    expectClose(got, want, { atol: 0, rtol: 0 });
  });
});

interface ConvCase {
  name: string;
  kind: 'conv' | 'transpose';
  tIn: number;
  cIn: number;
  cOut: number;
  kernel: number;
  stride: number;
  padding: number;
  /** Values of a torch row per scale. */
  groupSize: number;
  bias: boolean;
  activation?: 'gelu';
  /** The quantized matmul cannot run this conv at any bits, so the weight
   *  loads as its dequantized floats. */
  float?: boolean;
}

/** A conv loaded from a quantized file of its torch weight, [out, in·kernel]
 *  or a transposed conv's [in, out·kernel], matches the same conv loaded
 *  with the dequantized floats, int · f16 scale. */
async function checkQuantizedConv(c: ConvCase, bits: 4 | 8, dtype: 'f32' | 'f16') {
  const { tIn, cIn, cOut, kernel, stride, padding, groupSize } = c;
  const half = 2 ** (bits - 1);
  const [rows, cols] = c.kind === 'conv' ? [cOut, cIn * kernel] : [cIn, cOut * kernel];
  const ints = Array.from(randomF32(rows * cols, 1), (v) =>
    Math.min(half - 1, Math.floor(v * half)),
  );
  const scales = roundF16(
    randomF32((rows * cols) / groupSize, 2).map((v) => (Math.abs(v) + 0.1) / half),
  );
  const bias = roundF16(randomF32(cOut, 3));
  const xs = roundF16(randomF32(tIn * cIn, 4));
  const dequant = Float32Array.from(ints, (q, i) => q * scales[Math.floor(i / groupSize)]!);
  const torchWeight = {
    data: dequant,
    shape: c.kind === 'conv' ? [cOut, cIn, kernel] : [cIn, cOut, kernel],
  };
  const make = () =>
    c.kind === 'conv'
      ? new Conv1d(cIn, cOut, kernel, { stride, padding, bias: c.bias, activation: c.activation })
      : new ConvTranspose1d(cIn, cOut, kernel, { stride, padding, bias: c.bias });
  const run = async (layer: Conv1d | ConvTranspose1d, sd: LazyStateDict, quantized = false) => {
    if (dtype === 'f16') layer.half(root);
    onTestFinished(() => layer.dispose());
    await layer.loadStateDict(sd, { root });
    if (quantized) expect(layer.weight.shape.dtype).toBe(c.float ? dtype : 'quantW');
    return toArray(layer.forward(upload(root, xs, { ...matrix(tIn, cIn), dtype })), ex);
  };
  const file = quantFile(bits, groupSize, {
    weight: { ints, scales: [...scales], shape: [rows, cols] },
  });
  const quantSd = await fromSafetensors(bufferSource(file), { chunkBytes: 0 });
  if (c.bias) quantSd.tensors.set('bias', memoryStateDict({ bias }).tensors.get('bias')!);
  const warned = loadWarnings();
  const got = await run(make(), quantSd, true);
  expect(warned()).toEqual(c.float ? [expect.stringMatching(LOADED_AS_FLOAT)] : []);
  const want = await run(
    make(),
    memoryStateDict({ weight: torchWeight, ...(c.bias ? { bias } : {}) }),
  );
  // Both sum in f32 in another order; f16 rounds the output to half precision.
  const tol = dtype === 'f16' ? { atol: 2e-3, rtol: 2e-3 } : { atol: 1e-5, rtol: 1e-5 };
  expectClose(got, want, tol);
}

describe('Conv1d and ConvTranspose1d with a quantized weight', () => {
  const CASES: ConvCase[] = [
    {
      name: 'conv, padded, groups that end inside a channel, bias and gelu',
      kind: 'conv',
      tIn: 40,
      cIn: 16,
      cOut: 24,
      kernel: 7,
      stride: 1,
      padding: 3,
      groupSize: 16,
      bias: true,
      activation: 'gelu',
    },
    {
      name: 'conv 1×1, stride 2',
      kind: 'conv',
      tIn: 33,
      cIn: 32,
      cOut: 16,
      kernel: 1,
      stride: 2,
      padding: 0,
      groupSize: 32,
      bias: false,
    },
    // 1 and 3 output channels pad up to the packing width and are cut after.
    {
      name: 'conv to 1 channel, bias',
      kind: 'conv',
      tIn: 30,
      cIn: 32,
      cOut: 1,
      kernel: 7,
      stride: 1,
      padding: 3,
      groupSize: 56,
      bias: true,
    },
    {
      name: 'conv to 3 channels',
      kind: 'conv',
      tIn: 20,
      cIn: 8,
      cOut: 3,
      kernel: 3,
      stride: 1,
      padding: 1,
      groupSize: 6,
      bias: true,
    },
    {
      name: 'transposed, stride 4, bias',
      kind: 'transpose',
      tIn: 12,
      cIn: 16,
      cOut: 8,
      kernel: 8,
      stride: 4,
      padding: 2,
      groupSize: 16,
      bias: true,
    },
    {
      name: 'transposed, stride 2',
      kind: 'transpose',
      tIn: 9,
      cIn: 32,
      cOut: 8,
      kernel: 4,
      stride: 2,
      padding: 1,
      groupSize: 32,
      bias: false,
    },
    // The quantized matmul runs a Conv1d with in a multiple of 4 and kernel·in
    // of 8, and a ConvTranspose1d that also has out a multiple of 4 and
    // stride·out of 4 (int8) or 8 (int4). The rest run as float convs.
    // quantized: in 4 % 4 == 0, kernel·in 8 % 8 == 0
    {
      name: 'conv, the smallest that fits',
      kind: 'conv',
      tIn: 11,
      cIn: 4,
      cOut: 5,
      kernel: 2,
      stride: 1,
      padding: 0,
      groupSize: 8,
      bias: true,
    },
    // float: kernel·in 12 % 8 != 0
    {
      name: 'conv, kernel·in off the 8-row tile',
      kind: 'conv',
      tIn: 11,
      cIn: 4,
      cOut: 5,
      kernel: 3,
      stride: 1,
      padding: 1,
      groupSize: 4,
      bias: true,
      float: true,
    },
    // float: in 6 % 4 != 0, though kernel·in 24 % 8 == 0
    {
      name: 'conv, in off 4',
      kind: 'conv',
      tIn: 11,
      cIn: 6,
      cOut: 8,
      kernel: 4,
      stride: 2,
      padding: 1,
      groupSize: 6,
      bias: false,
      float: true,
    },
    // quantized: out 4 % 4 == 0, stride·out 8 % 8 == 0, taps·in 8 % 8 == 0
    {
      name: 'transposed, the smallest that fits',
      kind: 'transpose',
      tIn: 7,
      cIn: 4,
      cOut: 4,
      kernel: 4,
      stride: 2,
      padding: 1,
      groupSize: 16,
      bias: true,
    },
    // float: out 6 % 4 != 0
    {
      name: 'transposed, out off 4',
      kind: 'transpose',
      tIn: 7,
      cIn: 8,
      cOut: 6,
      kernel: 4,
      stride: 2,
      padding: 1,
      groupSize: 3,
      bias: true,
      float: true,
    },
    // float: taps·in 12 % 8 != 0
    {
      name: 'transposed, taps·in off the 8-row tile',
      kind: 'transpose',
      tIn: 7,
      cIn: 4,
      cOut: 4,
      kernel: 6,
      stride: 2,
      padding: 2,
      groupSize: 8,
      bias: false,
      float: true,
    },
  ];
  it.for(
    (['f32', 'f16'] as const).flatMap((dtype) =>
      CASES.flatMap((c) => ([8, 4] as const).map((bits) => ({ ...c, dtype, bits }))),
    ),
  )('$dtype int$bits $name', async (c, ctx) => {
    if (c.dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    await checkQuantizedConv(c, c.bits, c.dtype);
  });
});
