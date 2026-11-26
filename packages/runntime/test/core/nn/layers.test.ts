import { readFileSync } from 'node:fs';
import { describe, expect, it, onTestFinished, vi } from 'vitest';
import { ACT_CODE } from '../../../src/core/graph/ops/shared.ts';
import {
  materialized,
  pending,
  type GpuBufferRef,
  type ValueMeta,
} from '../../../src/core/graph/value.ts';
import {
  BatchNorm2d,
  Conv1d,
  Conv2d,
  ConvTranspose1d,
  ConvTranspose2d,
  Embedding,
  FusedLinear,
  GroupNorm,
  LayerNorm,
  Linear,
  MultiHeadAttention,
  RMSNorm,
  Snake,
} from '../../../src/core/nn/layers.ts';
import { loadLazyStateDict } from '../../../src/core/nn/loadStateDict.ts';
import { CpuParameter, Module, ModuleList, type AnyModule } from '../../../src/core/nn/module.ts';
import { f16ToF32, f32ArrayToF16Bits } from '../../../src/core/weights/convert.ts';
import {
  bufferSource,
  derive,
  fromSafetensors,
  memoryStateDict,
  type LazyStateDict,
} from '../../../src/core/weights/safetensors.ts';
import { scaleWeight } from '../../../src/core/weights/transforms.ts';
import { chunksFetched, onceInFileOrder, zeroCheckpoint } from '../../zoo/chunkedLoad.ts';
import { hwc4Input, input, meta } from '../graph/values.ts';
import {
  checkpoint,
  contents,
  mixedQuantFile,
  quantFile,
  safetensorsFile,
  type BitsQuantEntry,
  type Entry,
  type QuantEntry,
} from '../weights/checkpoint.ts';

/** Binds every parameter to a stand-in of its declared shape, so forward()
 *  can build its graph without real weights. */
function fakeLoad<M extends AnyModule>(m: M): M {
  for (const [, p] of m.namedParameters()) p.bind(pending(p.shape, 'reshape', []));
  return m;
}

/** name → declared shape, for comparing whole layers at once. */
const shapes = (m: AnyModule) =>
  Object.fromEntries([...m.namedParameters()].map(([n, p]) => [n, p.shape]));

/** A state dict with one F16 tensor named `weight`, read from a real file:
 *  memoryStateDict reads a Uint16Array as BF16. */
const f16Checkpoint = (bits: Uint16Array, shape: number[]): Promise<LazyStateDict> =>
  fromSafetensors(bufferSource(safetensorsFile({ weight: { data: bits, shape } })), {
    chunkBytes: 0,
  });

/** A state dict whose `weight` is a derived tensor holding `data`, as a zoo
 *  hook hands one over: a Uint16Array means f16 halfwords, a Uint32Array
 *  packed words. */
function derivedCheckpoint(data: Uint16Array | Uint32Array, shape: number[]): LazyStateDict {
  const tensor = derive(shape, [], () => Promise.resolve(data), data.byteLength);
  return {
    metadata: {},
    tensors: new Map([['weight', tensor]]),
    totalBytes: () => data.byteLength,
  };
}

/** bf16 bits: the top half of the f32 pattern. */
const bf16 = (values: number[]) =>
  Uint16Array.from(values, (v) => new Uint32Array(new Float32Array([v]).buffer)[0]! >>> 16);

/** Loads `sd` into a layer with only a weight and returns what reached its
 *  upload. */
async function uploadedWeight(layer: AnyModule, sd: LazyStateDict) {
  const upload = vi.fn((_data: Float32Array | Uint32Array | Uint16Array, shape: ValueMeta) =>
    pending(shape, 'reshape', []),
  );
  await loadLazyStateDict(layer, sd, { upload });
  return upload.mock.calls[0]![0];
}

/** The array type and raw bit patterns, so -0, subnormals and infinities
 *  compare exactly. */
function bitsOf(data: Float32Array | Uint32Array | Uint16Array) {
  if (data instanceof Float32Array) return { type: 'f32', bits: [...new Uint32Array(data.buffer)] };
  if (data instanceof Uint16Array) return { type: 'f16', bits: [...data] };
  return { type: 'u32', bits: [...data] };
}

/** f16 bits a widening to f32 could disturb: 1, −0, the smallest and largest
 *  subnormals, the largest finite value, −inf. */
const EDGE_BITS = [0x3c00, 0x8000, 0x0001, 0x03ff, 0x7bff, 0xfc00] as const;

/** Every f16 bit pattern but NaN. f16 → f32 → f16 maps each of these back to
 *  itself; NaN payloads collapse to one quiet NaN on that trip. */
const ALL_F16 = Uint16Array.from({ length: 1 << 16 }, (_, i) => i).filter(
  (b) => (b & 0x7c00) !== 0x7c00 || (b & 0x3ff) === 0,
);

describe('Linear', () => {
  it('stores the weight as [in, out] and the bias as [out]', () => {
    expect(shapes(new Linear(3, 4))).toEqual({ weight: meta([3, 4]), bias: meta([4]) });
  });

  it('can go without a bias', () => {
    expect(Object.keys(shapes(new Linear(3, 4, { bias: false })))).toEqual(['weight']);
  });

  it('maps [T, in] to [T, out]', () => {
    expect(fakeLoad(new Linear(3, 4)).forward(input([5, 3])).shape).toEqual(meta([5, 4]));
  });

  it('transposes a torch [out, in] weight at load', async () => {
    const sd = memoryStateDict({
      'fc.weight': { data: new Float32Array([1, 2, 3, 4, 5, 6]), shape: [2, 3] },
    });
    new Linear(3, 2).transformStateDict(sd, 'fc');
    const w = sd.tensors.get('fc.weight')!;
    expect(w.shape).toEqual([3, 2]);
    expect(await w.f32()).toEqual(new Float32Array([1, 4, 2, 5, 3, 6]));
    expect(w.byteLength).toBe(24); // still counts the download
  });

  it('rejects a weight already in [in, out] layout', () => {
    const sd = memoryStateDict({ 'fc.weight': { data: new Float32Array(6), shape: [3, 2] } });
    expect(() => new Linear(3, 2).transformStateDict(sd, 'fc')).toThrow(
      expect.objectContaining({ code: 'CHECKPOINT_MISMATCH' }),
    );
  });

  it('leaves a missing weight for the strict key check to report', () => {
    const sd = memoryStateDict({});
    new Linear(3, 2).transformStateDict(sd, 'fc');
    expect(sd.tensors.size).toBe(0);
  });

  it('uses the bare name at the root', () => {
    const sd = memoryStateDict({ weight: { data: new Float32Array(6), shape: [2, 3] } });
    new Linear(3, 2).transformStateDict(sd, '');
    expect(sd.tensors.get('weight')!.shape).toEqual([3, 2]);
  });

  // torch [2, 3] rows [a b c], [d e f] load as [3, 2]: a d, b e, c f.
  const [a, b, c, d, e, f] = EDGE_BITS;
  it.each([
    {
      name: 'F16 into an f16 layer uploads the halfwords transposed',
      checkpoint: () => f16Checkpoint(new Uint16Array(EDGE_BITS), [2, 3]),
      half: true,
      want: { type: 'f16', bits: [a, d, b, e, c, f] },
    },
    {
      name: 'F16 into an f32 layer decodes the transposed halfwords',
      checkpoint: () => f16Checkpoint(new Uint16Array(EDGE_BITS), [2, 3]),
      half: false,
      // a d b e c f = 1, (1023/1024)·2^-14, -0, 65504, 2^-24, -inf
      want: bitsOf(new Float32Array([1, (1023 / 1024) * 2 ** -14, -0, 65504, 2 ** -24, -Infinity])),
    },
    {
      name: 'F32 into an f32 layer uploads f32 transposed',
      checkpoint: async () =>
        memoryStateDict({ weight: { data: new Float32Array([1, 2, 3, 4, 5, 6]), shape: [2, 3] } }),
      half: false,
      want: bitsOf(new Float32Array([1, 4, 2, 5, 3, 6])),
    },
    {
      name: 'F32 into an f16 layer narrows the transposed f32',
      checkpoint: async () =>
        memoryStateDict({ weight: { data: new Float32Array([1, 2, 3, 4, 5, 6]), shape: [2, 3] } }),
      half: true,
      // f16 bits of 1, 4, 2, 5, 3, 6
      want: { type: 'f16', bits: [0x3c00, 0x4400, 0x4000, 0x4500, 0x4200, 0x4600] },
    },
    {
      name: 'BF16 into an f32 layer decodes, then transposes',
      checkpoint: async () =>
        memoryStateDict({ weight: { data: bf16([1, 2, 3, 4, 5, 6]), shape: [2, 3] } }),
      half: false,
      want: bitsOf(new Float32Array([1, 4, 2, 5, 3, 6])),
    },
    {
      name: 'BF16 into an f16 layer decodes, transposes, then narrows',
      checkpoint: async () =>
        memoryStateDict({ weight: { data: bf16([1, 2, 3, 4, 5, 6]), shape: [2, 3] } }),
      half: true,
      // f16 bits of 1, 4, 2, 5, 3, 6
      want: { type: 'f16', bits: [0x3c00, 0x4400, 0x4000, 0x4500, 0x4200, 0x4600] },
    },
    {
      name: 'derived halfwords into an f16 layer upload transposed',
      checkpoint: async () => derivedCheckpoint(new Uint16Array(EDGE_BITS), [2, 3]),
      half: true,
      want: { type: 'f16', bits: [a, d, b, e, c, f] },
    },
    {
      name: 'derived halfwords into an f32 layer decode transposed',
      checkpoint: async () => derivedCheckpoint(new Uint16Array(EDGE_BITS), [2, 3]),
      half: false,
      want: bitsOf(new Float32Array([1, (1023 / 1024) * 2 ** -14, -0, 65504, 2 ** -24, -Infinity])),
    },
  ])('$name', async ({ checkpoint, half, want }) => {
    const layer = new Linear(3, 2, { bias: false });
    if (half) layer.half();
    expect(bitsOf(await uploadedWeight(layer, await checkpoint()))).toEqual(want);
  });

  it('rejects derived packed words', async () => {
    const sd = derivedCheckpoint(new Uint32Array(6), [2, 3]);
    await expect(uploadedWeight(new Linear(3, 2, { bias: false }), sd)).rejects.toThrow(
      expect.objectContaining({
        code: 'CHECKPOINT_MISMATCH',
        message: 'transposeWeight: source holds packed u32 words',
      }),
    );
  });

  it('transposes F16 halfwords without decoding them to f32', async () => {
    const sd = await f16Checkpoint(new Uint16Array(EDGE_BITS), [2, 3]);
    const f32 = vi.spyOn(sd.tensors.get('weight')!, 'f32');
    new Linear(3, 2, { bias: false }).transformStateDict(sd, '');
    const w = sd.tensors.get('weight')!;
    if (w.kind !== 'derived') throw new Error('expected a derived tensor');
    expect(await w.data()).toBeInstanceOf(Uint16Array);
    expect(w.byteLength).toBe(12);
    expect(f32).not.toHaveBeenCalled();
  });

  // The reference is the f32 round trip: decode, transpose in f32, then
  // narrow for an f16 layer.
  it.each([
    { name: 'an f16 layer', half: true },
    { name: 'an f32 layer', half: false },
  ])(
    'loads every F16 value into $name with the same bits as the f32 round trip',
    async ({ half }) => {
      // 63490 values as 70 × 907.
      const rows = 70;
      const cols = ALL_F16.length / rows;
      const values = f16ToF32(ALL_F16);
      const transposed = new Float32Array(values.length);
      for (let r = 0; r < rows; r++) {
        for (let col = 0; col < cols; col++) transposed[col * rows + r] = values[r * cols + col]!;
      }
      const want = half ? f32ArrayToF16Bits(transposed).data : transposed;
      const layer = new Linear(cols, rows, { bias: false });
      if (half) layer.half();
      const got = await uploadedWeight(layer, await f16Checkpoint(ALL_F16, [rows, cols]));
      expect(bitsOf(got)).toEqual(bitsOf(want));
    },
  );
});

/** A quantized file of `entries` as a lazy state dict. */
const quantCheckpoint = (bits: 4 | 8, groupSize: number, entries: Record<string, QuantEntry>) =>
  fromSafetensors(bufferSource(quantFile(bits, groupSize, entries)), { chunkBytes: 0 });

describe('Linear with a quantized weight', () => {
  // torch [out 4, in 8], value o·8 + i, so a kernel column (one torch row)
  // holds o·8 + 0..7 and its 4-row chunks are 4 consecutive bytes.
  const numbered = Array.from({ length: 32 }, (_, i) => i);
  // f16-exact scales [out 4, in/4 = 2]: o + g/2
  const scales = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5];
  const entry: QuantEntry = { ints: numbered, scales, shape: [4, 8] };

  // int4 packs out features 8 to a word, so it takes [out 8, in 8].
  const int4Entry: QuantEntry = {
    ints: Array.from({ length: 64 }, (_, i) => (i % 16) - 8),
    scales: Array.from({ length: 16 }, (_, i) => i / 2),
    shape: [8, 8],
  };

  it.each([
    { name: 'int8', bits: 8 as const, quant: entry, words: 8 },
    { name: 'int4', bits: 4 as const, quant: int4Entry, words: 8 },
  ])('retypes the weight to $name quantW and adds f32 scales', async ({ bits, quant, words }) => {
    const out = quant.shape[0];
    const layer = new Linear(8, out);
    const sd = await quantCheckpoint(bits, 4, { 'fc.weight': quant });
    layer.transformStateDict(sd, 'fc');
    expect(shapes(layer)).toEqual({
      weight: { elems: words, dtype: 'quantW', dims: [8, out] },
      bias: meta([out]),
      weightScale: meta([2, out]),
    });
    expect(layer.weightScale!.pinned).toBe(true);
    expect([...sd.tensors.keys()]).toEqual(['fc.weight', 'fc.weightScale']);
    expect(sd.tensors.get('fc.weight')!.shape).toEqual([8, out]);
  });

  it('packs the torch [out, in] int8 ints as kernel [in, out] tiles', async () => {
    const sd = await quantCheckpoint(8, 4, { weight: entry });
    new Linear(8, 4).transformStateDict(sd, '');
    // Tile (unit 0, rows 0..3) then (unit 0, rows 4..7); column c of a tile
    // is torch row c, so its word holds bytes c·8 + 4·k4 + 0..3.
    expect([...(await sd.tensors.get('weight')!.words())]).toEqual([
      0x03020100, 0x0b0a0908, 0x13121110, 0x1b1a1918, 0x07060504, 0x0f0e0d0c, 0x17161514,
      0x1f1e1d1c,
    ]);
  });

  it('packs int4 ints two columns to a word', async () => {
    const sd = await quantCheckpoint(4, 4, { weight: int4Entry });
    new Linear(8, 8).transformStateDict(sd, '');
    // Column c is torch row c: i - 8 for an even row, i for an odd one. A
    // word holds an even column's 4 nibbles, then an odd one's.
    expect([...(await sd.tensors.get('weight')!.words())]).toEqual([
      0x3210ba98,
      0x3210ba98,
      0x7654fedc,
      0x7654fedc, // columns 0..3: rows 0..3, rows 4..7
      0x3210ba98,
      0x3210ba98,
      0x7654fedc,
      0x7654fedc, // columns 4..7
    ]);
  });

  it('transposes the scales to [in / groupSize, out]', async () => {
    const sd = await quantCheckpoint(8, 4, { weight: entry });
    new Linear(8, 4).transformStateDict(sd, '');
    const scale = sd.tensors.get('weightScale')!;
    expect(scale.shape).toEqual([2, 4]);
    expect(await contents(scale)).toEqual({
      type: 'f32',
      values: [0, 1, 2, 3, 0.5, 1.5, 2.5, 3.5],
    });
  });

  it('takes f32 scales from scaleWeight as they are', async () => {
    const sd = await quantCheckpoint(8, 4, { weight: entry });
    sd.tensors.set('weight', scaleWeight(sd.tensors.get('weight')!, [1, 2, 3, 4]));
    new Linear(8, 4).transformStateDict(sd, '');
    // torch row o times o + 1, then transposed
    expect(await contents(sd.tensors.get('weightScale')!)).toEqual({
      type: 'f32',
      values: [0, 2, 6, 12, 0.5, 3, 7.5, 14],
    });
  });

  it('puts the scales right after the weight, so binding keeps file order', async () => {
    const sd = await quantCheckpoint(8, 4, { 'fc.weight': entry });
    const float = derive([4], [], () => Promise.resolve(new Float32Array(4)));
    sd.tensors = new Map([
      ['first', float],
      ['fc.weight', sd.tensors.get('fc.weight')!],
      ['fc.bias', float],
    ]);
    new Linear(8, 4).transformStateDict(sd, 'fc');
    expect([...sd.tensors.keys()]).toEqual(['first', 'fc.weight', 'fc.weightScale', 'fc.bias']);
  });

  it('gives the words the bytes of the ints and scales, and the scales none', async () => {
    const sd = await quantCheckpoint(4, 4, { weight: int4Entry });
    const total = sd.totalBytes();
    new Linear(8, 8).transformStateDict(sd, '');
    expect(sd.tensors.get('weight')!.byteLength).toBe(32 + 32); // 64 nibbles, 16 f16 scales
    expect(sd.tensors.get('weightScale')!.byteLength).toBe(0);
    expect(sd.totalBytes()).toBe(total);
  });

  it('keeps the bias of a half() layer f16 and the scales f32', async () => {
    const layer = new Linear(8, 4).half();
    layer.transformStateDict(await quantCheckpoint(8, 4, { weight: entry }), '');
    expect(shapes(layer)).toEqual({
      weight: { elems: 8, dtype: 'quantW', dims: [8, 4] },
      bias: meta([4], 'f16'),
      weightScale: meta([2, 4]),
    });
  });

  // The quantized matmul takes in features a multiple of 8 and of the group
  // size, out features a multiple of 4 (int8) or 8 (int4) and a group size a
  // multiple of 4. A case names the one limit it sits on.
  it.each([
    // quantized: in 8 % 8 == 0, out 4 % 4 == 0, groups of 4
    { name: 'int8 [4, 8] in groups of 4', inF: 8, outF: 4, bits: 8 as const, groupSize: 4 },
    // quantized: int4 out 8 % 8 == 0
    { name: 'int4 [8, 8] in groups of 4', inF: 8, outF: 8, bits: 4 as const, groupSize: 4 },
    // quantized: one group over the whole row, 16 % 4 == 0
    { name: 'int8 [4, 16] in groups of 16', inF: 16, outF: 4, bits: 8 as const, groupSize: 16 },
  ])('loads $name as quantW', async ({ inF, outF, bits, groupSize }) => {
    const layer = new Linear(inF, outF);
    const sd = await quantCheckpoint(bits, groupSize, {
      weight: {
        ints: new Array(inF * outF).fill(0),
        scales: new Array((inF * outF) / groupSize).fill(1),
        shape: [outF, inF],
      },
    });
    layer.transformStateDict(sd, '');
    expect(shapes(layer)).toEqual({
      weight: { elems: (inF * outF * bits) / 32, dtype: 'quantW', dims: [inF, outF] },
      bias: meta([outF]),
      weightScale: meta([inF / groupSize, outF]),
    });
    expect(sd.quantizedAsFloat).toBeUndefined();
  });

  it.each([
    // float: in 12 % 8 != 0
    { name: 'in features off the 8-row tiles', inF: 12, outF: 4, bits: 8 as const, groupSize: 4 },
    // float: in 4 % 8 != 0
    { name: 'in features under one tile', inF: 4, outF: 4, bits: 8 as const, groupSize: 4 },
    // float: int8 out 6 % 4 != 0
    { name: 'int8 out features off 4 per word', inF: 8, outF: 6, bits: 8 as const, groupSize: 4 },
    // float: int4 out 4 % 8 != 0
    { name: 'int4 out features off 8 per word', inF: 8, outF: 4, bits: 4 as const, groupSize: 4 },
    // float: group size 2 % 4 != 0
    { name: 'a group size off 4', inF: 16, outF: 4, bits: 8 as const, groupSize: 2 },
    // float: group size 1
    { name: 'a scale per value', inF: 8, outF: 4, bits: 8 as const, groupSize: 1 },
  ])('loads $name as its dequantized floats', async ({ inF, outF, bits, groupSize }) => {
    // ints −7 … 7 and one scale per group, (g % 5 + 1) / 4, f16-exact
    const ints = Array.from({ length: inF * outF }, (_, i) => (i % 15) - 7);
    const scales = Array.from({ length: (inF * outF) / groupSize }, (_, g) => ((g % 5) + 1) / 4);
    const sd = await quantCheckpoint(bits, groupSize, {
      weight: { ints, scales, shape: [outF, inF] },
    });
    const layer = new Linear(inF, outF);
    layer.transformStateDict(sd, '');
    expect(shapes(layer)).toEqual({ weight: meta([inF, outF]), bias: meta([outF]) });
    expect(layer.weightScale).toBeUndefined();
    expect(sd.quantizedAsFloat).toEqual(['weight']);
    // stored [in, out]: row i, column o is torch value o·in + i
    const want = Array.from({ length: inF * outF }, (_, j) => {
      const at = (j % outF) * inF + Math.floor(j / outF);
      return ints[at]! * scales[Math.floor(at / groupSize)]!;
    });
    expect([...(await sd.tensors.get('weight')!.f32())]).toEqual(want);
  });

  it.each([
    { name: 'f32', half: false },
    { name: 'f16', half: true },
  ])(
    'runs the $name forward as a quantized matmul with bias and activation in the epilogue',
    async ({ half }) => {
      const layer = new Linear(8, 4, { activation: 'gelu' });
      if (half) layer.half();
      layer.transformStateDict(await quantCheckpoint(8, 4, { weight: entry }), '');
      fakeLoad(layer);
      const x = input([5, 8], half ? 'f16' : 'f32');
      const addend = input([5, 4], half ? 'f16' : 'f32');
      const y = layer.forward(x, { addend });
      expect(y.shape).toEqual(meta([5, 4], half ? 'f16' : 'f32'));
      expect(y.pending?.op).toBe('matmulQuantW');
      expect(y.pending?.inputs).toEqual([
        x,
        layer.weight.value,
        layer.weightScale!.value,
        layer.bias!.value,
        addend,
      ]);
      // baseRow, scaleBase, bits, groupSize, bias, addend, activation
      expect(y.pending?.attrs).toEqual([0, 0, 8, 4, 1, 1, ACT_CODE.gelu]);
    },
  );

  it('refuses a weight in [in, out] layout', async () => {
    const sd = await quantCheckpoint(8, 4, { weight: { ints: numbered, scales, shape: [8, 4] } });
    expect(() => new Linear(8, 4).transformStateDict(sd, '')).toThrow(
      expect.objectContaining({ code: 'CHECKPOINT_MISMATCH' }),
    );
  });
});

/** How a part's weight sits in the checkpoint. A float part has no group
 *  size. */
type PartSpec = {
  out: number;
  bias?: boolean;
  dtype: 'F32' | 'F16' | 'int8' | 'int4';
  groupSize?: number;
};

/** Linears over `inFeatures` inputs as the fields part0, part1, … and a
 *  FusedLinear over them as `fused`. */
class FusedParts extends Module {
  readonly parts: Linear[];
  readonly fused: FusedLinear;
  constructor(inFeatures: number, specs: readonly PartSpec[]) {
    super();
    this.parts = specs.map(({ out, bias }) => new Linear(inFeatures, out, { bias }));
    this.parts.forEach((part, p) => {
      (this as Record<string, unknown>)[`part${p}`] = part;
    });
    this.fused = new FusedLinear(this.parts);
  }
}

/** Part p's torch [out, in] weight. A float part holds p + o/4 + i/8. A
 *  quantized part holds ints −7 … 7 and one scale per group, 0.25 … 1. Every
 *  value and every int · scale is exact in f16. */
function partWeight({ out, dtype, groupSize = 1 }: PartSpec, p: number, inFeatures: number) {
  const ints = Array.from({ length: out * inFeatures }, (_, j) => ((p * 7 + j * 3) % 15) - 7);
  const scales = Array.from(
    { length: (out * inFeatures) / groupSize },
    (_, g) => (((p + g) % 4) + 1) / 4,
  );
  const floats = Array.from(
    { length: out * inFeatures },
    (_, j) => p + Math.floor(j / inFeatures) / 4 + (j % inFeatures) / 8,
  );
  const quantized = dtype === 'int8' || dtype === 'int4';
  return {
    ints,
    scales,
    /** The values the weight stands for. */
    values: quantized ? ints.map((q, j) => q * scales[Math.floor(j / groupSize)]!) : floats,
  };
}

/** The dotted path of every module of `model`, as loadStateDict hands it to
 *  fuseStateDict. */
const pathsOf = (model: AnyModule) =>
  new Map<AnyModule, string[]>([
    [model, ['']],
    ...[...model.namedModules()].map(([path, module]): [AnyModule, string[]] => [module, [path]]),
  ]);

/** Part p's bias: 10p + o. */
const partBias = (p: number, out: number) => Array.from({ length: out }, (_, o) => 10 * p + o);

/** The checkpoint of `specs`: `part<p>.weight`, and `part<p>.bias` unless
 *  the part has none. Float parts are all F32 or all F16. */
async function partsCheckpoint(inFeatures: number, specs: readonly PartSpec[]) {
  const floats: Record<string, Entry> = {};
  const quantized: Record<string, BitsQuantEntry> = {};
  for (const [p, spec] of specs.entries()) {
    const { ints, scales, values } = partWeight(spec, p, inFeatures);
    const shape: [number, number] = [spec.out, inFeatures];
    if (spec.dtype === 'int8' || spec.dtype === 'int4') {
      quantized[`part${p}.weight`] = { ints, scales, shape, bits: spec.dtype === 'int8' ? 8 : 4 };
    } else {
      floats[`part${p}.weight`] = { values, shape };
    }
    if (spec.bias !== false) {
      floats[`part${p}.bias`] = { values: partBias(p, spec.out), shape: [spec.out] };
    }
  }
  if (specs.some((spec) => spec.dtype === 'F16')) return checkpoint('F16', floats);
  const f32 = Object.fromEntries(
    Object.entries(floats).map(([name, { values, shape }]) => [
      name,
      { data: new Float32Array(values), shape },
    ]),
  );
  return fromSafetensors(bufferSource(mixedQuantFile(quantized, f32)), { chunkBytes: 0 });
}

/** The torch [Σout, in] weight and [Σout] bias a single Linear would hold
 *  for `specs`: the parts' rows one after another, zeros for a part without
 *  a bias. */
function stackedByHand(inFeatures: number, specs: readonly PartSpec[]) {
  return {
    out: specs.reduce((sum, spec) => sum + spec.out, 0),
    weight: specs.flatMap((spec, p) => partWeight(spec, p, inFeatures).values),
    bias: specs.flatMap((spec, p) =>
      spec.bias === false ? new Array<number>(spec.out).fill(0) : partBias(p, spec.out),
    ),
    hasBias: specs.some((spec) => spec.bias !== false),
  };
}

/** Loads `sd` into `model` and returns what reached the upload, by state-dict
 *  key. */
async function uploads(model: AnyModule, sd: LazyStateDict, strict = false) {
  const names: string[] = [];
  const upload = vi.fn((_data: Float32Array | Uint32Array | Uint16Array, shape: ValueMeta) =>
    pending(shape, 'reshape', []),
  );
  await loadLazyStateDict(model, sd, { upload, strict, onProgress: (name) => names.push(name) });
  return Object.fromEntries(
    upload.mock.calls.map(([data, shape], i) => [names[i]!, { shape, ...bitsOf(data) }]),
  );
}

/** What loading `specs` through a FusedLinear uploads, with the
 *  loadStateDict warnings it raised. */
async function fusedUploads(inFeatures: number, specs: readonly PartSpec[], half = false) {
  const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
  onTestFinished(() => warn.mockRestore());
  const model = new FusedParts(inFeatures, specs);
  if (half) model.half();
  const uploaded = await uploads(model, await partsCheckpoint(inFeatures, specs), true);
  return { model, uploaded, warnings: warn.mock.calls.map(([message]) => String(message)) };
}

/** What a single Linear named `fused` uploads from `sd`, which holds its
 *  `weight` and `bias`. */
async function singleUploads(layer: Linear, sd: LazyStateDict) {
  const uploaded = await uploads(layer, sd);
  return Object.fromEntries(
    Object.entries(uploaded).map(([name, value]) => [`fused.${name}`, value]),
  );
}

/** The checkpoint of the one Linear `specs` stack into, by hand: its `weight`
 *  as F32, F16, or the parts' ints and scales one after another, and its
 *  `bias` when a part has one. */
async function stackedCheckpoint(
  inFeatures: number,
  specs: readonly PartSpec[],
  stored: 'F32' | 'F16' | 'quant',
) {
  const { out, weight, bias, hasBias } = stackedByHand(inFeatures, specs);
  const shape: [number, number] = [out, inFeatures];
  if (stored !== 'quant') {
    return checkpoint(stored, {
      weight: { values: weight, shape },
      ...(hasBias ? { bias: { values: bias, shape: [out] } } : {}),
    });
  }
  const parts = specs.map((spec, p) => partWeight(spec, p, inFeatures));
  const file = quantFile(
    specs[0]!.dtype === 'int8' ? 8 : 4,
    specs[0]!.groupSize!,
    {
      weight: { ints: parts.flatMap((w) => w.ints), scales: parts.flatMap((w) => w.scales), shape },
    },
    hasBias ? { bias: { data: new Float32Array(bias), shape: [out] } } : {},
  );
  return fromSafetensors(bufferSource(file), { chunkBytes: 0 });
}

/** An f32 state dict of `name: [values, shape]` entries. */
const f32Dict = (entries: Record<string, [number[], number[]]>) =>
  memoryStateDict(
    Object.fromEntries(
      Object.entries(entries).map(([name, [values, shape]]) => [
        name,
        { data: new Float32Array(values), shape },
      ]),
    ),
  );

/** What `uploads` reports for f32 `values` bound to a slot of `dims`. */
const f32Upload = (dims: number[], values: number[]) => ({
  shape: meta(dims),
  ...bitsOf(new Float32Array(values)),
});

describe('FusedLinear', () => {
  const F32 = (out: number, bias?: boolean): PartSpec => ({ out, bias, dtype: 'F32' });
  const F16 = (out: number): PartSpec => ({ out, dtype: 'F16' });
  const int8 = (out: number, groupSize = 4): PartSpec => ({ out, dtype: 'int8', groupSize });
  const int4 = (out: number, groupSize = 4): PartSpec => ({ out, dtype: 'int4', groupSize });

  it('stacks the parts as the columns of one [in, Σout] weight, in part order', async () => {
    // torch rows: part0 [1 2], part1 [3 4] and [5 6]. Stored [in 2, out 3]:
    // input 0 holds 1 3 5, input 1 holds 2 4 6.
    const sd = f32Dict({
      'part0.weight': [
        [1, 2],
        [1, 2],
      ],
      'part0.bias': [[7], [1]],
      'part1.weight': [
        [3, 4, 5, 6],
        [2, 2],
      ],
      'part1.bias': [[8, 9], [2]],
    });
    expect(await uploads(new FusedParts(2, [F32(1), F32(2)]), sd, true)).toEqual({
      'fused.weight': f32Upload([2, 3], [1, 3, 5, 2, 4, 6]),
      'fused.bias': f32Upload([3], [7, 8, 9]),
    });
  });

  /** Loads `specs` through a FusedLinear and through one Linear over the
   *  rows stacked by hand, `stored` as that weight loads. Both upload the same. */
  async function expectLoadsAsOneLinear(
    specs: PartSpec[],
    stored: 'F32' | 'F16' | 'quant',
    half = false,
    warning?: string,
  ) {
    const { uploaded, warnings } = await fusedUploads(8, specs, half);
    const { out, hasBias } = stackedByHand(8, specs);
    const single = new Linear(8, out, { bias: hasBias });
    if (half) single.half();
    expect(uploaded).toEqual(
      await singleUploads(single, await stackedCheckpoint(8, specs, stored)),
    );
    expect(uploaded['fused.weight']!.shape.dtype).toBe(
      stored === 'quant' ? 'quantW' : half ? 'f16' : 'f32',
    );
    expect('fused.weightScale' in uploaded).toBe(stored === 'quant');
    expect(warnings).toEqual(warning ? [warning] : []);
  }

  // The quantized matmul takes in 8 % 8 == 0, out % 4 == 0 (int8) or
  // out % 8 == 0 (int4), and groups of a multiple of 4: the limits apply to
  // the stacked weight, not to a part.
  it.each<[name: string, specs: PartSpec[], stored: 'F32' | 'F16' | 'quant', half?: boolean]>([
    ['1 float part', [F32(3)], 'F32'],
    ['2 float parts', [F32(3), F32(5)], 'F32'],
    ['3 float parts', [F32(4), F32(2), F32(2)], 'F32'],
    ['5 float parts', [1, 4, 2, 7, 3].map((out) => F32(out)), 'F32'],
    ['3 parts without a bias', [4, 2, 2].map((out) => F32(out, false)), 'F32'],
    ['a part without a bias between two with one', [F32(2), F32(3, false), F32(1)], 'F32'],
    ['only the last part with a bias', [F32(2, false), F32(3)], 'F32'],
    ['F16 parts in an f32 model', [F16(2), F16(4)], 'F16'],
    ['F16 parts in a half() model', [F16(2), F16(4)], 'F16', true],
    ['1 int8 part', [int8(4)], 'quant'],
    ['3 int8 parts', [int8(4), int8(8), int8(4)], 'quant'],
    ['5 int8 parts of sizes off 4 that sum to 12', [1, 2, 3, 1, 5].map((o) => int8(o)), 'quant'],
    ['2 int4 parts, each too narrow to run quantized alone', [int4(4), int4(4)], 'quant'],
    ['3 int4 parts in groups of 8', [int4(8, 8), int4(4, 8), int4(4, 8)], 'quant'],
  ])('loads %s as one Linear of the stacked rows would', async (_, specs, stored, half) => {
    await expectLoadsAsOneLinear(specs, stored, half);
  });

  const DIFFER =
    'loadStateDict: fused layers loaded as float, their parts are not quantized alike: fused: ';
  // The warning names the fused layer and every part, by its file tensor.
  it.each<[name: string, specs: PartSpec[], warning: string]>([
    [
      'a float part next to an int8 one',
      [F32(4), int8(4)],
      `${DIFFER}part0.weight is float, part1.weight is int8 in groups of 4`,
    ],
    [
      'int8 next to int4',
      [int8(4), int4(8)],
      `${DIFFER}part0.weight is int8 in groups of 4, part1.weight is int4 in groups of 4`,
    ],
    [
      'int8 in groups of 4 next to groups of 8',
      [int8(4, 4), int8(4, 8)],
      `${DIFFER}part0.weight is int8 in groups of 4, part1.weight is int8 in groups of 8`,
    ],
    [
      'int8 between two float parts',
      [F32(2), int8(4), F32(3, false)],
      `${DIFFER}part0.weight is float, part1.weight is int8 in groups of 4, part2.weight is float`,
    ],
    // float: stacked out 4 + 2 = 6, 6 % 4 != 0
    [
      'int8 parts whose stacked size the quantized matmul cannot run',
      [int8(4), int8(2)],
      'loadStateDict: quantized weights loaded as float, the quantized kernel of their layer ' +
        'cannot run their shape or group size: fused (FusedLinear of part0.weight, part1.weight)',
    ],
  ])('loads %s as the dequantized floats, with one warning', async (_, specs, warning) => {
    await expectLoadsAsOneLinear(specs, 'F32', false, warning);
  });

  it('stacks F16 parts as halfwords, without decoding them to f32', async () => {
    const specs = [F16(2), F16(4)];
    const model = new FusedParts(8, specs);
    const sd = await partsCheckpoint(8, specs);
    model.fused.fuseStateDict(sd, pathsOf(model));
    expect((await contents(sd.tensors.get('fused.weight')!)).type).toBe('f16');
    expect((await contents(sd.tensors.get('fused.bias')!)).type).toBe('f16');
  });

  it('lists only the parts in the tree, and loads strict with nothing to report', async () => {
    const { model, warnings } = await fusedUploads(4, [F32(2), F32(3, false)]);
    expect(shapes(model)).toEqual({
      'part0.weight': meta([4, 2]),
      'part0.bias': meta([2]),
      'part1.weight': meta([4, 3]),
    });
    expect([...model.namedModules()].map(([name]) => name)).toEqual(['part0', 'part1', 'fused']);
    expect([...model.storageParameters()].map(([name]) => name)).toEqual([
      'fused.weight',
      'fused.bias',
    ]);
    expect(warnings).toEqual([]);
  });

  /** A [1] f32 tensor of `shape`, for a test to put in a state dict. */
  const zeros = (shape: number[]) =>
    f32Dict({
      t: [new Array<number>(shape.reduce((a, b) => a * b, 1)).fill(0), shape],
    }).tensors.get('t')!;

  // The model declares part0 [2, 4] and part1 [3, 4], both with a bias. The
  // wrong shapes would still stack, so each part is checked on its own.
  const MISMATCH = 'loadStateDict: state dict does not match model.';
  it.each<[name: string, change: (sd: LazyStateDict) => unknown, message: string]>([
    [
      "a part's missing weight, by its name in the checkpoint",
      (sd) => sd.tensors.delete('part1.weight'),
      `${MISMATCH} missing: part1.weight.`,
    ],
    [
      "a part's missing bias, by its name in the checkpoint",
      (sd) => sd.tensors.delete('part0.bias'),
      `${MISMATCH} missing: part0.bias.`,
    ],
    [
      'a tensor no parameter reads, when strict',
      (sd) => sd.tensors.set('extra', zeros([2])),
      `${MISMATCH} unexpected: extra.`,
    ],
    [
      'a file tensor named like its stacked weight, when strict',
      (sd) => sd.tensors.set('fused.weight', zeros([2, 4])),
      `${MISMATCH} unexpected: fused.weight.`,
    ],
    [
      "a part's weight that is not torch [out, in]",
      (sd) => sd.tensors.set('part1.weight', zeros([2, 4])),
      'Linear: part1.weight has shape [2,4], expected torch layout [3, 4]',
    ],
    [
      "a part's bias of another length, by the part's name",
      (sd) => sd.tensors.set('part1.bias', zeros([2])),
      'Linear: part1.bias has shape [2], expected [3]',
    ],
  ])('refuses %s', async (_, change, message) => {
    const specs = [F32(2), F32(3)];
    const sd = await partsCheckpoint(4, specs);
    change(sd);
    await expect(uploads(new FusedParts(4, specs), sd, true)).rejects.toMatchObject({
      code: 'CHECKPOINT_MISMATCH',
      message,
    });
  });

  it('warns about a file tensor named like its stacked weight, and loads the parts', async () => {
    const specs = [F32(2), F32(3, false)];
    const sd = await partsCheckpoint(4, specs);
    sd.tensors.set('fused.weight', zeros([5, 4]));
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    onTestFinished(() => warn.mockRestore());
    expect(await uploads(new FusedParts(4, specs), sd)).toEqual(
      await singleUploads(new Linear(4, 5), await stackedCheckpoint(4, specs, 'F32')),
    );
    expect(warn.mock.calls).toEqual([
      ['loadStateDict: state dict has tensors the model does not read: fused.weight.'],
    ]);
  });

  it('counts the bytes of every part in the load progress', async () => {
    const specs = [F32(2), F32(3, false)];
    const done: [string, number, number][] = [];
    await loadLazyStateDict(new FusedParts(4, specs), await partsCheckpoint(4, specs), {
      upload: (_data, shape) => pending(shape, 'reshape', []),
      onProgress: (...event) => done.push(event),
    });
    // part0's 2 biases, then the weights (2 + 3) · 4 f32, bound where the
    // file holds part1's, the largest
    expect(done).toEqual([
      ['fused.bias', 8, 88],
      ['fused.weight', 88, 88],
    ]);
  });

  class Bare extends Module {}
  /** q and k, each [1, 2] with a bias, and a hook that doubles q. */
  class Pair extends Module {
    q = new Linear(2, 1);
    k = new Linear(2, 1);
    override transformStateDict(sd: LazyStateDict, prefix: string): void {
      for (const name of [`${prefix}.q.weight`, `${prefix}.q.bias`]) {
        sd.tensors.set(name, scaleWeight(sd.tensors.get(name)!, 2));
      }
    }
  }
  const fusedPair = (pair: Pair) => new FusedLinear([pair.q, pair.k]);
  const holder = (fields: Record<string, unknown>): Module => Object.assign(new Bare(), fields);
  /** The checkpoint of a Pair at `path`: q and k, each a weight row and a bias. */
  const pairDict = (path: string, q: number[], qBias: number[], k: number[], kBias: number[]) =>
    f32Dict({
      [`${path}.q.weight`]: [q, [1, 2]],
      [`${path}.q.bias`]: [qBias, [1]],
      [`${path}.k.weight`]: [k, [1, 2]],
      [`${path}.k.bias`]: [kBias, [1]],
    });

  // The paths of the parts' module and of the FusedLinear. It stacks once
  // every hook has run, so q is doubled in each.
  it.each<[name: string, pair: string, fused: string, build: (pair: Pair) => Module]>([
    [
      'a field after the module whose hook edits a part',
      'pair',
      'qk',
      (pair) => Object.assign(new Bare(), { pair, qk: fusedPair(pair) }),
    ],
    [
      'a field before the module whose hook edits a part',
      'pair',
      'qk',
      (pair) => Object.assign(new Bare(), { qk: fusedPair(pair), pair }),
    ],
    [
      'a field of a sibling module of its parts',
      'pair',
      'fuser.qk',
      (pair) => Object.assign(new Bare(), { pair, fuser: holder({ qk: fusedPair(pair) }) }),
    ],
    [
      'a field above its parts, at any depth of the model',
      'layers.0.pair',
      'layers.0.qk',
      (pair) => holder({ layers: new ModuleList([holder({ pair, qk: fusedPair(pair) })]) }),
    ],
  ])('stacks its parts as the hooks left them, as %s', async (_, pair, fused, build) => {
    const sd = pairDict(pair, [1, 2], [5], [3, 4], [6]);
    expect(await uploads(build(new Pair()), sd, true)).toEqual({
      [`${fused}.weight`]: f32Upload([2, 2], [2, 3, 4, 4]),
      [`${fused}.bias`]: f32Upload([2], [10, 6]),
    });
  });

  const OUTSIDE_THE_TREE =
    "Linear at 'pair.q': the FusedLinear over it is not in the module tree. " +
    'Assign the FusedLinear to a field of a module of the model';
  it.each<[name: string, fields: (pair: Pair) => Record<string, unknown>, message: string]>([
    [
      'it is held in two fields',
      (pair) => {
        const qk = fusedPair(pair);
        return { qk, alias: qk };
      },
      "FusedLinear: held in two fields, 'qk' and 'alias'. Keep it in one",
    ],
    [
      'a part is held in two fields',
      (pair) => ({ alias: pair.q, qk: fusedPair(pair) }),
      "FusedLinear at 'qk': part 0 is held in two fields, 'pair.q' and 'alias'. Keep it in one",
    ],
    [
      'a part is not a field of the module tree',
      () => ({ qk: new FusedLinear([new Linear(2, 1)]) }),
      "FusedLinear at 'qk': part 0 of 1 is not in the module tree. " +
        'Assign every part to a field of a module of the model',
    ],
    [
      'it is held in an array, outside the module tree',
      (pair) => ({ list: [fusedPair(pair)] }),
      OUTSIDE_THE_TREE,
    ],
    [
      'it is held in an underscore field, outside the module tree',
      (pair) => ({ _qk: fusedPair(pair) }),
      OUTSIDE_THE_TREE,
    ],
    [
      'it is held in a local, outside the module tree',
      (pair) => (fusedPair(pair), {}),
      OUTSIDE_THE_TREE,
    ],
  ])('refuses the load when %s', async (_, fields, message) => {
    const pair = new Pair();
    const model = holder({ pair, ...fields(pair) });
    await expect(uploads(model, pairDict('pair', [0, 0], [0], [0, 0], [0]))).rejects.toThrow(
      message,
    );
  });

  const shared = new Linear(4, 2);
  new FusedLinear([shared]);
  const twice = new Linear(4, 2);
  it.each<[name: string, parts: Linear[], message: RegExp]>([
    ['no parts', [], /needs at least one part/],
    [
      'parts of different input sizes',
      [new Linear(4, 2), new Linear(8, 2)],
      /parts read 4 and 8 features/,
    ],
    [
      'a part with an activation',
      [new Linear(4, 2, { activation: 'gelu' })],
      /part 0 has the activation 'gelu'/,
    ],
    ['a part of another FusedLinear', [shared], /part 0 is already in another FusedLinear/],
    ['the same part twice', [twice, new Linear(4, 3), twice], /part 2 is listed twice/],
  ])('cannot be built from %s', (_, parts, message) => {
    expect(() => new FusedLinear(parts)).toThrow(message);
  });

  it('leaves the parts as they were when it cannot be built', () => {
    const good = new Linear(4, 2);
    expect(() => new FusedLinear([good, new Linear(8, 2)])).toThrow(/parts read 4 and 8/);
    expect(good.fused).toBe(false);
    expect(fakeLoad(good).forward(input([5, 4])).shape).toEqual(meta([5, 2]));
  });

  // File order k, o, q, v, as safetensors sorts an attention block. The
  // stacked weight carries its parts' sources, so it binds where the first
  // of these equally large parts sits and the loader walks the file once.
  it('binds each stacked weight where the file holds its parts, fetching every chunk once', async () => {
    class Attention extends Module {
      q_proj = new Linear(8, 8, { bias: false });
      k_proj = new Linear(8, 8, { bias: false });
      v_proj = new Linear(8, 8, { bias: false });
      o_proj = new Linear(8, 8, { bias: false });
      qkv = new FusedLinear([this.q_proj, this.k_proj, this.v_proj]);
    }
    class Model extends Module {
      embed = new Linear(8, 64, { bias: false });
      layers = new ModuleList([new Attention(), new Attention()]);
      head = new Linear(8, 64, { bias: false });
      // As the zoo hooks do: a part replaced by a scaled one.
      override transformStateDict(sd: LazyStateDict): void {
        for (const name of ['layers.0.q_proj.weight', 'layers.1.q_proj.weight']) {
          sd.tensors.set(name, scaleWeight(sd.tensors.get(name)!, 2));
        }
      }
    }
    // Two projections (256 bytes each) to a chunk, as a real file holds a
    // few layers per 8 MB chunk. The file spans more chunks than the 4 the
    // source keeps, so a weight read out of file order fetches one again.
    const chunks = await chunksFetched(new Model(), zeroCheckpoint(new Model(), [], {}), 512);
    expect(chunks.length).toBeGreaterThan(8);
    expect(chunks).toEqual(onceInFileOrder(chunks));
  });

  it('leaves its parts unloaded, and a part run alone throws', async () => {
    const { model } = await fusedUploads(4, [F32(2), F32(3)]);
    const [part] = model.parts;
    expect(part!.weight.loaded).toBe(false);
    expect(part!.bias!.loaded).toBe(false);
    expect(() => part!.forward(input([5, 4]))).toThrow(
      /this layer is a part of a FusedLinear, which holds its weight/,
    );
  });

  it.each<{ name: string; specs: PartSpec[]; stored: 'F32' | 'quant' }>([
    { name: 'float', specs: [F32(4), F32(8), F32(4)], stored: 'F32' },
    { name: 'int8', specs: [int8(4), int8(8), int8(4)], stored: 'quant' },
  ])('runs the $name forward as the one matmul a Linear of that size runs', async (c) => {
    const { model } = await fusedUploads(8, c.specs);
    const single = new Linear(8, 16);
    single.transformStateDict(await stackedCheckpoint(8, c.specs, c.stored), '');
    fakeLoad(single);
    for (const [, p] of model.storageParameters()) p.bind(pending(p.shape, 'reshape', []));
    const x = input([5, 8]);
    const got = model.fused.forward(x).pending;
    const want = single.forward(x).pending;
    const quantized = c.stored === 'quant';
    expect(got?.op).toBe(quantized ? 'matmulQuantW' : 'matmul');
    expect(got?.op).toBe(want?.op);
    expect(got?.attrs).toEqual(want?.attrs);
    // x, the weight, the scales of a quantized one, the bias
    expect(got?.inputs.length).toBe(quantized ? 4 : 3);
    expect(got?.inputs.map((v) => v.shape)).toEqual(want?.inputs.map((v) => v.shape));
  });

  it('half() retypes the stacked weight and bias, and dispose() frees them', async () => {
    const { model, uploaded } = await fusedUploads(4, [F32(2), F32(3)], true);
    expect(uploaded['fused.weight']!.shape).toEqual(meta([4, 5], 'f16'));
    expect(uploaded['fused.bias']!.shape).toEqual(meta([5], 'f16'));

    const stored = [...model.storageParameters()].map(([, p]) => p);
    const destroys = stored.map((p) => {
      const destroy = vi.fn();
      p.bind(materialized(p.shape, { destroy } as unknown as GpuBufferRef));
      return destroy;
    });
    expect(() => model.half()).toThrow(/'fused.weight' is already loaded/);
    model.dispose();
    for (const destroy of destroys) expect(destroy).toHaveBeenCalledOnce();
    expect(stored.map((p) => p.loaded)).toEqual([false, false]);
  });
});

describe('Conv1d and ConvTranspose1d with a quantized weight', () => {
  /** A [rows, cols] int8 entry of zeros, with `groups` scales per row. */
  const zeros = (rows: number, cols: number, groups: number): QuantEntry => ({
    ints: Array<number>(rows * cols).fill(0),
    scales: Array<number>(rows * groups).fill(1),
    shape: [rows, cols],
  });

  it('retypes a Conv1d weight to quantW [kernel·in, out] with its scales', async () => {
    const layer = new Conv1d(8, 4, 3);
    // groups of 12 values: 4 input channels with their 3 taps
    const sd = await quantCheckpoint(8, 12, { weight: zeros(4, 24, 2) });
    layer.transformStateDict(sd, '');
    expect(shapes(layer)).toEqual({
      weight: { elems: 24, dtype: 'quantW', dims: [24, 4] },
      bias: meta([4]),
      weightScale: meta([2, 4]),
    });
    expect([...sd.tensors.keys()]).toEqual(['weight', 'weightScale']);
  });

  it('pads a Conv1d to 1 output channel up to 4 columns, bias included', async () => {
    const layer = new Conv1d(8, 1, 3);
    const sd = await quantCheckpoint(8, 12, { weight: zeros(1, 24, 2) });
    sd.tensors.set('bias', memoryStateDict({ bias: Float32Array.of(2.5) }).tensors.get('bias')!);
    layer.transformStateDict(sd, '');
    expect(shapes(layer)).toEqual({
      weight: { elems: 24, dtype: 'quantW', dims: [24, 4] },
      bias: meta([4]),
      weightScale: meta([2, 4]),
    });
    expect([...(await sd.tensors.get('bias')!.f32())]).toEqual([2.5, 0, 0, 0]);
  });

  it('refuses a padded Conv1d bias of another length with CHECKPOINT_MISMATCH', async () => {
    const layer = new Conv1d(8, 1, 3);
    const sd = await quantCheckpoint(8, 12, { weight: zeros(1, 24, 2) });
    sd.tensors.set('bias', memoryStateDict({ bias: Float32Array.of(1, 2) }).tensors.get('bias')!);
    expect(() => layer.transformStateDict(sd, '')).toThrow(
      expect.objectContaining({
        code: 'CHECKPOINT_MISMATCH',
        message: expect.stringMatching(/bias has shape \[2\], expected \[1\]/),
      }),
    );
  });

  it('packs torch [out, in, kernel] ints tap-major and transposes the scales', async () => {
    // torch [out 4, in 4, kernel 2], value o·8 + c·2 + tap; scales [4, 2]
    // are o + g/2, f16-exact.
    const sd = await quantCheckpoint(8, 4, {
      weight: {
        ints: Array.from({ length: 32 }, (_, i) => i),
        scales: [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5],
        shape: [4, 8],
      },
    });
    new Conv1d(4, 4, 2).transformStateDict(sd, '');
    // Row tap·4 + c of column o holds o·8 + c·2 + tap: tile rows 0..3 are
    // tap 0, rows 4..7 tap 1.
    expect([...(await sd.tensors.get('weight')!.words())]).toEqual([
      0x06040200, 0x0e0c0a08, 0x16141210, 0x1e1c1a18, 0x07050301, 0x0f0d0b09, 0x17151311,
      0x1f1d1b19,
    ]);
    expect([...(await sd.tensors.get('weightScale')!.f32())]).toEqual([
      0, 1, 2, 3, 0.5, 1.5, 2.5, 3.5,
    ]);
  });

  it.for([8, 4] as const)(
    'takes the int%i Conv1d weight the Python quantizer writes',
    async (bits) => {
      // tools/python's golden file holds torch [out 6, in 16, kernel 4] as
      // [6, 64] in groups of 32. 6 outputs pad to the packing width, 8.
      const golden = new URL(
        `../../../../../tools/python/tests/golden/tiny.int${bits}.safetensors`,
        import.meta.url,
      );
      const sd = await fromSafetensors(bufferSource(readFileSync(golden)));
      const layer = new Conv1d(16, 6, 4, { bias: false });
      layer.transformStateDict(sd, 'conv');
      expect(shapes(layer)).toEqual({
        weight: { elems: (64 * 8 * bits) / 32, dtype: 'quantW', dims: [64, 8] },
        weightScale: meta([2, 8]),
      });
    },
  );

  it('keeps a Conv1d group that ends inside a channel', async () => {
    const layer = new Conv1d(8, 4, 3);
    // groups of 4 values: a channel's 3 taps and the next channel's first
    const sd = await quantCheckpoint(8, 4, { weight: zeros(4, 24, 6) });
    layer.transformStateDict(sd, '');
    expect(shapes(layer).weightScale).toEqual(meta([6, 4]));
  });

  it('retypes a ConvTranspose1d weight [in, out·kernel] to quantW [taps·in, stride·out]', async () => {
    const layer = new ConvTranspose1d(8, 4, 4, { stride: 2 });
    // groups of 4 values: one output's 4 kernel positions
    const sd = await quantCheckpoint(8, 4, { weight: zeros(8, 16, 4) });
    layer.transformStateDict(sd, '');
    expect(shapes(layer)).toEqual({
      weight: { elems: 32, dtype: 'quantW', dims: [16, 8] },
      bias: meta([8]),
      weightScale: meta([8, 4]),
    });
  });

  it('packs torch [in, out, kernel] ints by frame and tap and keeps the scales', async () => {
    // torch [in 4, out 4, kernel 4] with stride 2, so 2 taps; value
    // c·16 + o·4 + kk; scales [4, 2] are c + g/2, f16-exact.
    const scales = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5];
    const sd = await quantCheckpoint(8, 8, {
      weight: { ints: Array.from({ length: 64 }, (_, i) => i), scales, shape: [4, 16] },
    });
    new ConvTranspose1d(4, 4, 4, { stride: 2 }).transformStateDict(sd, '');
    // Row i·4 + c, column r·4 + o holds kernel position (1 − i)·2 + r. Tiles
    // run frame 0 taps 0 and 1, then frame 1.
    expect([...(await sd.tensors.get('weight')!.words())]).toEqual([
      0x32221202, 0x36261606, 0x3a2a1a0a, 0x3e2e1e0e, 0x30201000, 0x34241404, 0x38281808,
      0x3c2c1c0c, 0x33231303, 0x37271707, 0x3b2b1b0b, 0x3f2f1f0f, 0x31211101, 0x35251505,
      0x39291909, 0x3d2d1d0d,
    ]);
    expect([...(await sd.tensors.get('weightScale')!.f32())]).toEqual(scales);
  });

  it.each([
    { name: 'one input channel', cIn: 1, kernel: 3 },
    { name: 'kernel·in that is not a multiple of 8', cIn: 4, kernel: 3 },
    { name: 'in that is not a multiple of 4', cIn: 2, kernel: 4 },
  ])('loads a Conv1d weight on $name as its dequantized floats', async ({ cIn, kernel }) => {
    // One scale per output row, 0.5 and 0.25; ints 1, 2, … and −1, −2, …
    const cols = cIn * kernel;
    const ints = Array.from({ length: 2 * cols }, (_, i) => (i < cols ? i + 1 : cols - i - 1));
    const layer = new Conv1d(cIn, 2, kernel, { bias: false });
    const sd = await quantCheckpoint(8, cols, {
      weight: { ints, scales: [0.5, 0.25], shape: [2, cols] },
    });
    layer.transformStateDict(sd, '');
    expect(shapes(layer)).toEqual({ weight: meta([cols, 2]) });
    // tap-major [kernel·in, out]: row tap·in + c holds both outputs' (c, tap)
    const want = Array.from({ length: 2 * cols }, (_, i) => {
      const [row, o] = [Math.floor(i / 2), i % 2];
      const [tap, c] = [Math.floor(row / cIn), row % cIn];
      return ints[o * cols + c * kernel + tap]! * (o === 0 ? 0.5 : 0.25);
    });
    expect([...(await sd.tensors.get('weight')!.f32())]).toEqual(want);
  });

  // A depthwise conv never runs quantized, whatever its shape: 8 channels and
  // 8 taps would fit the quantized matmul as a groups-1 conv.
  it('loads a depthwise Conv1d weight as its dequantized floats', async () => {
    const [channels, kernel] = [8, 8];
    // torch [channels, kernel]: ints −30 … 33, one scale per row, (c + 1) / 4
    const ints = Array.from({ length: channels * kernel }, (_, i) => i - 30);
    const scales = Array.from({ length: channels }, (_, c) => (c + 1) / 4);
    const layer = new Conv1d(channels, channels, kernel, {
      groups: channels,
      padLeft: kernel - 1,
      bias: false,
    });
    const sd = await quantCheckpoint(8, kernel, {
      weight: { ints, scales, shape: [channels, kernel] },
    });
    layer.transformStateDict(sd, '');
    expect(shapes(layer)).toEqual({ weight: meta([kernel, channels]) });
    expect(layer.weightScale).toBeUndefined();
    expect(sd.quantizedAsFloat).toEqual(['weight']);
    // tap-major [kernel, channels]: row tap, column c holds torch's (c, tap)
    const want = Array.from({ length: kernel * channels }, (_, i) => {
      const [tap, c] = [Math.floor(i / channels), i % channels];
      return ints[c * kernel + tap]! * scales[c]!;
    });
    expect([...(await sd.tensors.get('weight')!.f32())]).toEqual(want);
  });

  it('refuses a quantized depthwise Conv1d weight of another shape', async () => {
    const layer = new Conv1d(4, 4, 2, { groups: 4, padLeft: 1, bias: false });
    const sd = await quantCheckpoint(8, 8, {
      weight: { ints: new Array<number>(32).fill(1), scales: [1, 1, 1, 1], shape: [4, 8] },
    });
    expect(() => layer.transformStateDict(sd, '')).toThrow(
      expect.objectContaining({
        code: 'CHECKPOINT_MISMATCH',
        message: 'Conv1d: weight is quantized as [4, 8], expected [out, kernel] = [4, 2]',
      }),
    );
  });

  it.each([
    { name: 'taps·in that is not a multiple of 8', cIn: 4, cOut: 4, kernel: 6 },
    { name: 'out that is not a multiple of 4', cIn: 8, cOut: 2, kernel: 4 },
  ])(
    'loads a ConvTranspose1d weight on $name as its dequantized floats',
    async ({ cIn, cOut, kernel }) => {
      // stride 2; one scale per torch row c, (c + 1) / 4; ints −25 … 24
      const [s, taps, cols] = [2, kernel / 2, cOut * kernel];
      const ints = Array.from({ length: cIn * cols }, (_, i) => (i % 50) - 25);
      const scales = Array.from({ length: cIn }, (_, c) => (c + 1) / 4);
      const layer = new ConvTranspose1d(cIn, cOut, kernel, { stride: s, bias: false });
      const sd = await quantCheckpoint(8, cols, {
        weight: { ints, scales, shape: [cIn, cols] },
      });
      layer.transformStateDict(sd, '');
      expect(shapes(layer)).toEqual({ weight: meta([taps * cIn, s * cOut]) });
      // row i·in + c, column r·out + o holds kernel position (taps − 1 − i)·s + r
      const want = Array.from({ length: taps * cIn * s * cOut }, (_, j) => {
        const [row, col] = [Math.floor(j / (s * cOut)), j % (s * cOut)];
        const [i, c, r, o] = [Math.floor(row / cIn), row % cIn, Math.floor(col / cOut), col % cOut];
        return ints[c * cols + o * kernel + (taps - 1 - i) * s + r]! * scales[c]!;
      });
      expect([...(await sd.tensors.get('weight')!.f32())]).toEqual(want);
    },
  );

  it.each([
    {
      name: 'a ConvTranspose1d weight in Conv1d layout',
      layer: () => new ConvTranspose1d(16, 8, 4, { stride: 2 }),
      file: () => quantCheckpoint(8, 8, { weight: zeros(8, 64, 8) }),
      error: /expected \[in, out·kernel\] = \[16, 32\]/,
    },
    {
      name: 'a weight of another shape on one input channel',
      layer: () => new Conv1d(1, 4, 3),
      file: () => quantCheckpoint(8, 4, { weight: zeros(3, 4, 1) }),
      error: /expected \[out, in·kernel\] = \[4, 3\]/,
    },
    {
      name: 'a weight of another shape',
      layer: () => new Conv1d(8, 4, 3),
      file: () => quantCheckpoint(8, 8, { weight: zeros(4, 16, 2) }),
      error: /expected \[out, in·kernel\] = \[4, 24\]/,
    },
  ])('refuses $name with CHECKPOINT_MISMATCH', async ({ layer, file, error }) => {
    const sd = await file();
    expect(() => layer().transformStateDict(sd, '')).toThrow(
      expect.objectContaining({
        code: 'CHECKPOINT_MISMATCH',
        message: expect.stringMatching(error),
      }),
    );
  });
});

describe('Conv1d', () => {
  it('stores the weight tap-major as [kernelSize·in, out]', () => {
    expect(shapes(new Conv1d(4, 6, 3))).toEqual({ weight: meta([12, 6]), bias: meta([6]) });
  });

  it('maps [T, in] to [T_out, out]', () => {
    const conv = fakeLoad(new Conv1d(4, 6, 3, { padding: 'same' }));
    expect(conv.forward(input([10, 4])).shape).toEqual(meta([10, 6]));
  });

  it('reorders a torch [out, in, k] weight at load', async () => {
    const sd = memoryStateDict({
      'c.weight': {
        data: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        shape: [2, 3, 2],
      },
    });
    new Conv1d(3, 2, 2).transformStateDict(sd, 'c');
    const w = sd.tensors.get('c.weight')!;
    // row tap·in + i, column out
    expect(w.shape).toEqual([6, 2]);
    expect(await w.f32()).toEqual(new Float32Array([1, 7, 3, 9, 5, 11, 2, 8, 4, 10, 6, 12]));
  });

  it('rejects a weight in another layout', () => {
    const sd = memoryStateDict({ 'c.weight': { data: new Float32Array(4), shape: [2, 2, 1] } });
    expect(() => new Conv1d(1, 2, 2).transformStateDict(sd, 'c')).toThrow(
      expect.objectContaining({ code: 'CHECKPOINT_MISMATCH' }),
    );
  });

  const depthwise = (channels: number, kernel: number) =>
    new Conv1d(channels, channels, kernel, {
      groups: channels,
      padLeft: kernel - 1,
      bias: false,
    });

  it.each([
    { name: 'groups that do not cover the channels', cIn: 8, cOut: 8, groups: 2 },
    { name: 'depthwise groups with more outputs than inputs', cIn: 8, cOut: 16, groups: 8 },
    { name: 'groups equal to the outputs alone', cIn: 4, cOut: 8, groups: 8 },
  ])('rejects $name with an INVALID_ARGUMENT', ({ cIn, cOut, groups }) => {
    expect(() => new Conv1d(cIn, cOut, 3, { groups })).toThrow(
      expect.objectContaining({
        code: 'INVALID_ARGUMENT',
        message: `Conv1d: groups=${groups} is neither 1 nor depthwise (cIn=${cIn}, cOut=${cOut})`,
      }),
    );
  });

  it('stores a depthwise weight tap-major as [kernelSize, channels]', () => {
    expect(shapes(depthwise(8, 3))).toEqual({ weight: meta([3, 8]) });
  });

  it('reorders a torch [channels, 1, k] depthwise weight at load', async () => {
    const sd = memoryStateDict({
      'c.weight': { data: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]), shape: [4, 1, 2] },
    });
    depthwise(4, 2).transformStateDict(sd, 'c');
    const w = sd.tensors.get('c.weight')!;
    // row tap, column channel
    expect(w.shape).toEqual([2, 4]);
    expect(await w.f32()).toEqual(new Float32Array([1, 3, 5, 7, 2, 4, 6, 8]));
  });

  it('rejects a depthwise weight stored with every input channel', () => {
    const sd = memoryStateDict({ 'c.weight': { data: new Float32Array(32), shape: [4, 4, 2] } });
    expect(() => depthwise(4, 2).transformStateDict(sd, 'c')).toThrow(
      expect.objectContaining({
        code: 'CHECKPOINT_MISMATCH',
        message: 'Conv1d: c.weight has shape [4,4,2], expected torch layout [4, 1, 2]',
      }),
    );
  });

  it('runs a depthwise conv as its own op and keeps [T, channels]', () => {
    const y = fakeLoad(depthwise(8, 3)).forward(input([10, 8]));
    expect(y.shape).toEqual(meta([10, 8]));
    expect(y.pending).toMatchObject({ op: 'depthwiseConv1d', attrs: [3] });
  });

  // torch [out 2, in 3, k 2] holding elements 0..11 loads tap-major as
  // 0 6 2 8 4 10 1 7 3 9 5 11 (row tap·in + i, column out), as in the f32
  // case above.
  const TAP_MAJOR = [0, 6, 2, 8, 4, 10, 1, 7, 3, 9, 5, 11];
  // EDGE_BITS, then each with its sign flipped.
  const halves = [...EDGE_BITS, ...EDGE_BITS.map((b) => b ^ 0x8000)];
  const reordered = new Uint16Array(TAP_MAJOR.map((i) => halves[i]!));
  const f32Values = Array.from({ length: 12 }, (_, i) => i + 1);
  it.each([
    {
      name: 'F16 into an f16 layer uploads the halfwords reordered',
      checkpoint: () => f16Checkpoint(new Uint16Array(halves), [2, 3, 2]),
      half: true,
      want: bitsOf(reordered),
    },
    {
      name: 'F16 into an f32 layer decodes the reordered halfwords',
      checkpoint: () => f16Checkpoint(new Uint16Array(halves), [2, 3, 2]),
      half: false,
      want: bitsOf(f16ToF32(reordered)),
    },
    {
      name: 'F32 into an f32 layer uploads f32 reordered',
      checkpoint: async () =>
        memoryStateDict({ weight: { data: new Float32Array(f32Values), shape: [2, 3, 2] } }),
      half: false,
      want: bitsOf(new Float32Array(TAP_MAJOR.map((i) => i + 1))),
    },
    {
      name: 'BF16 into an f32 layer decodes, then reorders',
      checkpoint: async () =>
        memoryStateDict({ weight: { data: bf16(f32Values), shape: [2, 3, 2] } }),
      half: false,
      want: bitsOf(new Float32Array(TAP_MAJOR.map((i) => i + 1))),
    },
    {
      name: 'BF16 into an f16 layer decodes, reorders, then narrows',
      checkpoint: async () =>
        memoryStateDict({ weight: { data: bf16(f32Values), shape: [2, 3, 2] } }),
      half: true,
      want: bitsOf(f32ArrayToF16Bits(new Float32Array(TAP_MAJOR.map((i) => i + 1))).data),
    },
    {
      name: 'derived halfwords into an f16 layer upload reordered',
      checkpoint: async () => derivedCheckpoint(new Uint16Array(halves), [2, 3, 2]),
      half: true,
      want: bitsOf(reordered),
    },
    {
      name: 'derived halfwords into an f32 layer decode reordered',
      checkpoint: async () => derivedCheckpoint(new Uint16Array(halves), [2, 3, 2]),
      half: false,
      want: bitsOf(f16ToF32(reordered)),
    },
  ])('$name', async ({ checkpoint, half, want }) => {
    const layer = new Conv1d(3, 2, 2, { bias: false });
    if (half) layer.half();
    expect(bitsOf(await uploadedWeight(layer, await checkpoint()))).toEqual(want);
  });

  it('rejects derived packed words', async () => {
    const sd = derivedCheckpoint(new Uint32Array(12), [2, 3, 2]);
    await expect(uploadedWeight(new Conv1d(3, 2, 2, { bias: false }), sd)).rejects.toThrow(
      expect.objectContaining({
        code: 'CHECKPOINT_MISMATCH',
        message: 'Conv1d: source holds packed u32 words',
      }),
    );
  });

  it('reorders F16 halfwords without decoding them to f32', async () => {
    const sd = await f16Checkpoint(new Uint16Array(halves), [2, 3, 2]);
    const f32 = vi.spyOn(sd.tensors.get('weight')!, 'f32');
    new Conv1d(3, 2, 2, { bias: false }).transformStateDict(sd, '');
    const w = sd.tensors.get('weight')!;
    if (w.kind !== 'derived') throw new Error('expected a derived tensor');
    expect(await w.data()).toBeInstanceOf(Uint16Array);
    expect(w.byteLength).toBe(24);
    expect(f32).not.toHaveBeenCalled();
  });

  // The reference is the f32 round trip: decode, reorder in f32, then narrow
  // for an f16 layer.
  it.each([
    { name: 'an f16 layer', half: true },
    { name: 'an f32 layer', half: false },
  ])(
    'loads every F16 value into $name with the same bits as the f32 round trip',
    async ({ half }) => {
      // 63490 values as out 10, in 907, k 7.
      const [cOut, cIn, k] = [10, 907, 7];
      const values = f16ToF32(ALL_F16);
      const tapMajor = new Float32Array(values.length);
      for (let o = 0; o < cOut; o++) {
        for (let i = 0; i < cIn; i++) {
          for (let t = 0; t < k; t++)
            tapMajor[(t * cIn + i) * cOut + o] = values[(o * cIn + i) * k + t]!;
        }
      }
      const want = half ? f32ArrayToF16Bits(tapMajor).data : tapMajor;
      const layer = new Conv1d(cIn, cOut, k, { bias: false });
      if (half) layer.half();
      const got = await uploadedWeight(layer, await f16Checkpoint(ALL_F16, [cOut, cIn, k]));
      expect(bitsOf(got)).toEqual(bitsOf(want));
    },
  );

  it('widens the window by the dilation', () => {
    const conv = (padding?: 'same') =>
      fakeLoad(new Conv1d(4, 6, 3, { dilation: 2, padding })).forward(input([10, 4])).shape;
    expect(conv()).toEqual(meta([6, 6]));
    expect(conv('same')).toEqual(meta([10, 6]));
  });
});

describe('ConvTranspose1d', () => {
  it('stores the weight as [taps·in, stride·out] and the bias once per frame', () => {
    expect(shapes(new ConvTranspose1d(4, 6, 4, { stride: 2 }))).toEqual({
      weight: meta([8, 12]),
      bias: meta([12]),
    });
  });

  it('maps [T, in] to [(T − 1)·stride − 2·padding + kernel + outputPadding, out]', () => {
    const conv = new ConvTranspose1d(4, 8, 6, { stride: 3, padding: 2, outputPadding: 1 });
    expect(fakeLoad(conv).forward(input([10, 4])).shape).toEqual(meta([30, 8]));
    const unpadded = new ConvTranspose1d(4, 8, 3, { stride: 3 });
    expect(fakeLoad(unpadded).forward(input([10, 4])).shape).toEqual(meta([30, 8]));
  });

  it('reorders a torch [in, out, k] weight and repeats the bias at load', async () => {
    const sd = memoryStateDict({
      'c.weight': { data: new Float32Array([1, 2, 3, 4]), shape: [1, 1, 4] },
      'c.bias': new Float32Array([5]),
    });
    new ConvTranspose1d(1, 1, 4, { stride: 2 }).transformStateDict(sd, 'c');
    // Tap 0 reads the later frame, so it takes kernel positions 2 and 3.
    expect(await sd.tensors.get('c.weight')!.f32()).toEqual(new Float32Array([3, 4, 1, 2]));
    expect(await sd.tensors.get('c.bias')!.f32()).toEqual(new Float32Array([5, 5]));
  });

  it('rejects a weight in another layout', () => {
    const sd = memoryStateDict({ 'c.weight': { data: new Float32Array(4), shape: [1, 4, 1] } });
    expect(() => new ConvTranspose1d(1, 1, 4, { stride: 2 }).transformStateDict(sd, 'c')).toThrow(
      expect.objectContaining({ code: 'CHECKPOINT_MISMATCH' }),
    );
  });

  it.each([
    { name: 'a kernel that is not a multiple of the stride', k: 5, opts: { stride: 2 } },
    {
      name: 'an outputPadding above the padding',
      k: 4,
      opts: { stride: 2, padding: 0, outputPadding: 1 },
    },
  ])('rejects $name', ({ k, opts }) => {
    expect(() => new ConvTranspose1d(2, 2, k, opts)).toThrow(/ConvTranspose1d/);
  });
});

describe('Snake', () => {
  it('declares a pinned f32 [2, channels] weight', () => {
    const s = new Snake(8);
    expect(shapes(s)).toEqual({ weight: meta([2, 8]) });
    expect(s.weight.pinned).toBe(true);
  });

  it('turns a torch alpha [1, C, 1] into alpha and 1 / (alpha + 1e-9) at load', async () => {
    const sd = memoryStateDict({
      's.alpha': { data: new Float32Array([0.5, 2]), shape: [1, 2, 1] },
    });
    new Snake(2).transformStateDict(sd, 's');
    expect(sd.tensors.has('s.alpha')).toBe(false);
    const w = sd.tensors.get('s.weight')!;
    expect(w.shape).toEqual([2, 2]);
    expect(await w.f32()).toEqual(new Float32Array([0.5, 2, 1 / (0.5 + 1e-9), 1 / (2 + 1e-9)]));
  });
});

describe('Conv2d', () => {
  it('declares an f32 hwc4 weight and bias that half() makes f16 and float() f32 again', () => {
    const conv = new Conv2d(8, 16, { kernelSize: 3 });
    const weight = (dtype: 'f32' | 'f16') => ({
      elems: 16 * 72,
      dtype,
      dims: [16, 8 * 3 * 3],
      layout: 'hwc4',
    });
    expect(shapes(conv)).toEqual({ weight: weight('f32'), bias: meta([16]) });
    expect(shapes(conv.half())).toEqual({ weight: weight('f16'), bias: meta([16], 'f16') });
    expect(shapes(conv.float())).toEqual({ weight: weight('f32'), bias: meta([16]) });
  });

  it('sizes a depthwise weight per channel', () => {
    expect(new Conv2d(8, 8, { kernelSize: 3, groups: 8 }).weight.shape.dims).toEqual([8, 9]);
  });

  it('takes a non-square kernel', () => {
    expect(new Conv2d(4, 2, { kernelSize: [1, 3] }).weight.shape.dims).toEqual([2, 12]);
  });

  it('rejects groups that are neither 1 nor depthwise', () => {
    expect(() => new Conv2d(8, 16, { kernelSize: 3, groups: 8 })).toThrow(
      /groups=8 is neither 1 nor depthwise \(cIn=8, cOut=16\)/,
    );
  });

  it('packs f16 weights into 4×4 tiles at load', () => {
    // [cOut=3, cIn=2, 1×1]: one tile laid out [in][out], zero-padded
    const packed = new Conv2d(2, 3, { kernelSize: 1 }).weight.preprocess!(
      new Uint16Array([1, 2, 3, 4, 5, 6]),
    );
    expect(packed).toEqual(new Uint16Array([1, 3, 5, 0, 2, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0]));
  });

  it('packs depthwise weights by groups of 4 channels', () => {
    const packed = new Conv2d(5, 5, { kernelSize: 1, groups: 5 }).weight.preprocess!(
      new Uint16Array([1, 2, 3, 4, 5]),
    );
    expect(packed).toEqual(new Uint16Array([1, 2, 3, 4, 5, 0, 0, 0]));
  });

  it('packs f32 weights the same way, as f32', () => {
    const packed = new Conv2d(2, 3, { kernelSize: 1 }).weight.preprocess!(
      new Float32Array([1, 2, 3, 4, 5, 6]),
    );
    expect(packed).toEqual(new Float32Array([1, 3, 5, 0, 2, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0]));
  });

  it('refuses an f32 weight for winograd at load', () => {
    const conv = new Conv2d(1, 1, { kernelSize: 3, padding: 1, winograd: true });
    expect(() => conv.weight.preprocess!(new Float32Array(9))).toThrow(
      expect.objectContaining({
        code: 'INVALID_ARGUMENT',
        message: 'Conv2d: winograd has no f32 kernel yet, run the model in f16 with half()',
      }),
    );
  });

  it('maps an hwc4 map to an hwc4 map', () => {
    const conv = fakeLoad(new Conv2d(8, 16, { kernelSize: 3, stride: 2, padding: 1 }).half());
    expect(conv.forward(hwc4Input(8, 32, 32)).shape).toEqual(hwc4Input(16, 16, 16).shape);
  });
});

describe('ConvTranspose2d', () => {
  it('declares an f16 [out·k·k, in] weight and an f32 bias', () => {
    const conv = new ConvTranspose2d(4, 3, { kernelSize: 2, stride: 2 });
    expect(shapes(conv)).toEqual({ weight: meta([12, 4], 'f16'), bias: meta([3]) });
    expect(conv.weight.pinned).toBe(true);
  });

  it('reorders f16 weights at load', () => {
    const p = new ConvTranspose2d(2, 3, { kernelSize: 1, stride: 1 }).weight;
    // [in=2, out=3] to [out, in]
    expect(p.preprocess!(new Uint16Array([1, 2, 3, 4, 5, 6]))).toEqual(
      new Uint16Array([1, 4, 2, 5, 3, 6]),
    );
  });

  it('scales each side by k', () => {
    const conv = fakeLoad(new ConvTranspose2d(4, 3, { kernelSize: 2, stride: 2 }));
    expect(conv.forward(input([4, 5, 6])).shape).toEqual(meta([3, 10, 12]));
  });
});

describe('BatchNorm2d', () => {
  /** Records every upload and hands back a stand-in of the slot's shape. */
  const recorder = () =>
    vi.fn((_data: Float32Array | Uint32Array | Uint16Array, shape: ValueMeta) =>
      pending(shape, 'reshape', []),
    );
  /** bf16 bits, which memoryStateDict decodes like a file tensor. */
  const bf16 = (values: number[]) =>
    Uint16Array.from(values, (v) => new Uint32Array(new Float32Array([v]).buffer)[0]! >>> 16);

  // Two 1×1 out channels over two in channels, eps = 1, worked by hand:
  //   scale   = gamma / sqrt(var + eps) = [4 / 2, 4 / 2] = [2, 2]
  //   weight' = weight · scale          = [[1, 2], [3, 4]] · 2 = [[2, 4], [6, 8]]
  //   bias'   = beta + (bias − mean) · scale = [0 + 0 · 2, 1 + 2 · 2] = [0, 5]
  const EPS = 1;
  const conv = (prefix = '') => ({
    [`${prefix}conv.weight`]: { data: bf16([1, 2, 3, 4]), shape: [2, 2, 1, 1] },
    [`${prefix}conv.bias`]: bf16([1, 2]),
  });
  const bn = (prefix = '') => ({
    [`${prefix}bn.weight`]: bf16([4, 4]),
    [`${prefix}bn.bias`]: bf16([0, 1]),
    [`${prefix}bn.running_mean`]: bf16([1, 0]),
    [`${prefix}bn.running_var`]: bf16([3, 3]),
  });
  // The folded weight, packed by Conv2d into one [in][out] hwc4 tile.
  const FOLDED_WEIGHT = new Float32Array(16);
  FOLDED_WEIGHT.set([2, 6, 0, 0, 4, 8]);
  const FOLDED_BIAS = new Float32Array([0, 5]);

  class ConvBn extends Module {
    conv = new Conv2d(2, 2, { kernelSize: 1, bias: true });
    bn = new BatchNorm2d(2, { eps: EPS });
  }

  it('folds into the conv declared right before it and loads only conv weights', async () => {
    const m = new ConvBn();
    const upload = recorder();
    await m.loadStateDict({ ...conv(), ...bn() }, { upload });
    expect(upload.mock.calls.map(([data]) => data)).toEqual([FOLDED_WEIGHT, FOLDED_BIAS]);
    expect(m.conv.weight.loaded).toBe(true);
    expect(m.conv.bias!.loaded).toBe(true);
  });

  it('finds its conv inside a nested block', async () => {
    class Net extends Module {
      block = new ConvBn();
    }
    const net = new Net();
    const upload = recorder();
    await net.loadStateDict({ ...conv('block.'), ...bn('block.') }, { upload });
    expect(upload.mock.calls[0]![0]).toEqual(FOLDED_WEIGHT);
    expect(net.block.conv.weight.loaded).toBe(true);
  });

  it('folds into a named conv with foldInto', async () => {
    class Apart extends Module {
      stem = new Conv2d(2, 2, { kernelSize: 1 });
      other = new Linear(2, 2);
      bn = new BatchNorm2d(2, { eps: EPS, foldInto: 'stem' });
    }
    const m = new Apart();
    const upload = recorder();
    await m.loadStateDict(
      {
        'stem.weight': { data: bf16([1, 2, 3, 4]), shape: [2, 2, 1, 1] },
        'other.weight': { data: new Float32Array(4), shape: [2, 2] },
        'other.bias': new Float32Array(2),
        ...bn(),
      },
      { upload },
    );
    expect(upload).toHaveBeenCalledWith(FOLDED_WEIGHT, m.stem.weight.shape);
    expect(m.stem.bias!.loaded).toBe(true);
  });

  it.each([
    {
      name: 'something other than a conv',
      Parent: class extends Module {
        fc = new Linear(2, 2);
        bn = new BatchNorm2d(2);
      },
    },
    {
      name: 'a conv with another channel count',
      Parent: class extends Module {
        conv = new Conv2d(2, 4, { kernelSize: 1 });
        bn = new BatchNorm2d(2);
      },
    },
    {
      name: 'a non-conv after a matching conv',
      Parent: class extends Module {
        conv = new Conv2d(2, 2, { kernelSize: 1 });
        fc = new Linear(2, 2);
        bn = new BatchNorm2d(2);
      },
    },
    {
      name: 'nothing',
      Parent: class extends Module {
        bn = new BatchNorm2d(2);
      },
    },
  ])('refuses to guess when right before it is $name', async ({ Parent }) => {
    await expect(new Parent().loadStateDict(bn(), { upload: recorder() })).rejects.toThrow(
      /cannot infer its conv — needs a Conv2d with 2 out-channels/,
    );
  });

  it('has no parameters and passes activations through', () => {
    const x = hwc4Input(2, 4, 4);
    expect([...new BatchNorm2d(2).namedParameters()]).toEqual([]);
    expect(new BatchNorm2d(2).forward(x)).toBe(x);
  });
});

describe('normalization layers', () => {
  it('LayerNorm has a weight, and a bias only when asked', () => {
    expect(Object.keys(shapes(new LayerNorm(8)))).toEqual(['weight']);
    expect(shapes(new LayerNorm(8, 1e-5, { bias: true }))).toEqual({
      weight: meta([8]),
      bias: meta([8]),
    });
  });

  it('GroupNorm has a weight and bias per channel, and groups that divide the channels', () => {
    expect(shapes(new GroupNorm(1, 8))).toEqual({ weight: meta([8]), bias: meta([8]) });
    expect(shapes(new GroupNorm(8, 8))).toEqual({ weight: meta([8]), bias: meta([8]) });
    expect(() => new GroupNorm(3, 8)).toThrow(/numGroups 3 must divide the 8 channels/);
  });

  it('RMSNorm has one weight per feature', () => {
    expect(shapes(new RMSNorm(8))).toEqual({ weight: meta([8]) });
  });

  it.each([
    ['LayerNorm', () => new LayerNorm(8, 1e-5, { bias: true })],
    ['GroupNorm', () => new GroupNorm(1, 8)],
    ['GroupNorm per channel', () => new GroupNorm(8, 8)],
    ['RMSNorm', () => new RMSNorm(8)],
  ] as const)('%s keeps the [T, N] shape', (_, make) => {
    expect(fakeLoad(make()).forward(input([5, 8])).shape).toEqual(meta([5, 8]));
  });
});

describe('Embedding', () => {
  it('stores a [vocab, dim] table on the GPU by default', () => {
    const e = new Embedding(100, 16);
    expect(shapes(e)).toEqual({ weight: meta([100, 16]) });
    expect(e.weight).not.toBeInstanceOf(CpuParameter);
  });

  it('looks up rows by token id', () => {
    const e = fakeLoad(new Embedding(100, 16));
    expect(e.forward([5, 0, 5]).shape).toEqual(meta([3, 16]));
    expect(e.forward(input([7])).shape).toEqual(meta([7, 16]));
  });

  describe('on the CPU', () => {
    // vocab 3, dim 2: row i holds [i, i + 0.5]
    const table = new Float32Array([0, 0.5, 1, 1.5, 2, 2.5]);
    const cpu = () => {
      const e = new Embedding(3, 2, { device: 'cpu' });
      expect(e.weight).toBeInstanceOf(CpuParameter);
      return e;
    };

    it('gathers rows from an f32 table', () => {
      const e = cpu();
      (e.weight as CpuParameter).bindData(table);
      expect(e.gather([2, 0, 2])).toEqual(new Float32Array([2, 2.5, 0, 0.5, 2, 2.5]));
    });

    it('expands rows from a bf16 table to f32', () => {
      const e = cpu();
      const bf16 = Uint16Array.from(
        table,
        (v) => new Uint32Array(new Float32Array([v]).buffer)[0]! >>> 16,
      );
      (e.weight as CpuParameter).bindData(bf16);
      expect(e.gather([1])).toEqual(new Float32Array([1, 1.5]));
    });

    it.each([3, -1, 0.5])('rejects token id %s', (id) => {
      const e = cpu();
      (e.weight as CpuParameter).bindData(table);
      expect(() => e.gather([id])).toThrow(/out of range \(vocab 3\)/);
    });

    it('cannot take token ids that live on the GPU', () => {
      expect(() => cpu().forward(input([2]))).toThrow(/GPU-resident ids need a gpu table/);
    });
  });

  it('gather() refuses a GPU table', () => {
    expect(() => new Embedding(3, 2).gather([0])).toThrow(/cpu-resident tables only/);
  });

  describe('with a quantized table', () => {
    it('uploads an int8 table as int · scale, one scale per row', async () => {
      const sd = await quantCheckpoint(8, 4, {
        weight: { ints: [1, -2, 3, -4, 5, 6, -7, 8], scales: [0.5, 2], shape: [2, 4] },
      });
      expect(await uploadedWeight(new Embedding(2, 4), sd)).toEqual(
        new Float32Array([0.5, -1, 1.5, -2, 10, 12, -14, 16]),
      );
    });

    it('gathers rows of an int4 table in groups of 2 on the CPU', async () => {
      const sd = await quantCheckpoint(4, 2, {
        weight: { ints: [1, -2, 3, -4, 5, 6, -7, 7], scales: [0.5, 1, 2, 0.25], shape: [2, 4] },
      });
      const e = new Embedding(2, 4, { device: 'cpu' });
      const upload = vi.fn();
      await loadLazyStateDict(e, sd, { upload });
      expect(upload).not.toHaveBeenCalled();
      expect(e.gather([1, 0])).toEqual(new Float32Array([10, 12, -1.75, 1.75, 0.5, -1, 3, -4]));
    });
  });
});

describe('MultiHeadAttention', () => {
  const opts = { qHeads: 4, kvHeads: 2, headDim: 8, windowLeft: Infinity, windowRight: 0 };

  it('has no parameters without sinks', () => {
    expect(shapes(new MultiHeadAttention(opts))).toEqual({});
  });

  it('declares one f32 sink per query head, pinned', () => {
    const attn = new MultiHeadAttention({ ...opts, sinks: true });
    expect(shapes(attn)).toEqual({ sinks: meta([4]) });
    expect(attn.sinks!.pinned).toBe(true);
  });

  it('attends from q, k, v or from one packed qkv', () => {
    const attn = fakeLoad(new MultiHeadAttention({ ...opts, sinks: true }));
    const kv = input([5, 16]);
    expect(attn.forward(input([5, 32]), kv, kv).shape).toEqual(meta([5, 32]));
    expect(attn.forwardPacked(input([5, 64])).shape).toEqual(meta([5, 32]));
  });

  // attrs[10] is the score scale.
  it.each([
    { name: '1/√headDim by default', scale: undefined, want: 1 / Math.sqrt(8) },
    { name: 'the scale it was built with', scale: 1, want: 1 },
  ])('scales the scores by $name', ({ scale, want }) => {
    const attn = new MultiHeadAttention({ ...opts, scale });
    const kv = input([5, 16]);
    expect(attn.forward(input([5, 32]), kv, kv).pending?.attrs?.[10]).toBe(want);
    expect(attn.forwardPacked(input([5, 64])).pending?.attrs?.[10]).toBe(want);
  });
});
