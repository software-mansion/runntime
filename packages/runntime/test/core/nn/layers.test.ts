import { describe, expect, it, vi } from 'vitest';
import { pending, type ValueMeta } from '../../../src/core/graph/value.ts';
import {
  BatchNorm2d,
  Conv1d,
  Conv2d,
  ConvTranspose1d,
  ConvTranspose2d,
  Embedding,
  GroupNorm,
  LayerNorm,
  Linear,
  MultiHeadAttention,
  RMSNorm,
  Snake,
} from '../../../src/core/nn/layers.ts';
import { loadLazyStateDict } from '../../../src/core/nn/loadStateDict.ts';
import { CpuParameter, Module, type AnyModule } from '../../../src/core/nn/module.ts';
import { f16ToF32, f32ArrayToF16Bits } from '../../../src/core/weights/convert.ts';
import {
  bufferSource,
  derivedTensor,
  fromSafetensors,
  memoryStateDict,
  type LazyStateDict,
} from '../../../src/core/weights/safetensors.ts';
import { scaleWeight } from '../../../src/core/weights/transforms.ts';
import { hwc4Input, input, meta } from '../graph/values.ts';
import { contents, quantFile, safetensorsFile, type QuantEntry } from '../weights/checkpoint.ts';

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
  const tensor = derivedTensor(shape, () => Promise.resolve(data), data.byteLength);
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
    const float = derivedTensor([4], () => Promise.resolve(new Float32Array(4)));
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

  it.each([
    { name: 'in features off the 8-row tiles', inF: 12, outF: 4, bits: 8 as const, groupSize: 4 },
    { name: 'int4 out features off 8 per word', inF: 8, outF: 4, bits: 4 as const, groupSize: 4 },
    { name: 'a group size off 4', inF: 16, outF: 4, bits: 8 as const, groupSize: 2 },
  ])('refuses $name with CHECKPOINT_MISMATCH', async ({ inF, outF, bits, groupSize }) => {
    const sd = await quantCheckpoint(bits, groupSize, {
      weight: {
        ints: new Array(inF * outF).fill(0),
        scales: new Array((inF * outF) / groupSize).fill(1),
        shape: [outF, inF],
      },
    });
    const layer = new Linear(inF, outF);
    expect(() => layer.transformStateDict(sd, '')).toThrow(
      expect.objectContaining({ code: 'CHECKPOINT_MISMATCH' }),
    );
    expect(layer.weight.shape.dtype).toBe('f32');
  });

  it('refuses a weight in [in, out] layout', async () => {
    const sd = await quantCheckpoint(8, 4, { weight: { ints: numbered, scales, shape: [8, 4] } });
    expect(() => new Linear(8, 4).transformStateDict(sd, '')).toThrow(
      expect.objectContaining({ code: 'CHECKPOINT_MISMATCH' }),
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
  it('declares an f16 hwc4 weight that half() and float() leave alone', () => {
    const conv = new Conv2d(8, 16, { kernelSize: 3 });
    expect(conv.weight.shape).toEqual({
      elems: 16 * 72,
      dtype: 'f16',
      dims: [16, 8 * 3 * 3],
      layout: 'hwc4',
    });
    expect(conv.weight.pinned).toBe(true);
    expect(conv.bias!.shape).toEqual(meta([16], 'f16'));
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

  it('refuses f32 weights at load', () => {
    const conv = new Conv2d(2, 2, { kernelSize: 1 });
    expect(() => conv.weight.preprocess!(new Float32Array(4))).toThrow(/must arrive as f16 bits/);
  });

  it('maps an hwc4 map to an hwc4 map', () => {
    const conv = fakeLoad(new Conv2d(8, 16, { kernelSize: 3, stride: 2, padding: 1 }));
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
  // The folded weight, packed by Conv2d into one [in][out] hwc4 tile, as f16
  // bits: 2 → 0x4000, 4 → 0x4400, 6 → 0x4600, 8 → 0x4800, 5 → 0x4500.
  const FOLDED_WEIGHT = new Uint16Array(16);
  FOLDED_WEIGHT.set([0x4000, 0x4600, 0, 0, 0x4400, 0x4800]);
  const FOLDED_BIAS = new Uint16Array([0, 0x4500]);

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

  it('GroupNorm has a weight and bias per channel, and needs one group', () => {
    expect(shapes(new GroupNorm(1, 8))).toEqual({ weight: meta([8]), bias: meta([8]) });
    expect(() => new GroupNorm(2, 8)).toThrow(/only numGroups = 1 is supported, got 2/);
  });

  it('RMSNorm has one scale per feature', () => {
    expect(shapes(new RMSNorm(8))).toEqual({ scale: meta([8]) });
  });

  it.each([
    ['LayerNorm', () => new LayerNorm(8, 1e-5, { bias: true })],
    ['GroupNorm', () => new GroupNorm(1, 8)],
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
});
