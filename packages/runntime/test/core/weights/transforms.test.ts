import { describe, expect, it, vi } from 'vitest';
import {
  bufferSource,
  derivedTensor,
  fromSafetensors,
  memoryStateDict,
  quantTensor,
  type LazyTensor,
} from '../../../src/core/weights/safetensors.ts';
import {
  concatRows,
  permuteRows,
  readFloats,
  reshapeWeight,
  scaleWeight,
  splitCols,
  splitRows,
  transposeWeight,
} from '../../../src/core/weights/transforms.ts';
import { contents, f16File, quantContents, quantSource, type QuantEntry } from './checkpoint.ts';

type Dtype = 'F32' | 'F16' | 'BF16';

/** Values as bf16 bits. Every value used here is exact in bf16 and f16. */
const bf16 = (values: number[]) =>
  Uint16Array.from(values, (v) => new Uint32Array(new Float32Array([v]).buffer)[0]! >>> 16);

/** A file-like tensor of `dtype`: F16 from a real safetensors file, F32 and
 *  BF16 from memoryStateDict. */
async function source(dtype: Dtype, values: number[], shape: number[]): Promise<LazyTensor> {
  const sd =
    dtype === 'F16'
      ? await fromSafetensors(bufferSource(f16File({ x: { values, shape } })), { chunkBytes: 0 })
      : memoryStateDict({
          x: { data: dtype === 'F32' ? new Float32Array(values) : bf16(values), shape },
        });
  return sd.tensors.get('x')!;
}

/** F16 stays f16; F32 stays f32; BF16 decodes to f32, which is exact. */
const DTYPES: { dtype: Dtype; type: 'f16' | 'f32' }[] = [
  { dtype: 'F32', type: 'f32' },
  { dtype: 'F16', type: 'f16' },
  { dtype: 'BF16', type: 'f32' },
];

describe('concatRows', () => {
  it.for(DTYPES)('stacks $dtype parts along dim 0', async ({ dtype, type }) => {
    const a = await source(dtype, [1, 2], [1, 2]);
    const b = await source(dtype, [3, 4, 5, 6], [2, 2]);
    const out = concatRows([a, b]);
    expect(out.shape).toEqual([3, 2]);
    expect(out.byteLength).toBe(a.byteLength + b.byteLength);
    expect(await contents(out)).toEqual({ type, values: [1, 2, 3, 4, 5, 6] });
  });

  it('reads a part listed several times once', async () => {
    const compute = vi.fn(async () => new Float32Array([1, 2]));
    const norm = derivedTensor([2], compute);
    const out = concatRows([norm, norm, norm]);
    expect(await contents(out)).toEqual({ type: 'f32', values: [1, 2, 1, 2, 1, 2] });
    expect(compute).toHaveBeenCalledOnce();
  });

  it('gives f32 when f16 and f32 parts mix', async () => {
    const out = concatRows([
      await source('F16', [1, 2], [1, 2]),
      await source('F32', [3, 4], [1, 2]),
    ]);
    expect(await contents(out)).toEqual({ type: 'f32', values: [1, 2, 3, 4] });
  });

  it.for([
    { name: 'no parts', shapes: [] },
    {
      name: 'different widths',
      shapes: [
        [1, 2],
        [1, 3],
      ],
    },
    { name: 'different ranks', shapes: [[2, 2], [2]] },
    { name: 'a scalar', shapes: [[]] },
  ])('throws CHECKPOINT_MISMATCH for $name', async ({ shapes }) => {
    const parts = await Promise.all(
      shapes.map((shape) =>
        source('F32', new Array(shape.reduce((a, b) => a * b, 1)).fill(0), shape),
      ),
    );
    expect(() => concatRows(parts)).toThrow(
      expect.objectContaining({
        code: 'CHECKPOINT_MISMATCH',
        message: expect.stringMatching(/^concatRows: cannot stack/),
      }),
    );
  });
});

describe('splitRows', () => {
  it.for(DTYPES)('splits $dtype along dim 0', async ({ dtype, type }) => {
    const t = await source(dtype, [1, 2, 3, 4, 5, 6], [3, 2]);
    const [top, bottom] = splitRows(t, [1, 2]);
    expect([top!.shape, bottom!.shape]).toEqual([
      [1, 2],
      [2, 2],
    ]);
    expect(top!.byteLength + bottom!.byteLength).toBe(t.byteLength);
    expect(await contents(top!)).toEqual({ type, values: [1, 2] });
    expect(await contents(bottom!)).toEqual({ type, values: [3, 4, 5, 6] });
  });

  it('splits dim 0 of a 3D tensor, keeping the rest', async () => {
    // [2, 1, 2]: one [1, 1, 2] block per index of dim 0.
    const [first, second] = splitRows(await source('F32', [1, 2, 3, 4], [2, 1, 2]), [1, 1]);
    expect(first!.shape).toEqual([1, 1, 2]);
    expect(await contents(second!)).toEqual({ type: 'f32', values: [3, 4] });
  });

  it('throws CHECKPOINT_MISMATCH when the sizes do not cover dim 0', async () => {
    const t = await source('F32', [1, 2, 3, 4, 5, 6], [3, 2]);
    expect(() => splitRows(t, [1, 1])).toThrow(
      expect.objectContaining({
        code: 'CHECKPOINT_MISMATCH',
        message: expect.stringMatching(/sum to 2, not dim 0/),
      }),
    );
  });
});

describe('splitCols', () => {
  // [2, 1, 4], like an expert tensor split into gate and up halves:
  //   row 0: 1 2 3 4    row 1: 5 6 7 8
  it.for(DTYPES)('splits $dtype along the last dim', async ({ dtype, type }) => {
    const t = await source(dtype, [1, 2, 3, 4, 5, 6, 7, 8], [2, 1, 4]);
    const [left, right] = splitCols(t, [1, 3]);
    expect([left!.shape, right!.shape]).toEqual([
      [2, 1, 1],
      [2, 1, 3],
    ]);
    expect(left!.byteLength + right!.byteLength).toBe(t.byteLength);
    expect(await contents(left!)).toEqual({ type, values: [1, 5] });
    expect(await contents(right!)).toEqual({ type, values: [2, 3, 4, 6, 7, 8] });
  });

  it('throws CHECKPOINT_MISMATCH when the sizes do not cover the last dim', async () => {
    const t = await source('F32', [1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);
    expect(() => splitCols(t, [2])).toThrow(
      expect.objectContaining({
        code: 'CHECKPOINT_MISMATCH',
        message: expect.stringMatching(/sum to 2, not dim 1/),
      }),
    );
  });
});

describe('permuteRows', () => {
  // [[1, 2], [3, 4], [5, 6]] with order [2, 0, 1] → [[5, 6], [1, 2], [3, 4]]
  it.for(DTYPES)('reorders the rows of $dtype', async ({ dtype, type }) => {
    const t = await source(dtype, [1, 2, 3, 4, 5, 6], [3, 2]);
    const out = permuteRows(t, [2, 0, 1]);
    expect(out.shape).toEqual([3, 2]);
    expect(out.byteLength).toBe(t.byteLength);
    expect(await contents(out)).toEqual({ type, values: [5, 6, 1, 2, 3, 4] });
  });

  it('moves whole blocks of a 3D tensor and single elements of a 1D one', async () => {
    const blocks = permuteRows(await source('F32', [1, 2, 3, 4], [2, 1, 2]), [1, 0]);
    expect(blocks.shape).toEqual([2, 1, 2]);
    expect(await contents(blocks)).toEqual({ type: 'f32', values: [3, 4, 1, 2] });
    const elems = permuteRows(await source('F16', [1, 2, 3, 4], [4]), [0, 2, 1, 3]);
    expect(await contents(elems)).toEqual({ type: 'f16', values: [1, 3, 2, 4] });
  });

  it.for([
    { name: 'too few rows', order: [0, 1] },
    { name: 'too many rows', order: [0, 1, 2, 0] },
    { name: 'a repeated row', order: [0, 0, 1] },
    { name: 'a row past the end', order: [0, 1, 3] },
    { name: 'a negative row', order: [0, 1, -1] },
    { name: 'a fractional row', order: [0, 1, 1.5] },
  ])('throws CHECKPOINT_MISMATCH for $name', async ({ order }) => {
    const t = await source('F32', [1, 2, 3, 4, 5, 6], [3, 2]);
    expect(() => permuteRows(t, order)).toThrow(
      expect.objectContaining({
        code: 'CHECKPOINT_MISMATCH',
        message: 'permuteRows: order is not a permutation of the 3 rows of shape [3, 2]',
      }),
    );
  });

  it('throws CHECKPOINT_MISMATCH for a scalar', async () => {
    const t = await source('F32', [1], []);
    expect(() => permuteRows(t, [])).toThrow(
      expect.objectContaining({
        code: 'CHECKPOINT_MISMATCH',
        message: 'permuteRows: cannot permute a scalar',
      }),
    );
  });
});

describe('scaleWeight', () => {
  // 3 · 0.1 = 0.3, which f16 would round to 0.300048828125. The product is
  // f32 for every source, so an f32 model loads what it did before the helper.
  it.for([
    { dtype: 'F32', scale: 0.5, values: [1, 2, 3, 4], want: [0.5, 1, 1.5, 2] },
    { dtype: 'F16', scale: 0.5, values: [1, 2, 3, 4], want: [0.5, 1, 1.5, 2] },
    { dtype: 'BF16', scale: 0.5, values: [1, 2, 3, 4], want: [0.5, 1, 1.5, 2] },
    { dtype: 'F32', scale: [2, -1], values: [1, 2, 3, 4], want: [2, 4, -3, -4] },
    { dtype: 'F16', scale: [2, -1], values: [1, 2, 3, 4], want: [2, 4, -3, -4] },
    { dtype: 'F32', scale: 0.1, values: [3, 0, 0, 0], want: [Math.fround(0.3), 0, 0, 0] },
    { dtype: 'F16', scale: 0.1, values: [3, 0, 0, 0], want: [Math.fround(0.3), 0, 0, 0] },
  ] satisfies { dtype: Dtype; scale: number | number[]; values: number[]; want: number[] }[])(
    'scales $dtype by $scale',
    async ({ dtype, scale, values, want }) => {
      const t = await source(dtype, values, [2, 2]);
      const out = scaleWeight(t, scale);
      expect(out.shape).toEqual([2, 2]);
      expect(out.byteLength).toBe(t.byteLength);
      expect(await contents(out)).toEqual({ type: 'f32', values: want });
    },
  );

  it('throws CHECKPOINT_MISMATCH for one row scale too many', async () => {
    const t = await source('F32', [1, 2, 3, 4], [2, 2]);
    expect(() => scaleWeight(t, [1, 2, 3])).toThrow(
      expect.objectContaining({
        code: 'CHECKPOINT_MISMATCH',
        message: expect.stringMatching(/3 row scales/),
      }),
    );
  });
});

describe('transposeWeight', () => {
  // [[1, 2, 3], [4, 5, 6]] → [[1, 4], [2, 5], [3, 6]]
  it.for(DTYPES)('transposes a 2D $dtype tensor', async ({ dtype, type }) => {
    const t = await source(dtype, [1, 2, 3, 4, 5, 6], [2, 3]);
    const out = transposeWeight(t);
    expect(out.shape).toEqual([3, 2]);
    expect(out.byteLength).toBe(t.byteLength);
    expect(await contents(out)).toEqual({ type, values: [1, 4, 2, 5, 3, 6] });
  });

  it('throws CHECKPOINT_MISMATCH for a 3D tensor', async () => {
    const t = await source('F32', new Array(8).fill(0), [2, 2, 2]);
    expect(() => transposeWeight(t)).toThrow(
      expect.objectContaining({
        code: 'CHECKPOINT_MISMATCH',
        message: expect.stringMatching(/needs a 2D tensor/),
      }),
    );
  });
});

describe('reshapeWeight', () => {
  // [2, 1, 3] → [2, 3]: an expert tensor with its expert axis folded into rows.
  it.for(DTYPES)('reshapes $dtype, keeping the source dtype', async ({ dtype }) => {
    const t = await source(dtype, [1, 2, 3, 4, 5, 6], [2, 1, 3]);
    const out = reshapeWeight(t, [2, 3]);
    expect(out.shape).toEqual([2, 3]);
    expect(out.byteLength).toBe(t.byteLength);
    if (t.kind === 'quant' || out.kind === 'quant') throw new Error('expected float tensors');
    expect([out.kind, out.dtype]).toEqual([t.kind, t.dtype]);
    expect([...(await out.f32())]).toEqual([1, 2, 3, 4, 5, 6]);
    if (dtype !== 'F32') expect(await out.u16()).toEqual(await t.u16());
  });

  it('keeps the f16 halfwords of a derived source', async () => {
    const [left] = splitCols(await source('F16', [1, 2, 3, 4, 5, 6, 7, 8], [2, 1, 4]), [2, 2]);
    const out = reshapeWeight(left!, [2, 2]);
    expect(out.shape).toEqual([2, 2]);
    expect(await contents(out)).toEqual({ type: 'f16', values: [1, 2, 5, 6] });
  });

  it('throws CHECKPOINT_MISMATCH when the element counts differ', async () => {
    const t = await source('F32', [1, 2, 3, 4, 5, 6], [2, 3]);
    expect(() => reshapeWeight(t, [4, 2])).toThrow(
      expect.objectContaining({
        code: 'CHECKPOINT_MISMATCH',
        message: 'reshapeWeight: cannot reshape [2, 3] to [4, 2]',
      }),
    );
  });
});

describe('weight helpers', () => {
  it('fetch nothing until an output is read, and split sources read once', async () => {
    const compute = vi.fn(async () => new Float32Array([1, 2, 3, 4]));
    const t = derivedTensor([2, 2], compute);
    const [top, bottom] = splitRows(t, [1, 1]);
    const [left, right] = splitCols(t, [1, 1]);
    concatRows([t, t]);
    permuteRows(t, [1, 0]);
    scaleWeight(t, 2);
    transposeWeight(t);
    reshapeWeight(t, [4]);
    expect(compute).not.toHaveBeenCalled();
    await Promise.all([top!, bottom!].map(contents));
    expect(compute).toHaveBeenCalledOnce();
    // Each split reads the source once for all its outputs.
    await Promise.all([left!, right!].map(contents));
    expect(compute).toHaveBeenCalledTimes(2);
  });

  it('chain: f16 halfwords from one helper stay f16 through the next', async () => {
    // q|k fused, then transposed: [[1, 2], [3, 4]] → [[1, 3], [2, 4]]
    const q = await source('F16', [1, 2], [1, 2]);
    const k = await source('F16', [3, 4], [1, 2]);
    const out = transposeWeight(concatRows([q, k]));
    expect(await contents(out)).toEqual({ type: 'f16', values: [1, 3, 2, 4] });
  });

  it('chain: a scaled F16 part makes the fused result f32', async () => {
    // q|k fused with k scaled by 0.5: [[1, 2], [3, 4]] → [[1, 2], [1.5, 2]]
    const q = await source('F16', [1, 2], [1, 2]);
    const k = await source('F16', [3, 4], [1, 2]);
    const out = concatRows([q, scaleWeight(k, 0.5)]);
    expect(await contents(out)).toEqual({ type: 'f32', values: [1, 2, 1.5, 2] });
  });

  it.for([
    { name: 'concatRows', run: (t: LazyTensor) => concatRows([t]) },
    { name: 'splitRows', run: (t: LazyTensor) => splitRows(t, [1])[0]! },
    { name: 'permuteRows', run: (t: LazyTensor) => permuteRows(t, [0]) },
    { name: 'scaleWeight', run: (t: LazyTensor) => scaleWeight(t, 2) },
  ])('$name rejects packed u32 words with CHECKPOINT_MISMATCH', async ({ name, run }) => {
    const words = derivedTensor([1], async () => new Uint32Array([1]));
    await expect(contents(run(words))).rejects.toMatchObject({
      code: 'CHECKPOINT_MISMATCH',
      message: `${name}: source holds packed u32 words`,
    });
  });

  it.for([
    { name: 'concatRows', run: (t: LazyTensor) => concatRows([t]) },
    { name: 'splitRows', run: (t: LazyTensor) => splitRows(t, [2])[0]! },
    { name: 'splitCols', run: (t: LazyTensor) => splitCols(t, [3])[0]! },
    { name: 'scaleWeight', run: (t: LazyTensor) => scaleWeight(t, 2) },
    { name: 'transposeWeight', run: (t: LazyTensor) => transposeWeight(t) },
  ])('$name rejects a derived source whose data does not fill its shape', async ({ name, run }) => {
    // A hook bug: shape [2, 3] says 6 values, data holds 5. A file tensor
    // cannot get here, its header check rejects the length mismatch.
    const short = derivedTensor([2, 3], async () => new Float32Array([1, 2, 3, 4, 5]));
    await expect(contents(run(short))).rejects.toMatchObject({
      code: 'CHECKPOINT_MISMATCH',
      message: `${name}: source holds 5 values for shape [2, 3]`,
    });
  });
});

// Every int fits int4 (-8..7) and every scale is exact in f16, so each case
// runs for both widths and the expected values are the same.
describe.each([8, 4] as const)('int%i quantized sources', (bits) => {
  // [2, 4] in groups of 2:
  //   ints   1 -2 |  3 -4      scales 0.5 1
  //          5 -6 |  7 -8             2   4
  const w: QuantEntry = {
    ints: [1, -2, 3, -4, 5, -6, 7, -8],
    scales: [0.5, 1, 2, 4],
    shape: [2, 4],
  };
  // [1, 4]: ints -1 0 | 6 -7, scales 0.25 8
  const v: QuantEntry = { ints: [-1, 0, 6, -7], scales: [0.25, 8], shape: [1, 4] };
  const mismatch = (message: RegExp) =>
    expect.objectContaining({
      code: 'CHECKPOINT_MISMATCH',
      message: expect.stringMatching(message),
    });

  it('concatRows stacks ints and scales, keeping f16 scales', async () => {
    const a = await quantSource(bits, 2, 'w.weight', w);
    const b = await quantSource(bits, 2, 'v.weight', v);
    const out = concatRows([a, b]);
    expect(out).toMatchObject({ kind: 'quant', shape: [3, 4], bits, groupSize: 2 });
    expect(out.byteLength).toBe(a.byteLength + b.byteLength);
    expect(await quantContents(out)).toEqual({
      ints: [1, -2, 3, -4, 5, -6, 7, -8, -1, 0, 6, -7],
      scaleType: 'f16',
      scales: [0.5, 1, 2, 4, 0.25, 8],
    });
  });

  it('concatRows throws CHECKPOINT_MISMATCH for mixed parts', async () => {
    const a = await quantSource(bits, 2, 'w.weight', w);
    const float = await source('F32', [1, 2, 3, 4], [1, 4]);
    const other = await quantSource(bits, 4, 'v.weight', { ...v, scales: [1] });
    expect(() => concatRows([a, float])).toThrow(mismatch(/mix quantized and float/));
    expect(() => concatRows([a, other])).toThrow(mismatch(/differ in bits or group size/));
  });

  it('splitRows splits ints and scales rows', async () => {
    const t = await quantSource(bits, 2, 'w.weight', w);
    const [top, bottom] = splitRows(t, [1, 1]);
    expect(top).toMatchObject({ kind: 'quant', shape: [1, 4], bits, groupSize: 2 });
    expect(top!.byteLength + bottom!.byteLength).toBe(t.byteLength);
    expect(await quantContents(top!)).toEqual({
      ints: [1, -2, 3, -4],
      scaleType: 'f16',
      scales: [0.5, 1],
    });
    expect(await quantContents(bottom!)).toEqual({
      ints: [5, -6, 7, -8],
      scaleType: 'f16',
      scales: [2, 4],
    });
  });

  it('splitCols splits on the group grid', async () => {
    const t = await quantSource(bits, 2, 'w.weight', w);
    const [left, right] = splitCols(t, [2, 2]);
    expect(left).toMatchObject({ kind: 'quant', shape: [2, 2], bits, groupSize: 2 });
    expect(left!.byteLength + right!.byteLength).toBe(t.byteLength);
    // Columns 0-1 of each row and their group's scale, then columns 2-3.
    expect(await quantContents(left!)).toEqual({
      ints: [1, -2, 5, -6],
      scaleType: 'f16',
      scales: [0.5, 2],
    });
    expect(await quantContents(right!)).toEqual({
      ints: [3, -4, 7, -8],
      scaleType: 'f16',
      scales: [1, 4],
    });
  });

  it('splitCols throws CHECKPOINT_MISMATCH off the group grid', async () => {
    const t = await quantSource(bits, 2, 'w.weight', w);
    expect(() => splitCols(t, [1, 3])).toThrow(
      mismatch(/sizes \[1, 3\] split the groups of 2 columns of w\.weight/),
    );
  });

  it('permuteRows reorders ints and scales rows', async () => {
    const out = permuteRows(await quantSource(bits, 2, 'w.weight', w), [1, 0]);
    expect(out).toMatchObject({ kind: 'quant', shape: [2, 4], bits, groupSize: 2 });
    expect(await quantContents(out)).toEqual({
      ints: [5, -6, 7, -8, 1, -2, 3, -4],
      scaleType: 'f16',
      scales: [2, 4, 0.5, 1],
    });
  });

  it.for([
    // 0.5·3, 1·3, 2·3, 4·3
    { scale: 3, want: [1.5, 3, 6, 12] },
    // row 0 (0.5, 1) by 2, row 1 (2, 4) by -10
    { scale: [2, -10], want: [1, 2, -20, -40] },
  ])('scaleWeight by $scale multiplies the scales into f32', async ({ scale, want }) => {
    const t = await quantSource(bits, 2, 'w.weight', w);
    const out = scaleWeight(t, scale);
    expect(out).toMatchObject({ kind: 'quant', shape: [2, 4], bits, groupSize: 2 });
    expect(out.byteLength).toBe(t.byteLength);
    expect(await quantContents(out)).toEqual({ ints: w.ints, scaleType: 'f32', scales: want });
  });

  it('reshapeWeight keeps the last dim and the groups', async () => {
    const t = await quantSource(bits, 2, 'w.weight', w);
    const out = reshapeWeight(t, [2, 1, 4]);
    expect(out).toMatchObject({ kind: 'quant', shape: [2, 1, 4], bits, groupSize: 2 });
    expect(await quantContents(out)).toEqual(await quantContents(t));
    expect(() => reshapeWeight(t, [4, 2])).toThrow(mismatch(/w\.weight is quantized.*last dim/));
  });

  it('transposeWeight throws CHECKPOINT_MISMATCH', async () => {
    const t = await quantSource(bits, 2, 'w.weight', w);
    expect(() => transposeWeight(t)).toThrow(mismatch(/w\.weight is quantized/));
  });

  it('readFloats and the float accessors of an output reject it', async () => {
    const t = await quantSource(bits, 2, 'w.weight', w);
    await expect(readFloats('test', t)).rejects.toThrow(mismatch(/^test: w\.weight is quantized/));
    await expect(splitRows(t, [1, 1])[0]!.f32()).rejects.toThrow(
      mismatch(/splitRows\(w\.weight\) is quantized/),
    );
  });
});

describe('quantized helpers', () => {
  /** A [2, 2] int8 tensor in one group per row, counting its reads. */
  function counted() {
    const ints = vi.fn(async () => new Int8Array([1, 2, 3, 4]));
    const scales = vi.fn(async () => new Float32Array([1, 2]));
    const t = quantTensor({
      name: 'q',
      shape: [2, 2],
      bits: 8,
      groupSize: 2,
      byteLength: 6,
      ints,
      scales,
    });
    return { t, ints, scales };
  }

  it('concatRows reads a part listed several times once', async () => {
    const { t, ints, scales } = counted();
    expect(await quantContents(concatRows([t, t]))).toEqual({
      ints: [1, 2, 3, 4, 1, 2, 3, 4],
      scaleType: 'f32',
      scales: [1, 2, 1, 2],
    });
    expect(ints).toHaveBeenCalledOnce();
    expect(scales).toHaveBeenCalledOnce();
  });

  it('splits read the source once for all their outputs', async () => {
    const { t, ints, scales } = counted();
    await Promise.all(splitRows(t, [1, 1]).map(quantContents));
    expect([ints.mock.calls.length, scales.mock.calls.length]).toEqual([1, 1]);
    await Promise.all(splitCols(t, [2]).map(quantContents));
    expect([ints.mock.calls.length, scales.mock.calls.length]).toEqual([2, 2]);
  });
});
