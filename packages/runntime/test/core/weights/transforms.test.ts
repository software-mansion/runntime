import { describe, expect, it, vi } from 'vitest';
import { f16ToF32, f32ToF16Bits } from '../../../src/core/weights/convert.ts';
import {
  bufferSource,
  derivedTensor,
  fromSafetensors,
  memoryStateDict,
  type LazyTensor,
} from '../../../src/core/weights/safetensors.ts';
import {
  concatRows,
  scaleWeight,
  splitCols,
  splitRows,
  transposeWeight,
} from '../../../src/core/weights/transforms.ts';

type Dtype = 'F32' | 'F16' | 'BF16';

/** A one-tensor safetensors file, built from the format spec: an 8-byte
 *  little-endian header length, the JSON header, then the payload. */
function f16File(values: number[], shape: number[]): Uint8Array {
  const payload = new Uint8Array(Uint16Array.from(values, f32ToF16Bits).buffer);
  const header = new TextEncoder().encode(
    JSON.stringify({ x: { dtype: 'F16', shape, data_offsets: [0, payload.length] } }),
  );
  const out = new Uint8Array(8 + header.length + payload.length);
  new DataView(out.buffer).setBigUint64(0, BigInt(header.length), true);
  out.set(header, 8);
  out.set(payload, 8 + header.length);
  return out;
}

/** Values as bf16 bits. Every value used here is exact in bf16 and f16. */
const bf16 = (values: number[]) =>
  Uint16Array.from(values, (v) => new Uint32Array(new Float32Array([v]).buffer)[0]! >>> 16);

/** A file-like tensor of `dtype`: F16 from a real safetensors file, F32 and
 *  BF16 from memoryStateDict. */
async function source(dtype: Dtype, values: number[], shape: number[]): Promise<LazyTensor> {
  const sd =
    dtype === 'F16'
      ? await fromSafetensors(bufferSource(f16File(values, shape)), { chunkBytes: 0 })
      : memoryStateDict({
          x: { data: dtype === 'F32' ? new Float32Array(values) : bf16(values), shape },
        });
  return sd.tensors.get('x')!;
}

/** What a helper's output holds: f16 halfwords or f32, decoded to numbers. */
async function contents(t: LazyTensor): Promise<{ type: 'f16' | 'f32'; values: number[] }> {
  if (t.kind !== 'derived') throw new Error('expected a derived tensor');
  const data = await t.data();
  if (data instanceof Uint16Array) return { type: 'f16', values: [...f16ToF32(data)] };
  if (data instanceof Float32Array) return { type: 'f32', values: [...data] };
  throw new Error('expected floats');
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

describe('weight helpers', () => {
  it('fetch nothing until an output is read, and split sources read once', async () => {
    const compute = vi.fn(async () => new Float32Array([1, 2, 3, 4]));
    const t = derivedTensor([2, 2], compute);
    const [top, bottom] = splitRows(t, [1, 1]);
    const [left, right] = splitCols(t, [1, 1]);
    concatRows([t, t]);
    scaleWeight(t, 2);
    transposeWeight(t);
    expect(compute).not.toHaveBeenCalled();
    await Promise.all([top!, bottom!, left!, right!].map(contents));
    expect(compute).toHaveBeenCalledOnce();
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
    { name: 'scaleWeight', run: (t: LazyTensor) => scaleWeight(t, 2) },
  ])('$name rejects packed u32 words with CHECKPOINT_MISMATCH', async ({ name, run }) => {
    const words = derivedTensor([1], async () => new Uint32Array([1]));
    await expect(contents(run(words))).rejects.toMatchObject({
      code: 'CHECKPOINT_MISMATCH',
      message: `${name}: source holds packed u32 words`,
    });
  });
});
