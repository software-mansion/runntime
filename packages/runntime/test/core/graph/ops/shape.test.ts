import { describe, expect, it } from 'vitest';
import {
  astype,
  cat,
  chunk,
  gatherRows,
  gatherRowsFrom,
  reshape,
  slice,
  split,
  transpose,
  writeRows,
} from '../../../../src/core/graph/ops.ts';
import { hwc4Input, input, materializedInput, meta, quantWeight } from '../values.ts';

describe('reshape', () => {
  it.each([{ to: [3, 2] }, { to: [6] }, { to: [1, 6, 1] }])('[2, 3] to $to', ({ to }) => {
    expect(reshape(input([2, 3], 'f16'), to).shape).toEqual(meta(to, 'f16'));
  });

  it('rejects a different element count', () => {
    expect(() => reshape(input([2, 3]), [4, 2])).toThrow(
      /mismatch inputs has - 6, newDims imply: 8/,
    );
    expect(() => {});
  });

  it('rejects hwc4 storage', () => {
    expect(() => reshape(hwc4Input(4, 2, 2), [16])).toThrow(/convert with toChw\(\) first/);
  });
});

describe('astype', () => {
  it('converts f32 to f16 and keeps the dims', () => {
    expect(astype(input([2, 3]), 'f16').shape).toEqual(meta([2, 3], 'f16'));
  });

  it('returns the same value when the dtype already matches', () => {
    const x = input([2, 3]);
    expect(astype(x, 'f32')).toBe(x);
  });

  it.each([
    { name: 'a quantized weight', x: quantWeight(4, 8, 8), error: /needs a float input/ },
    { name: 'hwc4 storage', x: hwc4Input(4, 2, 2), error: /stored 'hwc4'/ },
  ])('rejects $name', ({ x, error }) => {
    expect(() => astype(x, 'f32')).toThrow(error);
  });
});

describe('transpose', () => {
  it('swaps the two dims', () => {
    expect(transpose(input([2, 5])).shape).toEqual(meta([5, 2]));
  });

  it('rejects anything but 2D', () => {
    expect(() => transpose(input([2, 5, 1]))).toThrow(/expected a 2D/);
  });
});

describe('cat', () => {
  it.each([
    { name: 'rows (dim 0)', a: [2, 3], b: [4, 3], dim: 0, out: [6, 3], op: 'concatRows' },
    { name: 'cols (dim 1)', a: [2, 3], b: [2, 5], dim: 1, out: [2, 8], op: 'concatCols' },
    { name: 'cols (dim -1)', a: [2, 3], b: [2, 5], dim: -1, out: [2, 8], op: 'concatCols' },
    {
      name: 'channels (3D, dim 0)',
      a: [2, 4, 4],
      b: [3, 4, 4],
      dim: 0,
      out: [5, 4, 4],
      op: 'concatChannels',
    },
  ])('joins $name', ({ a, b, dim, out, op }) => {
    const c = cat([input(a), input(b)], dim);
    expect(c.shape).toEqual(meta(out));
    expect(c.pending?.op).toBe(op);
  });

  it('folds three tensors left to right', () => {
    expect(cat([input([1, 2]), input([2, 2]), input([3, 2])], 0).shape).toEqual(meta([6, 2]));
  });

  it('returns a single tensor as it is', () => {
    const x = input([2, 3]);
    expect(cat([x], 0)).toBe(x);
  });

  it('joins hwc4 maps whose channel counts are multiples of 4', () => {
    expect(cat([hwc4Input(4, 2, 2), hwc4Input(8, 2, 2)], 0).shape).toEqual(
      hwc4Input(12, 2, 2).shape,
    );
  });

  it.each([
    { name: 'an empty list', xs: [], dim: 0, error: /empty tensor list/ },
    { name: '1D tensors', xs: [input([3]), input([3])], dim: 0, error: /needs 2D or 3D/ },
    {
      name: 'mismatched cols on a row join',
      xs: [input([2, 3]), input([2, 4])],
      dim: 0,
      error: /col mismatch — a has 3, b has 4/,
    },
    {
      name: 'mismatched rows on a col join',
      xs: [input([2, 3]), input([3, 3])],
      dim: 1,
      error: /row mismatch — a has 2, b has 3/,
    },
    {
      name: 'mixed dtypes',
      xs: [input([2, 3]), input([2, 3], 'f16')],
      dim: 0,
      error: /dtype mismatch/,
    },
    {
      name: 'a 3D join along dim 1',
      xs: [input([2, 4, 4]), input([2, 4, 4])],
      dim: 1,
      error: /only the channel dim/,
    },
    {
      name: 'mismatched spatial sizes',
      xs: [input([2, 4, 4]), input([2, 4, 5])],
      dim: 0,
      error: /spatial mismatch/,
    },
    {
      name: 'a dim out of range',
      xs: [input([2, 3]), input([2, 3])],
      dim: 2,
      error: /dim 2 out of range/,
    },
    {
      name: 'hwc4 channel counts not multiples of 4',
      xs: [hwc4Input(4, 2, 2), hwc4Input(5, 2, 2)],
      dim: 0,
      error: /multiples of 4, got 4 \+\+ 5/,
    },
  ])('rejects $name', ({ xs, dim, error }) => {
    expect(() => cat(xs, dim)).toThrow(error);
  });
});

describe('slice', () => {
  it.each([
    { name: 'rows', x: [5, 3], dim: 0, out: [3, 3], op: 'sliceRows' },
    { name: 'cols', x: [3, 5], dim: 1, out: [3, 3], op: 'sliceCols' },
    { name: 'cols by dim -1', x: [3, 5], dim: -1, out: [3, 3], op: 'sliceCols' },
    { name: 'channels', x: [5, 2, 2], dim: 0, out: [3, 2, 2], op: 'sliceChannels' },
  ])('cuts [1, 4) from $name', ({ x, dim, out, op }) => {
    const c = slice(input(x), dim, 1, 4);
    expect(c.shape).toEqual(meta(out));
    expect(c.pending).toMatchObject({ op, attrs: [1, 4] });
  });

  it('takes a single row', () => {
    expect(slice(input([5, 3]), 0, 4, 5).shape).toEqual(meta([1, 3]));
  });

  it.each([
    { name: 'an empty range', start: 2, end: 2 },
    { name: 'a backwards range', start: 3, end: 1 },
    { name: 'a range past the end', start: 2, end: 6 },
    { name: 'a negative start', start: -1, end: 2 },
    { name: 'a fractional bound', start: 0.5, end: 2 },
  ])('rejects $name', ({ start, end }) => {
    expect(() => slice(input([5, 3]), 0, start, end)).toThrow(/bad range/);
  });

  it.each([
    { name: 'a 3D slice along dim 1', x: input([4, 2, 2]), dim: 1, error: /only the channel dim/ },
    { name: 'a 1D input', x: input([4]), dim: 0, error: /needs a 2D or 3D/ },
    { name: 'a dim out of range', x: input([4, 2]), dim: -3, error: /dim -3 out of range/ },
    { name: 'a fractional dim', x: input([4, 2]), dim: 0.5, error: /dim must be an integer/ },
  ])('rejects $name', ({ x, dim, error }) => {
    expect(() => slice(x, dim, 0, 1)).toThrow(error);
  });

  it('slices hwc4 channels only at multiples of 4', () => {
    expect(slice(hwc4Input(12, 2, 2), 0, 4, 12).shape).toEqual(hwc4Input(8, 2, 2).shape);
    expect(() => slice(hwc4Input(12, 2, 2), 0, 2, 6)).toThrow(/must align to 4/);
  });
});

describe('chunk', () => {
  it('splits into equal parts', () => {
    const parts = chunk(input([4, 6]), 3, 1);
    expect(parts.map((p) => p.shape)).toEqual([meta([4, 2]), meta([4, 2]), meta([4, 2])]);
    expect(parts.map((p) => p.pending?.attrs)).toEqual([
      [0, 2],
      [2, 4],
      [4, 6],
    ]);
  });

  it('gives the whole range for one part', () => {
    expect(chunk(input([4, 6]), 1, 0).map((p) => p.shape)).toEqual([meta([4, 6])]);
  });

  it.each([
    { name: 'a size that does not divide', n: 4, error: /size 6 does not divide into 4/ },
    { name: 'zero parts', n: 0, error: /n must be a positive integer/ },
  ])('rejects $name', ({ n, error }) => {
    expect(() => chunk(input([4, 6]), n, 1)).toThrow(error);
  });
});

describe('split', () => {
  it('splits into the given sizes', () => {
    const parts = split(input([4, 6]), [1, 2, 3], -1);
    expect(parts.map((p) => p.shape.dims)).toEqual([
      [4, 1],
      [4, 2],
      [4, 3],
    ]);
    expect(parts.map((p) => p.pending?.attrs)).toEqual([
      [0, 1],
      [1, 3],
      [3, 6],
    ]);
  });

  it('rejects sizes that do not cover the dim', () => {
    expect(() => split(input([4, 6]), [2, 2], 1)).toThrow(/sum to 4, but dim 1 has 6/);
  });

  it('rejects a zero-size part', () => {
    expect(() => split(input([4, 6]), [0, 6], 1)).toThrow(/bad range \[0, 0\)/);
  });
});

describe('writeRows', () => {
  // A KV cache with room for 8 rows of 4 values
  const cache = materializedInput([8, 4]);

  it('returns a view up to the last written row', () => {
    const c = writeRows(cache, input([2, 4]), 3);
    expect(c.shape).toEqual(meta([5, 4]));
    expect(c.pending?.attrs).toEqual([3]);
  });

  it('fills the cache to the last row', () => {
    expect(writeRows(cache, input([2, 4]), 6).shape).toEqual(meta([8, 4]));
  });

  it.each([
    {
      name: 'a pending destination',
      dst: input([8, 4]),
      src: input([2, 4]),
      start: 0,
      error: /dst must be a materialized/,
    },
    {
      name: 'rows past the capacity',
      dst: cache,
      src: input([2, 4]),
      start: 7,
      error: /rows \[7, 9\) exceed dst capacity 8/,
    },
    {
      name: 'a negative start',
      dst: cache,
      src: input([2, 4]),
      start: -1,
      error: /exceed dst capacity/,
    },
    { name: 'mismatched cols', dst: cache, src: input([2, 5]), start: 0, error: /col mismatch/ },
    {
      name: 'mixed dtypes',
      dst: cache,
      src: input([2, 4], 'f16'),
      start: 0,
      error: /dtype mismatch/,
    },
  ])('rejects $name', ({ dst, src, start, error }) => {
    expect(() => writeRows(dst, src, start)).toThrow(error);
  });
});

describe('gatherRows', () => {
  it('picks rows, repeats allowed', () => {
    const c = gatherRows(input([5, 3]), [4, 0, 4]);
    expect(c.shape).toEqual(meta([3, 3]));
    expect(c.pending?.attrs).toEqual([4, 0, 4]);
  });

  it('keeps its own copy of the indices', () => {
    const idx = [1, 2];
    const c = gatherRows(input([5, 3]), idx);
    idx[0] = 99;
    expect(c.pending?.attrs).toEqual([1, 2]);
  });

  it.each([
    { name: 'no indices', idx: [], error: /empty index list/ },
    { name: 'an index past the end', idx: [5], error: /index 5 out of range for 5 rows/ },
    { name: 'a negative index', idx: [-1], error: /index -1 out of range/ },
    { name: 'a fractional index', idx: [1.5], error: /index 1.5 out of range/ },
  ])('rejects $name', ({ idx, error }) => {
    expect(() => gatherRows(input([5, 3]), idx)).toThrow(error);
  });
});

describe('gatherRowsFrom', () => {
  it('gives one row per index', () => {
    expect(gatherRowsFrom(input([5, 3], 'f16'), input([7])).shape).toEqual(meta([7, 3], 'f16'));
  });

  it('needs f32 indices', () => {
    expect(() => gatherRowsFrom(input([5, 3]), input([7], 'f16'))).toThrow(/idx must be f32/);
  });
});
