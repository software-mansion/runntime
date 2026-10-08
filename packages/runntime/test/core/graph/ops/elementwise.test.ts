import { describe, expect, it } from 'vitest';
import {
  add,
  asinh,
  clamp,
  gelu,
  mul,
  rsqrt,
  sigmoid,
  silu,
  sub,
  swiglu,
  swigluChunk,
  tanh,
} from '../../../../src/core/graph/ops.ts';
import { hwc4Input, input, meta, quantWeight } from '../values.ts';

// attrs[0] is the broadcast mode the kernel runs.
const ELEMENTWISE = 0;
const ROW = 1;
const COLUMN = 2;
const SCALAR = 3;

describe('add / mul broadcasting', () => {
  it.each([
    { name: 'the same shape', b: [2, 3], mode: ELEMENTWISE },
    { name: 'a row [N]', b: [3], mode: ROW },
    { name: 'a row [1, N]', b: [1, 3], mode: ROW },
    { name: 'a column [M, 1]', b: [2, 1], mode: COLUMN },
    { name: 'a single value', b: [1], mode: SCALAR },
  ])('[2, 3] with $name', ({ b, mode }) => {
    for (const op of [add, mul]) {
      const c = op(input([2, 3]), input(b));
      expect(c.shape).toEqual(meta([2, 3]));
      expect(c.pending?.attrs).toEqual([mode]);
    }
  });

  it('tells a row from a column on a square matrix by dims', () => {
    // [3] and [3, 1] both hold 3 values; only dims say which way they go
    expect(add(input([3, 3]), input([3])).pending?.attrs).toEqual([ROW]);
    expect(add(input([3, 3]), input([3, 1])).pending?.attrs).toEqual([COLUMN]);
  });

  it('puts the full operand first whichever side it is on', () => {
    const full = input([2, 3]);
    const row = input([3]);
    const c = add(row, full);
    expect(c.shape).toEqual(meta([2, 3]));
    expect(c.pending?.inputs).toEqual([full, row]);
  });

  it('keeps f16', () => {
    expect(add(input([2, 3], 'f16'), input([3], 'f16')).shape).toEqual(meta([2, 3], 'f16'));
  });

  it.each([
    {
      name: 'a row of the wrong length',
      a: input([2, 3]),
      b: input([4]),
      error: /incompatible shapes/,
    },
    {
      name: 'mixed dtypes',
      a: input([2, 3]),
      b: input([2, 3], 'f16'),
      error: /incompatible shapes/,
    },
    {
      name: 'a quantized weight',
      a: quantWeight(4, 8, 8),
      b: quantWeight(4, 8, 8),
      error: /weight-only/,
    },
    {
      name: 'hwc4 with row-major',
      a: hwc4Input(4, 2, 2),
      b: input([16], 'f16'),
      error: /incompatible shapes/,
    },
    {
      name: 'hwc4 maps of different channel counts but equal padded size',
      a: hwc4Input(5, 2, 2), // 5 channels pad to 8
      b: hwc4Input(8, 2, 2),
      error: /incompatible shapes/,
    },
  ])('rejects $name', ({ a, b, error }) => {
    expect(() => add(a, b)).toThrow(error);
  });

  it('adds two hwc4 maps of the same shape', () => {
    expect(add(hwc4Input(5, 2, 2), hwc4Input(5, 2, 2)).shape).toEqual(hwc4Input(5, 2, 2).shape);
  });
});

describe('scalar operands', () => {
  it.each([
    { name: 'add', c: add(input([2, 3]), 2), op: 'addScalar', scalar: 2 },
    { name: 'sub (as adding the negative)', c: sub(input([2, 3]), 2), op: 'addScalar', scalar: -2 },
    { name: 'mul', c: mul(input([2, 3]), 0.5), op: 'mulScalar', scalar: 0.5 },
  ])('$name folds the number into the op', ({ c, op, scalar }) => {
    expect(c.shape).toEqual(meta([2, 3]));
    expect(c.pending).toMatchObject({ op, scalar });
  });
});

describe('sub', () => {
  it('broadcasts b as a row or column', () => {
    expect(sub(input([2, 3]), input([3])).pending?.attrs).toEqual([ROW]);
    expect(sub(input([2, 3]), input([2, 1])).pending?.attrs).toEqual([COLUMN]);
  });

  it.each([
    {
      name: 'b larger than a',
      a: input([3]),
      b: input([2, 3]),
      error: /a must be the full operand/,
    },
    {
      name: 'mixed dtypes',
      a: input([2, 3]),
      b: input([3], 'f16'),
      error: /dtype mismatch — f32 − f16/,
    },
  ])('rejects $name', ({ a, b, error }) => {
    expect(() => sub(a, b)).toThrow(error);
  });
});

describe('unary ops', () => {
  it.each([
    ['rsqrt', rsqrt],
    ['sigmoid', sigmoid],
    ['tanh', tanh],
    ['gelu', gelu],
    ['silu', silu],
    ['asinh', asinh],
  ] as const)('%s keeps shape and dtype', (_, op) => {
    expect(op(input([4, 7], 'f16')).shape).toEqual(meta([4, 7], 'f16'));
  });

  // hwc4 pads channels with zeros; an op with f(0) ≠ 0 would fill the padding
  it.each([
    ['tanh', tanh],
    ['gelu', gelu],
    ['silu', silu],
    ['asinh', asinh],
  ] as const)('%s runs on hwc4, since it maps 0 to 0', (_, op) => {
    expect(op(hwc4Input(5, 2, 2)).shape.layout).toBe('hwc4');
  });

  it.each([
    ['rsqrt', rsqrt],
    ['sigmoid', sigmoid],
  ] as const)('%s refuses hwc4, since it does not map 0 to 0', (_, op) => {
    expect(() => op(hwc4Input(5, 2, 2))).toThrow(/refuses hwc4/);
  });

  it('adding a nonzero number refuses hwc4, adding zero does not', () => {
    expect(() => add(hwc4Input(4, 2, 2), 1)).toThrow(/refuses hwc4/);
    expect(add(hwc4Input(4, 2, 2), 0).shape.layout).toBe('hwc4');
    expect(mul(hwc4Input(4, 2, 2), 3).shape.layout).toBe('hwc4');
  });
});

describe('clamp', () => {
  it('keeps the shape and carries the bounds', () => {
    const c = clamp(input([2, 3]), -1, 1);
    expect(c.shape).toEqual(meta([2, 3]));
    expect(c.pending?.attrs).toEqual([-1, 1]);
  });

  it('takes one-sided bounds', () => {
    expect(clamp(input([2]), 0, Infinity).pending?.attrs).toEqual([0, Infinity]);
  });

  it('rejects lo above hi, and NaN', () => {
    expect(() => clamp(input([2]), 1, -1)).toThrow(/lo \(1\) must be <= hi \(-1\)/);
    expect(() => clamp(input([2]), NaN, 1)).toThrow(/must be <= hi/);
  });

  it('runs on hwc4 only when the range contains 0', () => {
    expect(clamp(hwc4Input(4, 2, 2), -1, 1).shape.layout).toBe('hwc4');
    expect(() => clamp(hwc4Input(4, 2, 2), 1, 2)).toThrow(/refuses hwc4/);
    expect(() => clamp(hwc4Input(4, 2, 2), -2, -1)).toThrow(/refuses hwc4/);
  });
});

describe('swiglu', () => {
  it('keeps the shape of matching glu and lin', () => {
    expect(swiglu(input([4, 6]), input([4, 6])).shape).toEqual(meta([4, 6]));
  });

  it.each([
    { name: 'different sizes', lin: input([4, 5]), error: /glu\/lin must match/ },
    { name: 'different dtypes', lin: input([4, 6], 'f16'), error: /glu\/lin must match/ },
  ])('rejects $name', ({ lin, error }) => {
    expect(() => swiglu(input([4, 6]), lin)).toThrow(error);
  });
});

describe('swigluChunk', () => {
  it('halves the columns', () => {
    expect(swigluChunk(input([4, 10], 'f16')).shape).toEqual(meta([4, 5], 'f16'));
  });

  it('rejects an odd column count', () => {
    expect(() => swigluChunk(input([4, 9]))).toThrow(/cols 9 must be even/);
  });
});
