import { describe, expect, it } from 'vitest';
import {
  ACT_CODE,
  argmaxDot,
  matmul,
  matmulGather,
  moeDown,
  moeUp,
} from '../../../../src/core/graph/ops.ts';
import { input, meta, quantWeight } from '../values.ts';

describe('matmul', () => {
  it.each([
    { name: '[2,3] × [3,5]', a: [2, 3], b: [3, 5], out: [2, 5] },
    { name: 'a single row (gemv)', a: [1, 288], b: [288, 864], out: [1, 864] },
    { name: 'a single column out', a: [4, 3], b: [3, 1], out: [4, 1] },
    { name: '1×1', a: [1, 1], b: [1, 1], out: [1, 1] },
  ])('$name gives $out', ({ a, b, out }) => {
    const c = matmul(input(a), input(b));
    expect(c.shape).toEqual(meta(out));
    expect(c.pending?.op).toBe('matmul');
  });

  it('keeps f16', () => {
    expect(matmul(input([2, 3], 'f16'), input([3, 5], 'f16')).shape).toEqual(meta([2, 5], 'f16'));
  });

  it.each([
    {
      name: 'inner dims that disagree',
      a: input([2, 3]),
      b: input([4, 5]),
      error: /inner dims disagree — a=\[2,3\] b=\[4,5\]/,
    },
    { name: 'a 1D input', a: input([6]), b: input([6, 2]), error: /must be 2D/ },
    { name: 'a 3D input', a: input([2, 2, 3]), b: input([3, 5]), error: /must be 2D/ },
    {
      name: 'mixed dtypes',
      a: input([2, 3]),
      b: input([3, 5], 'f16'),
      error: /must be f32 or both f16, got f32 and f16/,
    },
  ])('rejects $name', ({ a, b, error }) => {
    expect(() => matmul(a, b)).toThrow(error);
  });

  describe('epilogue', () => {
    const a = input([2, 3]);
    const b = input([3, 5]);

    it('takes a bias of [N] and an addend of [M, N] as extra inputs', () => {
      const bias = input([5]);
      const addend = input([2, 5]);
      const c = matmul(a, b, { bias, addend, activation: 'gelu' });
      expect(c.shape).toEqual(meta([2, 5]));
      expect(c.pending?.inputs).toEqual([a, b, bias, addend]);
      // attrs: hasBias, hasAddend, activation, baseRow, hasBaseRow
      expect(c.pending?.attrs).toEqual([1, 1, ACT_CODE.gelu, 0, 0]);
    });

    it('takes the tanh-approximated gelu', () => {
      const c = matmul(a, b, { activation: 'geluTanh' });
      expect(c.pending?.attrs).toEqual([0, 0, ACT_CODE.geluTanh, 0, 0]);
    });

    it.each([
      {
        name: 'a bias of the wrong length',
        opts: { bias: input([3]) },
        error: /bias must be f32 \[N\] = 5 elems/,
      },
      { name: 'an f16 bias on f32', opts: { bias: input([5], 'f16') }, error: /bias must be f32/ },
      {
        name: 'an addend of the wrong size',
        opts: { addend: input([5]) },
        error: /addend must be f32 \[M,N\] = 10 elems/,
      },
      {
        name: 'quant options on a float weight',
        opts: { bits: 8 as const },
        error: /quantW weights only/,
      },
    ])('rejects $name', ({ opts, error }) => {
      expect(() => matmul(a, b, opts)).toThrow(error);
    });
  });

  describe('baseRow (a view into a taller weight)', () => {
    const a = input([2, 3]);
    const tall = input([10, 5]); // e.g. 3 experts' K rows fused, plus spare

    it.each([0, 4, 7])('reads K rows from row %i', (baseRow) => {
      expect(matmul(a, tall, { baseRow }).shape).toEqual(meta([2, 5]));
    });

    it.each([
      { name: 'a view running past the end', baseRow: 8 },
      { name: 'a negative row', baseRow: -1 },
      { name: 'a fractional row', baseRow: 1.5 },
    ])('rejects $name', ({ baseRow }) => {
      expect(() => matmul(a, tall, { baseRow })).toThrow(/bad baseRow/);
    });

    it('still needs matching inner dims without baseRow', () => {
      expect(() => matmul(a, tall)).toThrow(/inner dims disagree/);
    });
  });

  describe('quantized weights', () => {
    const a = input([2, 8]);
    // K 8 in groups of 4: two scale rows of 16 columns.
    const q = { scales: input([2, 16]), bits: 8 as const, groupSize: 4 };

    it.each([8, 4] as const)('int%i gives f32 [M, N]', (bits) => {
      const c = matmul(a, quantWeight(8, 16, bits), { ...q, bits });
      expect(c.shape).toEqual(meta([2, 16]));
      expect(c.pending?.op).toBe('matmulQuantW');
      // attrs: baseRow, scaleBase, bits, groupSize, hasBias, hasAdd, act
      expect(c.pending?.attrs).toEqual([0, 0, bits, 4, 0, 0, 0]);
    });

    it('gives f16 [M, N] for f16 activations', () => {
      const c = matmul(input([2, 8], 'f16'), quantWeight(8, 16, 8), q);
      expect(c.shape).toEqual(meta([2, 16], 'f16'));
    });

    it('takes a bias, an addend and an activation on the whole weight', () => {
      const bias = input([16]);
      const addend = input([2, 16]);
      const c = matmul(a, quantWeight(8, 16, 8), { ...q, bias, addend, activation: 'silu' });
      expect(c.pending?.inputs.slice(3)).toEqual([bias, addend]);
      expect(c.pending?.attrs?.slice(4)).toEqual([1, 1, ACT_CODE.silu]);
    });

    it('reads one expert of a fused weight with a K-aligned baseRow', () => {
      expect(matmul(a, quantWeight(24, 16, 8), { ...q, baseRow: 16 }).shape).toEqual(meta([2, 16]));
    });

    it.each([
      {
        name: 'an epilogue on one expert',
        w: quantWeight(24, 16, 8),
        opts: { ...q, baseRow: 16, bias: input([16]) },
        error: /an epilogue on a quantW weight needs the whole weight/,
      },
      {
        name: 'a bias of the wrong width',
        w: quantWeight(8, 16, 8),
        opts: { ...q, bias: input([8]) },
        error: /bias must be f32 \[N\] = 16 elems/,
      },
      {
        name: 'missing scales',
        w: quantWeight(8, 16, 8),
        opts: { bits: 8 as const, groupSize: 4 },
        error: /need scales, bits and groupSize/,
      },
      {
        name: 'int4 cols not a multiple of 8',
        w: quantWeight(8, 4, 4),
        opts: { ...q, bits: 4 as const },
        error: /cols 4 must be divisible by 8/,
      },
      {
        name: 'a group size that does not divide K',
        w: quantWeight(8, 16, 8),
        opts: { ...q, groupSize: 3 },
        error: /groupSize 3 must divide K 8/,
      },
      {
        name: 'a baseRow inside an expert',
        w: quantWeight(24, 16, 8),
        opts: { ...q, baseRow: 4 },
        error: /must be K-aligned/,
      },
      {
        name: 'a baseRow past the end',
        w: quantWeight(16, 16, 8),
        opts: { ...q, baseRow: 16 },
        error: /bad baseRow/,
      },
      {
        name: 'a negative scaleBase',
        w: quantWeight(8, 16, 8),
        opts: { ...q, scaleBase: -1 },
        error: /bad scaleBase/,
      },
    ])('rejects $name', ({ w, opts, error }) => {
      expect(() => matmul(a, w, opts)).toThrow(error);
    });

    it('rejects a weight whose word count does not match its shape', () => {
      const w = quantWeight(8, 16, 4); // packed as int4, read as int8
      expect(() => matmul(a, w, q)).toThrow(/elems 16 != rows\*cols\/4 = 32/);
    });

    it('needs the whole weight for f16 activations', () => {
      expect(() =>
        matmul(input([2, 8], 'f16'), quantWeight(24, 16, 8), { ...q, baseRow: 16 }),
      ).toThrow(/an f16 input or an epilogue on a quantW weight needs the whole weight/);
    });

    it('rejects scales that do not match the whole weight', () => {
      expect(() => matmul(a, quantWeight(8, 16, 8), { ...q, scales: input([16]) })).toThrow(
        /scales must be f32 \[K\/groupSize, N\] = \[2, 16\]/,
      );
    });

    describe('with K not a multiple of the tiled K step', () => {
      // K 12 in groups of 4: three scale rows of 16 columns.
      const q12 = { scales: input([3, 16]), bits: 8 as const, groupSize: 4 };
      const w = quantWeight(12, 16, 8);

      it('runs f32 with no epilogue', () => {
        expect(matmul(input([2, 12]), w, q12).shape).toEqual(meta([2, 16]));
      });

      it.each([
        { name: 'an f16 input', a: input([2, 12], 'f16'), opts: q12 },
        { name: 'a bias', a: input([2, 12]), opts: { ...q12, bias: input([16]) } },
        { name: 'an activation', a: input([2, 12]), opts: { ...q12, activation: 'gelu' as const } },
      ])('rejects $name', ({ a, opts }) => {
        expect(() => matmul(a, w, opts)).toThrow(
          /needs K a multiple of 8 and N and groupSize multiples of 4, got K 12, N 16, groupSize 4/,
        );
      });
    });
  });
});

describe('matmulGather', () => {
  // 3 rows, each picking one of 2 experts of [K=4, N=8]
  const a = input([3, 4]);
  const w = input([8, 8]);
  const idx = input([3]);
  const bias = input([16]);

  it('gives [M, N]', () => {
    const c = matmulGather(a, w, idx, { bias });
    expect(c.shape).toEqual(meta([3, 8]));
    expect(c.pending?.inputs).toEqual([a, w, bias, idx]);
  });

  it.each([
    { name: 'N not a multiple of 4', w: input([8, 6]), idx, error: /n 6 must be divisible by 4/ },
    {
      name: 'weight rows not a multiple of K',
      w: input([6, 8]),
      idx,
      error: /w rows 6 not a multiple of K 4/,
    },
    {
      name: 'one index per row missing',
      w,
      idx: input([2]),
      error: /expertIdx must be f32 with 3 elems/,
    },
    {
      name: 'a weight of another dtype',
      w: input([8, 8], 'f16'),
      idx,
      error: /w is f16 but a is f32/,
    },
    {
      name: 'a quantized weight',
      w: quantWeight(8, 8, 8),
      idx,
      error: /matmulGather.w: expected a 2D f32 or f16 tensor/,
    },
  ])('rejects $name', ({ w, idx, error }) => {
    expect(() => matmulGather(a, w, idx, { bias })).toThrow(error);
  });
});

// 3 tokens routed over 2 experts, int8 weights in groups of 4. route is
// topk's [T, 8]: 4 expert ids, then 4 gate weights.
const moe = { bits: 8 as const, groupSize: 4 };
const route = input([3, 8]);

/** One routed layer for 2 experts of [K, N]. */
const experts = (k: number, n: number, bits: 8 | 4 = 8) => ({
  w: quantWeight(2 * k, n, bits),
  scales: input([(2 * k) / moe.groupSize, n]),
  bias: input([2, n]),
});

describe('moeUp', () => {
  // K=8 in, N=16 out per slot
  const x = input([3, 8]);
  const glu = experts(8, 16);
  const lin = experts(8, 16);

  it('gives one row per (token, slot) and passes the expert count', () => {
    const c = moeUp(x, route, glu, lin, moe);
    expect(c.shape).toEqual(meta([12, 16]));
    expect(c.pending?.op).toBe('moeUp');
    // attrs: bits, groupSize, experts
    expect(c.pending?.attrs).toEqual([8, 4, 2]);
  });

  it('joins the glu and lin biases into one [2E, N] input', () => {
    const c = moeUp(x, route, glu, lin, moe);
    expect(c.pending?.inputs.slice(0, 6)).toEqual([x, route, glu.w, glu.scales, lin.w, lin.scales]);
    const bias = c.pending?.inputs[6];
    expect(bias?.shape).toEqual(meta([4, 16]));
    expect(bias?.pending?.inputs).toEqual([glu.bias, lin.bias]);
  });

  it('takes int4 weights', () => {
    const c = moeUp(x, route, experts(8, 16, 4), experts(8, 16, 4), { ...moe, bits: 4 });
    expect(c.pending?.attrs).toEqual([4, 4, 2]);
  });

  it.each([
    { name: 'f16 activations', x: input([3, 8], 'f16'), error: /expected a 2D f32 tensor/ },
    {
      name: 'a float weight',
      glu: { ...glu, w: input([16, 16]) },
      error: /w must be a quantW \[E·8, N\] weight/,
    },
    {
      name: 'weight rows not a multiple of K',
      glu: { ...glu, w: quantWeight(12, 16, 8) },
      error: /w must be a quantW \[E·8, N\] weight/,
    },
    {
      name: 'a weight whose word count does not match bits',
      glu: { ...glu, w: quantWeight(16, 16, 4) },
      error: /elems 32 != rows·cols·8\/32/,
    },
    {
      name: 'N not a multiple of 4',
      glu: experts(8, 6),
      lin: experts(8, 6),
      error: /K 8 and N 6 must be multiples of 4/,
    },
    {
      name: 'a group size that does not divide K',
      opts: { groupSize: 3 },
      error: /groupSize 3 must be a multiple of 4 that divides K 8/,
    },
    {
      name: 'scales of the wrong shape',
      glu: { ...glu, scales: input([2, 16]) },
      error: /scales must be f32 \[4, 16\]/,
    },
    {
      name: 'a bias of the wrong shape',
      glu: { ...glu, bias: input([16]) },
      error: /bias must be f32 \[2, 16\]/,
    },
    {
      name: 'a route that is not topk-shaped',
      route: input([3, 4]),
      error: /route must be topk's f32 \[3, 8\] output/,
    },
    {
      name: 'glu and lin of different widths',
      lin: experts(8, 8),
      error: /glu and lin weights must have the same shape/,
    },
  ])(
    'rejects $name',
    ({ x: xIn = x, route: r = route, glu: g = glu, lin: l = lin, opts, error }) => {
      expect(() => moeUp(xIn, r, g, l, { ...moe, ...opts })).toThrow(error);
    },
  );
});

describe('moeDown', () => {
  // moeUp's [T·4, K=16] in, N=8 out per token
  const act = input([12, 16]);
  const out = experts(16, 8);
  const residual = input([3, 8]);

  it('gives one row per token and passes the expert count', () => {
    const c = moeDown(act, route, out, residual, moe);
    expect(c.shape).toEqual(meta([3, 8]));
    expect(c.pending?.inputs).toEqual([act, route, out.w, out.scales, out.bias, residual]);
    // attrs: bits, groupSize, experts
    expect(c.pending?.attrs).toEqual([8, 4, 2]);
  });

  it.each([
    {
      name: 'act rows not a multiple of 4 slots',
      act: input([10, 16]),
      error: /act rows 10 must be a multiple of 4/,
    },
    {
      name: 'a residual of the wrong shape',
      residual: input([3, 4]),
      error: /residual must be f32 \[3, 8\]/,
    },
    {
      name: 'a route for another token count',
      route: input([2, 8]),
      error: /route must be topk's f32 \[3, 8\] output/,
    },
    {
      name: 'a bias of the wrong shape',
      out: { ...out, bias: input([3, 8]) },
      error: /bias must be f32 \[2, 8\]/,
    },
  ])(
    'rejects $name',
    ({ act: a = act, route: r = route, out: o = out, residual: res = residual, error }) => {
      expect(() => moeDown(a, r, o, res, moe)).toThrow(error);
    },
  );
});

describe('argmaxDot', () => {
  it('gives one f32 index, even for f16 inputs', () => {
    const c = argmaxDot(input([50000, 16], 'f16'), input([1, 16], 'f16'));
    expect(c.shape).toEqual(meta([1, 1]));
  });

  it.each([
    { name: 'more than one row', x: input([2, 16]), error: /must be a single row, got 2/ },
    { name: 'mismatched cols', x: input([1, 8]), error: /x cols 8 != w cols 16/ },
    { name: 'mismatched dtypes', x: input([1, 16], 'f16'), error: /must match x's f16/ },
  ])('rejects $name', ({ x, error }) => {
    expect(() => argmaxDot(input([100, 16]), x)).toThrow(error);
  });
});
