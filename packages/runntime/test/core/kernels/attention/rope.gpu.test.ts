import { describe, expect, it } from 'vitest';
import { toArray } from '../../../../src/core/gpu/eval.ts';
import { rope } from '../../../../src/core/graph/ops.ts';
import { matrix } from '../../../../src/core/graph/value.ts';
import { GRID_Y_STRIDE } from '../../../../src/core/kernels/common.ts';
import { expectClose, gpu, randomF32, requireFeature, roundF16, upload } from '../../../gpu.ts';

const { root, ex } = await gpu();

type Layout = 'half' | 'interleaved';

/** The two columns of pair `i` within a head, written from each layout's
 *  definition: the interleaved form pairs (2i, 2i + 1). HF's rotate_half
 *  pairs (j, j + ropeDim/2) within each rope of ropeDim = headDim /
 *  ropesPerHead columns, pair i being pair j = i mod ropeDim/2 of rope
 *  ⌊i / (ropeDim/2)⌋. */
function pairColumns(
  layout: Layout,
  i: number,
  headDim: number,
  ropesPerHead: number,
): [number, number] {
  if (layout === 'interleaved') return [2 * i, 2 * i + 1];
  const ropeDim = headDim / ropesPerHead;
  const first = Math.floor(i / (ropeDim / 2)) * ropeDim + (i % (ropeDim / 2));
  return [first, first + ropeDim / 2];
}

/** The cos and sin tables a model passes: angle t · 10000^(−2i / rotaryDim)
 *  for pair i at position t, written to both columns of the pair. Pairs past
 *  rotaryDim / 2 get cos 1 and sin 0, a partial rotary. */
function ropeTables(
  layout: Layout,
  rows: number,
  headDim: number,
  rotaryDim: number,
  ropesPerHead: number,
): { cos: Float32Array; sin: Float32Array } {
  const cos = new Float32Array(rows * headDim);
  const sin = new Float32Array(rows * headDim);
  for (let t = 0; t < rows; t++) {
    for (let i = 0; i < headDim / 2; i++) {
      const angle = i < rotaryDim / 2 ? t * 10000 ** ((-2 * i) / rotaryDim) : 0;
      for (const col of pairColumns(layout, i, headDim, ropesPerHead)) {
        cos[t * headDim + col] = Math.cos(angle);
        sin[t * headDim + col] = Math.sin(angle);
      }
    }
  }
  return { cos, sin };
}

/** Rotates each pair of every head in columns [srcStart, srcStart + width)
 *  by its angle, in f64. */
function ropeRef(
  layout: Layout,
  x: Float32Array,
  cos: Float32Array,
  sin: Float32Array,
  rows: number,
  cols: number,
  headDim: number,
  srcStart: number,
  width: number,
  ropesPerHead: number,
): Float64Array {
  const out = new Float64Array(rows * width);
  for (let t = 0; t < rows; t++) {
    for (let head = 0; head < width / headDim; head++) {
      for (let i = 0; i < headDim / 2; i++) {
        const [a, b] = pairColumns(layout, i, headDim, ropesPerHead);
        const c = cos[t * headDim + a]!;
        const s = sin[t * headDim + a]!;
        const xa = x[t * cols + srcStart + head * headDim + a]!;
        const xb = x[t * cols + srcStart + head * headDim + b]!;
        out[t * width + head * headDim + a] = xa * c - xb * s;
        out[t * width + head * headDim + b] = xb * c + xa * s;
      }
    }
  }
  return out;
}

interface Case {
  rows: number;
  cols: number;
  headDim: number;
  /** Columns of a head that turn; the rest pass through. Default headDim. */
  rotaryDim?: number;
  /** Ropes a head holds side by side. Default 1. */
  ropesPerHead?: number;
  srcStart?: number;
  width?: number;
}

async function checkRope(
  layout: Layout,
  { rows, cols, headDim, rotaryDim = headDim, ropesPerHead, srcStart, width }: Case,
  dtype: 'f32' | 'f16' = 'f32',
): Promise<void> {
  const x = dtype === 'f16' ? roundF16(randomF32(rows * cols, 1)) : randomF32(rows * cols, 1);
  const { cos, sin } = ropeTables(layout, rows, headDim, rotaryDim, ropesPerHead ?? 1);
  const got = await toArray(
    rope(
      upload(root, x, { ...matrix(rows, cols), dtype }),
      upload(root, cos, [rows, headDim]),
      upload(root, sin, [rows, headDim]),
      { headDim, layout, ropesPerHead, srcStart, width },
    ),
    ex,
  );
  const want = ropeRef(
    layout,
    x,
    cos,
    sin,
    rows,
    cols,
    headDim,
    srcStart ?? 0,
    width ?? cols,
    ropesPerHead ?? 1,
  );
  // f32: two products and a sum. f16 output adds 2^-11 relative.
  const tol = dtype === 'f16' ? { atol: 1e-3, rtol: 1e-3 } : { atol: 1e-6, rtol: 1e-6 };
  expectClose(got, want, tol);
}

const LAYOUTS = ['interleaved', 'half'] as const;

describe('rope kernel', () => {
  // One route: a thread per pair, the layout picking the pair's columns.
  it.for(
    LAYOUTS.flatMap((layout) =>
      (['f32', 'f16'] as const).flatMap((dtype) =>
        [
          { name: 'one pair', c: { rows: 1, cols: 2, headDim: 2 } },
          { name: 'one head', c: { rows: 5, cols: 64, headDim: 64 } },
          // 5 pairs a head: an odd pair count, an odd head count
          { name: 'several heads, odd row count', c: { rows: 7, cols: 7 * 10, headDim: 10 } },
          {
            name: 'pairs not a multiple of the workgroup',
            c: { rows: 3, cols: 4 * 34, headDim: 34 },
          },
          // Moonshine-tiny: 32 of a head's 36 columns turn
          {
            name: 'a partial rotary',
            c: { rows: 6, cols: 8 * 36, headDim: 36, rotaryDim: 32 },
          },
          // The k section of a fused [T, q | k | v] slab, 4 q heads, 2 kv heads
          {
            name: 'a slice of a wider slab',
            c: { rows: 9, cols: 8 * 16, headDim: 16, srcStart: 4 * 16, width: 2 * 16 },
          },
          {
            name: 'the last head of a slab',
            c: { rows: 3, cols: 48, headDim: 16, srcStart: 32, width: 16 },
          },
          {
            name: 'a partial rotary on a slice of a wider slab',
            c: { rows: 4, cols: 5 * 12, headDim: 12, rotaryDim: 8, srcStart: 12, width: 3 * 12 },
          },
          // EmbeddingGemma 2 vision: q and k of 3 heads, each head an x rope
          // and a y rope of 8 columns
          {
            name: 'two ropes a head on a slice of a wider slab',
            c: { rows: 5, cols: 9 * 16, headDim: 16, ropesPerHead: 2, width: 6 * 16 },
          },
          // 3 ropes of 6 columns: an odd rope count, an odd pair count a rope
          {
            name: 'three ropes a head',
            c: { rows: 3, cols: 5 * 18, headDim: 18, ropesPerHead: 3 },
          },
        ].map((c) => ({ layout, dtype, ...c })),
      ),
    ),
  )('$layout $dtype $name', async ({ layout, dtype, c }, ctx) => {
    if (dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    await checkRope(layout, c, dtype);
  });

  // Position 1, headDim 2: angle 1, so (1, 0) turns to (cos 1, sin 1) and
  // (0, 1) to (−sin 1, cos 1), as rotating by 1 radian gives.
  // With one pair a head the two layouts are the same.
  it.each(LAYOUTS)('%s rotates by the angle, worked out by hand', async (layout) => {
    const x = upload(root, new Float32Array([1, 0, 0, 1]), [2, 2]);
    const cos = upload(root, new Float32Array([1, 1, Math.cos(1), Math.cos(1)]), [2, 2]);
    const sin = upload(root, new Float32Array([0, 0, Math.sin(1), Math.sin(1)]), [2, 2]);
    const got = await toArray(rope(x, cos, sin, { headDim: 2, layout }), ex);
    expectClose(got, [1, 0, -Math.sin(1), Math.cos(1)], { atol: 1e-7, rtol: 1e-6 });
  });

  // One row of one head of 4, turned a quarter (cos 0, sin 1), so every
  // column lands on its partner: (a, b) → (−b, a). The partner of column 0
  // is column 1 interleaved and column 2 in the half layout.
  it.each([
    { layout: 'interleaved' as const, ropesPerHead: 1, want: [-2, 1, -4, 3] },
    { layout: 'half' as const, ropesPerHead: 1, want: [-3, -4, 1, 2] },
    // Two ropes of 2 columns: each rope's half split is its two neighbours.
    { layout: 'half' as const, ropesPerHead: 2, want: [-2, 1, -4, 3] },
    { layout: 'interleaved' as const, ropesPerHead: 2, want: [-2, 1, -4, 3] },
  ])(
    '$layout, $ropesPerHead ropes a head, pairs the columns of a head, worked out by hand',
    async ({ layout, ropesPerHead, want }) => {
      const x = upload(root, new Float32Array([1, 2, 3, 4]), [1, 4]);
      const cos = upload(root, new Float32Array(4), [1, 4]);
      const sin = upload(root, new Float32Array(4).fill(1), [1, 4]);
      const got = await toArray(rope(x, cos, sin, { headDim: 4, layout, ropesPerHead }), ex);
      expect(Array.from(got)).toEqual(want);
    },
  );

  // Past GRID_Y_STRIDE pairs the threads take the second grid dimension.
  it.each(LAYOUTS)('%s runs past 2^24 elements', async (layout) => {
    const rows = 4100;
    const cols = 4096;
    expect(rows * cols).toBeGreaterThan(2 ** 24);
    expect((rows * cols) / 2).toBeGreaterThan(GRID_Y_STRIDE);
    await checkRope(layout, { rows, cols, headDim: 64 });
  });

  // f32 counts exactly only up to 2^24, so the flat pair index past it must be
  // split with integer division or rows come out wrong. f16 is the dtype
  // whose 2^25 elements still fit one binding.
  it('f16 past 2^24 pairs', async (ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const rows = 2 ** 14 + 1;
    const cols = 2048;
    expect((rows * cols) / 2).toBeGreaterThan(2 ** 24);
    await checkRope('interleaved', { rows, cols, headDim: 64 }, 'f16');
  });

  it('rejects an empty input with an INVALID_ARGUMENT', () => {
    expect(() => upload(root, new Float32Array(0), [0, 4])).toThrow(
      expect.objectContaining({ code: 'INVALID_ARGUMENT' }),
    );
  });

  it.each([
    {
      name: 'an odd headDim',
      opts: { headDim: 3 },
      err: /headDim 3 must be a positive even integer/,
    },
    {
      name: 'an odd srcStart',
      opts: { headDim: 4, srcStart: 1, width: 4 },
      err: /srcStart 1 must be a non-negative even integer/,
    },
    {
      name: 'a window past the input',
      opts: { headDim: 4, srcStart: 8, width: 8 },
      err: /window \[8, 16\) exceeds 12 input cols/,
    },
    {
      name: 'a width that is not whole heads',
      opts: { headDim: 4, width: 6 },
      err: /width 6 must be a multiple of headDim 4/,
    },
  ])('refuses $name', ({ opts, err }) => {
    const x = upload(root, new Float32Array(2 * 12), [2, 12]);
    const table = upload(root, new Float32Array(2 * opts.headDim), [2, opts.headDim]);
    expect(() => rope(x, table, table, { layout: 'half', ...opts })).toThrow(err);
  });

  it('refuses tables of the wrong size', () => {
    const x = upload(root, new Float32Array(2 * 8), [2, 8]);
    const table = upload(root, new Float32Array(4), [1, 4]);
    expect(() => rope(x, table, table, { headDim: 4, layout: 'half' })).toThrow(
      /cos must be f32 \[T,headDim\] = 8 elems, got 4/,
    );
  });
});
