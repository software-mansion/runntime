import { describe, expect, it } from 'vitest';
import { toArray } from '../../../../src/core/gpu/eval.ts';
import { rope } from '../../../../src/core/graph/ops.ts';
import { matrix } from '../../../../src/core/graph/value.ts';
import { GRID_Y_STRIDE } from '../../../../src/core/kernels/common.ts';
import { expectClose, gpu, randomF32, requireFeature, roundF16, upload } from '../../../gpu.ts';

const { root, ex } = await gpu();

/** The pair-duplicated cos and sin tables a model passes: angle t · 10000^(−2i
 *  / headDim) for pair i at position t, written to both columns of the pair. */
function ropeTables(rows: number, headDim: number): { cos: Float32Array; sin: Float32Array } {
  const cos = new Float32Array(rows * headDim);
  const sin = new Float32Array(rows * headDim);
  for (let t = 0; t < rows; t++) {
    for (let i = 0; i < headDim / 2; i++) {
      const angle = t * 10000 ** ((-2 * i) / headDim);
      cos[t * headDim + 2 * i] = cos[t * headDim + 2 * i + 1] = Math.cos(angle);
      sin[t * headDim + 2 * i] = sin[t * headDim + 2 * i + 1] = Math.sin(angle);
    }
  }
  return { cos, sin };
}

/** Rotates each interleaved pair (x[2p], x[2p+1]) of columns [srcStart,
 *  srcStart + width) by its angle, in f64. */
function ropeRef(
  x: Float32Array,
  cos: Float32Array,
  sin: Float32Array,
  rows: number,
  cols: number,
  headDim: number,
  srcStart: number,
  width: number,
): Float64Array {
  const out = new Float64Array(rows * width);
  for (let t = 0; t < rows; t++) {
    for (let col = 0; col < width; col += 2) {
      const c = cos[t * headDim + (col % headDim)]!;
      const s = sin[t * headDim + (col % headDim)]!;
      const x0 = x[t * cols + srcStart + col]!;
      const x1 = x[t * cols + srcStart + col + 1]!;
      out[t * width + col] = x0 * c - x1 * s;
      out[t * width + col + 1] = x1 * c + x0 * s;
    }
  }
  return out;
}

interface Case {
  rows: number;
  cols: number;
  headDim: number;
  srcStart?: number;
  width?: number;
}

async function checkRope(
  { rows, cols, headDim, srcStart, width }: Case,
  dtype: 'f32' | 'f16' = 'f32',
): Promise<void> {
  const x = dtype === 'f16' ? roundF16(randomF32(rows * cols, 1)) : randomF32(rows * cols, 1);
  const { cos, sin } = ropeTables(rows, headDim);
  const got = await toArray(
    rope(
      upload(root, x, { ...matrix(rows, cols), dtype }),
      upload(root, cos, [rows, headDim]),
      upload(root, sin, [rows, headDim]),
      { headDim, srcStart, width },
    ),
    ex,
  );
  const want = ropeRef(x, cos, sin, rows, cols, headDim, srcStart ?? 0, width ?? cols);
  // f32: two products and a sum. f16 output adds 2^-11 relative.
  const tol = dtype === 'f16' ? { atol: 1e-3, rtol: 1e-3 } : { atol: 1e-6, rtol: 1e-6 };
  expectClose(got, want, tol);
}

describe('rope kernel', () => {
  // One route: a thread per pair.
  it.for(
    (['f32', 'f16'] as const).flatMap((dtype) =>
      [
        { name: 'one pair', c: { rows: 1, cols: 2, headDim: 2 } },
        { name: 'one head', c: { rows: 5, cols: 64, headDim: 64 } },
        { name: 'several heads, odd row count', c: { rows: 7, cols: 6 * 10, headDim: 10 } },
        {
          name: 'pairs not a multiple of the workgroup',
          c: { rows: 3, cols: 4 * 34, headDim: 34 },
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
      ].map((c) => ({ dtype, ...c })),
    ),
  )('$dtype $name', async ({ dtype, c }, ctx) => {
    if (dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    await checkRope(c, dtype);
  });

  // Position 1, headDim 2: angle 1, so (1, 0) turns to (cos 1, sin 1) and
  // (0, 1) to (−sin 1, cos 1), as rotating by 1 radian gives.
  it('rotates by the angle, worked out by hand', async () => {
    const x = upload(root, new Float32Array([1, 0, 0, 1]), [2, 2]);
    const cos = upload(root, new Float32Array([1, 1, Math.cos(1), Math.cos(1)]), [2, 2]);
    const sin = upload(root, new Float32Array([0, 0, Math.sin(1), Math.sin(1)]), [2, 2]);
    const got = await toArray(rope(x, cos, sin, { headDim: 2 }), ex);
    expectClose(got, [1, 0, -Math.sin(1), Math.cos(1)], { atol: 1e-7, rtol: 1e-6 });
  });

  // Past GRID_Y_STRIDE pairs the threads take the second grid dimension.
  it('runs past 2^24 elements', async () => {
    const rows = 4100;
    const cols = 4096;
    expect(rows * cols).toBeGreaterThan(2 ** 24);
    expect((rows * cols) / 2).toBeGreaterThan(GRID_Y_STRIDE);
    await checkRope({ rows, cols, headDim: 64 });
  });

  // Known bug: the kernel splits the flat pair index into row and pair with
  // d.u32(idx / pairs), which TypeGPU compiles to u32(f32(idx) / f32(pairs)).
  // f32 counts exactly only up to 2^24, so past 2^24 pairs rows come out
  // wrong. f16 is the dtype whose 2^25 elements still fit one binding. Turn
  // into a plain `it` with the fix.
  it.fails('f16 past 2^24 pairs', async (ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const rows = 2 ** 14 + 1;
    const cols = 2048;
    expect((rows * cols) / 2).toBeGreaterThan(2 ** 24);
    await checkRope({ rows, cols, headDim: 64 }, 'f16');
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
    expect(() => rope(x, table, table, opts)).toThrow(err);
  });

  it('refuses tables of the wrong size', () => {
    const x = upload(root, new Float32Array(2 * 8), [2, 8]);
    const table = upload(root, new Float32Array(4), [1, 4]);
    expect(() => rope(x, table, table, { headDim: 4 })).toThrow(
      /cos must be f32 \[T,headDim\] = 8 elems, got 4/,
    );
  });
});
