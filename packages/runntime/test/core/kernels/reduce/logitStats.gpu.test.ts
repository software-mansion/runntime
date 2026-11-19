import { describe, expect, it } from 'vitest';
import { toArray } from '../../../../src/core/gpu/eval.ts';
import { logitStats } from '../../../../src/core/graph/ops.ts';
import { matrix } from '../../../../src/core/graph/value.ts';
import { MAX_WORKGROUPS_PER_DIM } from '../../../../src/core/kernels/common.ts';
import { expectClose, gpu, randomF32, requireFeature, roundF16, upload } from '../../../gpu.ts';

const { root, ex } = await gpu();

interface StatsCase {
  rows: number;
  cols: number;
  blocks: number;
  block: number;
  argmaxCols: number;
  guideWeight?: number;
}

/** For block b of row r, at (b·rows + r)·2: the lowest column of the largest
 *  x, or (1 + w)·x − w·guide with a guide, among the block's first argmaxCols,
 *  then that column's x minus the logsumexp of the whole block. In f64. */
function logitStatsRef(x: Float32Array, c: StatsCase, guide?: Float32Array): Float64Array {
  const { rows, cols, blocks, block, argmaxCols } = c;
  const w = c.guideWeight ?? 0;
  const out = new Float64Array(blocks * rows * 2);
  for (let b = 0; b < blocks; b++) {
    for (let r = 0; r < rows; r++) {
      const base = r * cols + b * block;
      let top = 0;
      let best = -Infinity;
      for (let j = 0; j < argmaxCols; j++) {
        const v = guide ? (1 + w) * x[base + j]! - w * guide[base + j]! : x[base + j]!;
        if (v > best) {
          best = v;
          top = j;
        }
      }
      let max = -Infinity;
      for (let j = 0; j < block; j++) max = Math.max(max, x[base + j]!);
      let sum = 0;
      for (let j = 0; j < block; j++) sum += Math.exp(x[base + j]! - max);
      out[(b * rows + r) * 2] = top;
      out[(b * rows + r) * 2 + 1] = x[base + top]! - max - Math.log(sum);
    }
  }
  return out;
}

/** f32: a logsumexp over up to 1025 logits. f16 inputs are exact and the
 *  sums stay f32. */
const TOLERANCE = { atol: 1e-5, rtol: 1e-5 };

/** Random logits, and a random guide when the case has a guideWeight. */
async function checkStats(c: StatsCase, dtype: 'f32' | 'f16') {
  const round = dtype === 'f16' ? roundF16 : (v: Float32Array) => v;
  const x = round(randomF32(c.rows * c.cols, 1).map((v) => 6 * v));
  const guide =
    c.guideWeight !== undefined
      ? round(randomF32(c.rows * c.cols, 2).map((v) => 6 * v))
      : undefined;
  const shape = { ...matrix(c.rows, c.cols), dtype };
  const y = logitStats(upload(root, x, shape), {
    blocks: c.blocks,
    block: c.block,
    argmaxCols: c.argmaxCols,
    guide: guide && upload(root, guide, shape),
    guideWeight: c.guideWeight,
  });
  expectClose(await toArray(y, ex), logitStatsRef(x, c, guide), TOLERANCE);
}

describe('logitStats kernel', () => {
  // One route: a 64-thread workgroup per block of a row, reduced as a tree.
  it.for(
    (['f32', 'f16'] as const).flatMap((dtype) =>
      [
        { name: 'one block of one row', rows: 1, cols: 7, blocks: 1, block: 7, argmaxCols: 7 },
        {
          name: 'blocks under a workgroup',
          rows: 5,
          cols: 30,
          blocks: 3,
          block: 10,
          argmaxCols: 9,
        },
        // Columns past blocks·block are padding the kernel never reads.
        { name: 'padded rows', rows: 3, cols: 50, blocks: 2, block: 20, argmaxCols: 20 },
        {
          name: 'blocks past a workgroup',
          rows: 4,
          cols: 3080,
          blocks: 3,
          block: 1025,
          argmaxCols: 1024,
        },
        {
          name: 'a guide',
          rows: 6,
          cols: 2050,
          blocks: 2,
          block: 1025,
          argmaxCols: 1024,
          guideWeight: 0.75,
        },
      ].map((c) => ({ dtype, ...c })),
    ),
  )('$dtype $name: [$rows, $cols], $blocks × $block', async ({ dtype, ...c }, ctx) => {
    if (dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    await checkStats(c, dtype);
  });

  // The largest logit sits past argmaxCols, so only the log-probability sees it.
  it('takes the argmax over the first argmaxCols columns only', async () => {
    const x = new Float32Array([1, 3, 2, 9]);
    const got = await toArray(
      logitStats(upload(root, x, [1, 4]), { blocks: 1, block: 4, argmaxCols: 3 }),
      ex,
    );
    expect(got[0]).toBe(1);
    expectClose(
      got,
      logitStatsRef(x, { rows: 1, cols: 4, blocks: 1, block: 4, argmaxCols: 3 }),
      TOLERANCE,
    );
  });

  // Thread t reads columns t, t + 64, … Thread 1 sees the tie at 1 and 129
  // and keeps 1. The tree then meets column 64 from thread 0 and keeps 1.
  it('breaks ties toward the lowest column', async () => {
    const x = new Float32Array(130);
    x[64] = 5;
    x[1] = 5;
    x[129] = 5;
    const got = await toArray(
      logitStats(upload(root, x, [1, 130]), { blocks: 1, block: 130, argmaxCols: 130 }),
      ex,
    );
    expect(got[0]).toBe(1);
  });

  // Past 65535 workgroups the (block, row) slots take the second grid
  // dimension.
  it('runs past MAX_WORKGROUPS_PER_DIM blocks', async () => {
    const c = { rows: 33_000, cols: 8, blocks: 2, block: 4, argmaxCols: 4 };
    expect(c.rows * c.blocks).toBeGreaterThan(MAX_WORKGROUPS_PER_DIM);
    await checkStats(c, 'f32');
  });
});
