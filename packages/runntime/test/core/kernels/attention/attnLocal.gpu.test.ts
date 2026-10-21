import { describe, it } from 'vitest';
import { toArray } from '../../../../src/core/gpu/eval.ts';
import { localAttention } from '../../../../src/core/graph/ops.ts';
import { matrix } from '../../../../src/core/graph/value.ts';
import { ATTN_LOCAL_MAX_WINDOW } from '../../../../src/core/kernels/attention/attnLocal.ts';
import { expectClose, gpu, randomF32, requireFeature, roundF16, upload } from '../../../gpu.ts';

const { root, ex } = await gpu();

/** k's scale in the EmbeddingGemma 2 audio tower. */
const GEMMA_K_SCALE = Math.log1p(Math.E) / Math.LN2;

interface Case {
  name: string;
  rows: number;
  heads: number;
  headDim: number;
  window: number;
  kScale?: number;
  softcap?: number;
  dtype?: 'f32' | 'f16';
}

/** Local attention from its definition, in f64: row t of head h scores the
 *  key j rows back as cap · tanh(Σ_d q_d · qScale_d · (k_d · kScale + rel_j,d) / cap)
 *  for j < min(window, t + 1), softmaxes the scores and sums the values. */
function localAttentionRef(
  qkv: Float32Array,
  rel: Float32Array,
  qScale: Float32Array,
  c: Required<Omit<Case, 'name' | 'dtype'>>,
): Float64Array {
  const { rows, heads, headDim, window, kScale, softcap } = c;
  const width = heads * headDim;
  const out = new Float64Array(rows * width);
  const scores = new Float64Array(window);
  for (let t = 0; t < rows; t++) {
    const n = Math.min(window, t + 1);
    for (let h = 0; h < heads; h++) {
      const q = t * 3 * width + h * headDim;
      for (let j = 0; j < n; j++) {
        const k = (t - j) * 3 * width + width + h * headDim;
        let s = 0;
        for (let i = 0; i < headDim; i++) {
          const key = qkv[k + i]! * kScale + rel[j * width + h * headDim + i]!;
          s += qkv[q + i]! * qScale[i]! * key;
        }
        scores[j] = softcap * Math.tanh(s / softcap);
      }
      let max = -Infinity;
      for (let j = 0; j < n; j++) max = Math.max(max, scores[j]!);
      let sum = 0;
      for (let j = 0; j < n; j++) sum += Math.exp(scores[j]! - max);
      for (let i = 0; i < headDim; i++) {
        let acc = 0;
        for (let j = 0; j < n; j++) {
          acc +=
            Math.exp(scores[j]! - max) * qkv[(t - j) * 3 * width + 2 * width + h * headDim + i]!;
        }
        out[t * width + h * headDim + i] = acc / sum;
      }
    }
  }
  return out;
}

async function check(c: Case): Promise<void> {
  const { rows, heads, headDim, window, kScale = 1, softcap = 50, dtype = 'f32' } = c;
  const width = heads * headDim;
  // The reference sees the f16-rounded qkv the kernel reads; rel and qScale
  // are f32 in both dtypes.
  const qkv =
    dtype === 'f16' ? roundF16(randomF32(rows * 3 * width, 1)) : randomF32(rows * 3 * width, 1);
  const rel = randomF32(window * width, 2);
  const qScale = randomF32(headDim, 3);
  const got = await toArray(
    localAttention(
      upload(root, qkv, { ...matrix(rows, 3 * width), dtype }),
      upload(root, rel, matrix(window, width)),
      upload(root, qScale, matrix(1, headDim)),
      { heads, headDim, window, kScale, softcap },
    ),
    ex,
  );
  const want = localAttentionRef(qkv, rel, qScale, {
    rows,
    heads,
    headDim,
    window,
    kScale,
    softcap,
  });
  // f16 output adds 2^-11 relative.
  const tol = dtype === 'f16' ? { atol: 1e-3, rtol: 2e-3 } : { atol: 1e-5, rtol: 1e-4 };
  expectClose(got, want, tol);
}

describe('attnLocal kernel', () => {
  // One route, one workgroup per (row, head). The op rejects a window past
  // ATTN_LOCAL_MAX_WINDOW (test/core/graph/ops/attention.test.ts).
  // it.for, not it.each: only it.for passes the test context the skip needs.
  it.for<Case>([
    {
      name: 'EmbeddingGemma 2 audio layer: 8 heads of 128, window 12, softcap 50',
      rows: 50,
      heads: 8,
      headDim: 128,
      window: 12,
      kScale: GEMMA_K_SCALE,
    },
    { name: 'window 1: each row sees only itself', rows: 6, heads: 2, headDim: 8, window: 1 },
    {
      name: 'the largest window',
      rows: 40,
      heads: 2,
      headDim: 16,
      window: ATTN_LOCAL_MAX_WINDOW,
    },
    { name: 'fewer rows than the window', rows: 3, heads: 2, headDim: 8, window: 12 },
    { name: 'one row', rows: 1, heads: 1, headDim: 8, window: 12 },
    { name: 'a head of 4 values', rows: 9, heads: 3, headDim: 4, window: 5 },
    {
      name: 'a head wider than 128 values',
      rows: 7,
      heads: 1,
      headDim: 136,
      window: 4,
    },
    {
      name: 'a softcap small enough to saturate tanh',
      rows: 16,
      heads: 2,
      headDim: 32,
      window: 6,
      kScale: 4,
      softcap: 0.5,
    },
    { name: 'a negative kScale', rows: 10, heads: 2, headDim: 8, window: 3, kScale: -1.5 },
    // More than 65535 (row, head) workgroups: the grid goes 2-D.
    { name: 'a 2-D grid, 8200 × 8 workgroups', rows: 8200, heads: 8, headDim: 4, window: 3 },
    // qkv past 2^24 elements: 5462 rows of 3 · 8 · 128.
    {
      name: 'qkv past 2^24 elements',
      rows: 5462,
      heads: 8,
      headDim: 128,
      window: 12,
      kScale: GEMMA_K_SCALE,
    },
    {
      name: 'f16, EmbeddingGemma 2 audio layer',
      rows: 50,
      heads: 8,
      headDim: 128,
      window: 12,
      kScale: GEMMA_K_SCALE,
      dtype: 'f16',
    },
    {
      name: 'f16, fewer rows than the window',
      rows: 3,
      heads: 2,
      headDim: 8,
      window: 12,
      dtype: 'f16',
    },
    {
      name: 'f16, a head of 4 values',
      rows: 9,
      heads: 3,
      headDim: 4,
      window: 5,
      dtype: 'f16',
    },
  ])('$name', async (c, ctx) => {
    if (c.dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    await check(c);
  });
});
