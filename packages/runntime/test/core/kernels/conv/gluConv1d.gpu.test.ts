import { describe, it } from 'vitest';
import { toArray } from '../../../../src/core/gpu/eval.ts';
import { gluConv1d } from '../../../../src/core/graph/ops.ts';
import { matrix } from '../../../../src/core/graph/value.ts';
import { expectClose, gpu, randomF32, requireFeature, roundF16, upload } from '../../../gpu.ts';

const { root, ex } = await gpu();

interface Case {
  name: string;
  rows: number;
  channels: number;
  kernel: number;
  lo: number;
  hi: number;
  dtype?: 'f32' | 'f16';
}

/** GLU then a causal depthwise conv, from the definition, in f64:
 *  g[t, c] = clamp(x[t, c]) · σ(clamp(x[t, C + c])), and
 *  out[t, c] = Σ_k w[k, c] · g[t − K + 1 + k, c], 0 before the first frame. */
function gluConv1dRef(x: Float32Array, w: Float32Array, c: Case): Float64Array {
  const { rows, channels: C, kernel: K, lo, hi } = c;
  const clamp = (v: number) => Math.min(Math.max(v, lo), hi);
  const g = (t: number, ch: number) =>
    clamp(x[t * 2 * C + ch]!) / (1 + Math.exp(-clamp(x[t * 2 * C + C + ch]!)));
  const out = new Float64Array(rows * C);
  for (let t = 0; t < rows; t++) {
    for (let ch = 0; ch < C; ch++) {
      let s = 0;
      for (let k = 0; k < K; k++) {
        const src = t - K + 1 + k;
        if (src >= 0) s += w[k * C + ch]! * g(src, ch);
      }
      out[t * C + ch] = s;
    }
  }
  return out;
}

async function check(c: Case): Promise<void> {
  const { rows, channels, kernel, lo, hi, dtype = 'f32' } = c;
  // The reference sees the f16-rounded inputs the kernel reads.
  const round = dtype === 'f16' ? roundF16 : (v: Float32Array) => v;
  const x = round(randomF32(rows * 2 * channels, 1));
  const w = round(randomF32(kernel * channels, 2));
  const got = await toArray(
    gluConv1d(
      upload(root, x, { ...matrix(rows, 2 * channels), dtype }),
      upload(root, w, { ...matrix(kernel, channels), dtype }),
      { lo, hi },
    ),
    ex,
  );
  // f16 output adds 2^-11 relative.
  const tol = dtype === 'f16' ? { atol: 1e-3, rtol: 2e-3 } : { atol: 1e-6, rtol: 1e-5 };
  expectClose(got, gluConv1dRef(x, w, c), tol);
}

describe('gluConv1d kernel', () => {
  // One route. Inputs are in [-1, 1), so bounds inside that range clamp. The
  // op rejects bad shapes and lo > hi (test/core/graph/ops/conv/conv1d.test.ts).
  // it.for, not it.each: only it.for passes the test context the skip needs.
  it.for<Case>([
    {
      name: 'EmbeddingGemma 2 audio light conv: 1024 channels, 5 taps',
      rows: 40,
      channels: 1024,
      kernel: 5,
      lo: -0.6,
      hi: 0.7,
    },
    { name: 'one tap: GLU alone', rows: 6, channels: 8, kernel: 1, lo: -0.5, hi: 0.5 },
    { name: 'fewer frames than taps', rows: 3, channels: 8, kernel: 5, lo: -0.5, hi: 0.5 },
    { name: 'one frame', rows: 1, channels: 4, kernel: 5, lo: -0.5, hi: 0.5 },
    { name: 'four channels', rows: 9, channels: 4, kernel: 3, lo: -0.5, hi: 0.5 },
    { name: 'bounds past every input', rows: 10, channels: 12, kernel: 4, lo: -100, hi: 100 },
    { name: 'lo equal to hi', rows: 7, channels: 8, kernel: 3, lo: 0.25, hi: 0.25 },
    // x past 2^24 elements: 2100 frames of 2 · 4096.
    {
      name: 'x past 2^24 elements',
      rows: 2100,
      channels: 4096,
      kernel: 5,
      lo: -0.6,
      hi: 0.7,
    },
    {
      name: 'f16, EmbeddingGemma 2 audio light conv',
      rows: 40,
      channels: 1024,
      kernel: 5,
      lo: -0.6,
      hi: 0.7,
      dtype: 'f16',
    },
    {
      name: 'f16, fewer frames than taps',
      rows: 3,
      channels: 8,
      kernel: 5,
      lo: -0.5,
      hi: 0.5,
      dtype: 'f16',
    },
  ])('$name', async (c, ctx) => {
    if (c.dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    await check(c);
  });
});
