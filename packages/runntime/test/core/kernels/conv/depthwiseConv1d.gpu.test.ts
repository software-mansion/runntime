import { describe, expect, it, onTestFinished } from 'vitest';
import { toArray } from '../../../../src/core/gpu/eval.ts';
import { conv1d, gluConv1d } from '../../../../src/core/graph/ops.ts';
import { matrix } from '../../../../src/core/graph/value.ts';
import { Conv1d } from '../../../../src/core/nn/layers.ts';
import { memoryStateDict } from '../../../../src/core/weights/safetensors.ts';
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

interface PlainCase {
  name: string;
  rows: number;
  channels: number;
  kernel: number;
  dtype?: 'f32' | 'f16';
}

/** A causal depthwise conv from the definition, in f64:
 *  out[t, c] = Σ_k w[k, c] · x[t − K + 1 + k, c], 0 before the first frame. */
function depthwiseRef(x: Float32Array, w: Float32Array, c: PlainCase): Float64Array {
  const { rows, channels: C, kernel: K } = c;
  const out = new Float64Array(rows * C);
  for (let t = 0; t < rows; t++) {
    for (let ch = 0; ch < C; ch++) {
      let s = 0;
      for (let k = 0; k < K; k++) {
        const src = t - K + 1 + k;
        if (src >= 0) s += w[k * C + ch]! * x[src * C + ch]!;
      }
      out[t * C + ch] = s;
    }
  }
  return out;
}

/** f16 output adds 2^-11 relative. */
const PLAIN_TOLERANCE = {
  f32: { atol: 1e-6, rtol: 1e-5 },
  f16: { atol: 1e-3, rtol: 2e-3 },
};

describe('depthwise conv1d kernel', () => {
  // One route: conv1d with groups = C_in, the kernel above without its GLU.
  // The op rejects every other geometry (test/core/graph/ops/conv/conv1d.test.ts).
  it.for<PlainCase>([
    {
      name: 'EmbeddingGemma 2 audio light conv: 1024 channels, 5 taps',
      rows: 40,
      channels: 1024,
      kernel: 5,
    },
    { name: 'one tap: a per-channel scale', rows: 6, channels: 8, kernel: 1 },
    { name: 'fewer frames than taps', rows: 3, channels: 8, kernel: 5 },
    { name: 'one frame', rows: 1, channels: 4, kernel: 5 },
    { name: 'four channels', rows: 9, channels: 4, kernel: 3 },
    // x past 2^24 elements: 4200 frames of 4096.
    { name: 'x past 2^24 elements', rows: 4200, channels: 4096, kernel: 5 },
    {
      name: 'f16, EmbeddingGemma 2 audio light conv',
      rows: 40,
      channels: 1024,
      kernel: 5,
      dtype: 'f16',
    },
    { name: 'f16, fewer frames than taps', rows: 3, channels: 8, kernel: 5, dtype: 'f16' },
  ])('$name', async (c, ctx) => {
    const { rows, channels, kernel, dtype = 'f32' } = c;
    if (dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    // The reference sees the f16-rounded inputs the kernel reads.
    const round = dtype === 'f16' ? roundF16 : (v: Float32Array) => v;
    const x = round(randomF32(rows * channels, 1));
    const w = round(randomF32(kernel * channels, 2));
    const y = conv1d(
      upload(root, x, { ...matrix(rows, channels), dtype }),
      upload(root, w, { ...matrix(kernel, channels), dtype }),
      { kernelSize: kernel, stride: 1, padLeft: kernel - 1, groups: channels },
    );
    expectClose(await toArray(y, ex), depthwiseRef(x, w, c), PLAIN_TOLERANCE[dtype]);
  });

  // The layer loads torch's [channels, 1, kernel] and runs it tap-major.
  it.for(['f32', 'f16'] as const)(
    'nn.Conv1d runs a torch depthwise weight, %s',
    async (dtype, ctx) => {
      if (dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
      const c = { name: 'layer', rows: 12, channels: 8, kernel: 3 };
      const round = dtype === 'f16' ? roundF16 : (v: Float32Array) => v;
      const x = round(randomF32(c.rows * c.channels, 1));
      const torch = round(randomF32(c.channels * c.kernel, 2));
      const conv = new Conv1d(c.channels, c.channels, c.kernel, {
        groups: c.channels,
        padLeft: c.kernel - 1,
        bias: false,
      });
      if (dtype === 'f16') conv.half();
      await conv.loadStateDict(
        memoryStateDict({ weight: { data: torch, shape: [c.channels, 1, c.kernel] } }),
        { root },
      );
      onTestFinished(() => conv.dispose());
      expect(conv.weight.shape.dtype).toBe(dtype);
      const tapMajor = new Float32Array(torch.length);
      for (let ch = 0; ch < c.channels; ch++) {
        for (let k = 0; k < c.kernel; k++)
          tapMajor[k * c.channels + ch] = torch[ch * c.kernel + k]!;
      }
      const y = conv.forward(upload(root, x, { ...matrix(c.rows, c.channels), dtype }));
      expectClose(await toArray(y, ex), depthwiseRef(x, tapMajor, c), PLAIN_TOLERANCE[dtype]);
    },
  );
});
