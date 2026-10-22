import { describe, expect, it } from 'vitest';
import { toArray } from '../../../../src/core/gpu/eval.ts';
import { convTranspose2d } from '../../../../src/core/graph/ops.ts';
import { matrix, tensor3d } from '../../../../src/core/graph/value.ts';
import { expectClose, gpu, randomF32, requireFeature, roundF16, upload } from '../../../gpu.ts';

const { root, ex } = await gpu();

interface Case {
  cIn: number;
  cOut: number;
  h: number;
  w: number;
  k: number;
  bias?: boolean;
}

/** torch's ConvTranspose2d with kernelSize = stride: out[co, k·i + di, k·j + dj]
 *  = Σ W[co·k² + di·k + dj, ci] · x[ci, i, j] + b[co], in f64. */
function convTranspose2dRef(x: Float32Array, wt: Float32Array, c: Case, bias?: Float32Array) {
  const { cIn, cOut, h, w, k } = c;
  const hOut = h * k;
  const wOut = w * k;
  const out = new Float64Array(cOut * hOut * wOut);
  for (let co = 0; co < cOut; co++) {
    for (let i = 0; i < h; i++) {
      for (let j = 0; j < w; j++) {
        for (let di = 0; di < k; di++) {
          for (let dj = 0; dj < k; dj++) {
            let sum = bias ? bias[co]! : 0;
            const row = co * k * k + di * k + dj;
            for (let ci = 0; ci < cIn; ci++) sum += wt[row * cIn + ci]! * x[(ci * h + i) * w + j]!;
            out[(co * hOut + k * i + di) * wOut + k * j + dj] = sum;
          }
        }
      }
    }
  }
  return out;
}

async function checkConvTranspose(c: Case, dtype: 'f32' | 'f16') {
  const round = dtype === 'f16' ? roundF16 : (v: Float32Array) => v;
  const x = round(randomF32(c.cIn * c.h * c.w, 1));
  // The weight is always f16.
  const wt = roundF16(randomF32(c.cOut * c.k * c.k * c.cIn, 2));
  const bias = c.bias ? round(randomF32(c.cOut, 3)) : undefined;
  const got = await toArray(
    convTranspose2d(
      upload(root, x, { ...tensor3d(c.cIn, c.h, c.w), dtype }),
      upload(root, wt, { ...matrix(c.cOut * c.k * c.k, c.cIn), dtype: 'f16' }),
      bias && upload(root, bias, { elems: c.cOut, dtype }),
      { kernelSize: c.k, stride: c.k },
    ),
    ex,
  );
  const tol = dtype === 'f16' ? { atol: 2e-3, rtol: 2e-3 } : { atol: 1e-6 * c.cIn, rtol: 1e-5 };
  expectClose(got, convTranspose2dRef(x, wt, c, bias), tol);
}

describe('convTranspose2d kernel', () => {
  // Routes are picked per pipeline in kernels/conv/convTranspose2d.ts: paired
  // reads two f16 weights per u32 when C_in is even, unpaired picks one half
  // per tap when it is odd.
  const SHAPES: { name: string; c: Case }[] = [
    // paired: C_in % 2 == 0
    { name: 'k = 2 (paired)', c: { cIn: 4, cOut: 3, h: 3, w: 5, k: 2 } },
    { name: 'k = 3, bias (paired)', c: { cIn: 6, cOut: 2, h: 4, w: 3, k: 3, bias: true } },
    {
      name: 'SAM mask upscaling (paired)',
      c: { cIn: 64, cOut: 16, h: 16, w: 16, k: 2, bias: true },
    },
    // unpaired: C_in % 2 != 0
    { name: 'one input channel (unpaired)', c: { cIn: 1, cOut: 2, h: 2, w: 2, k: 2 } },
    { name: 'k = 3, bias (unpaired)', c: { cIn: 5, cOut: 3, h: 3, w: 4, k: 3, bias: true } },
    { name: 'k = 1 (unpaired)', c: { cIn: 3, cOut: 4, h: 5, w: 5, k: 1, bias: true } },
    { name: '1×1 in (unpaired)', c: { cIn: 1, cOut: 1, h: 1, w: 1, k: 1 } },
  ];
  it.for((['f32', 'f16'] as const).flatMap((dtype) => SHAPES.map((s) => ({ dtype, ...s }))))(
    '$dtype $name',
    async ({ dtype, c }, ctx) => {
      if (dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
      await checkConvTranspose(c, dtype);
    },
  );

  it('runs past GRID_Y_STRIDE outputs, in the second grid dimension', () =>
    checkConvTranspose({ cIn: 2, cOut: 1, h: 1024, w: 1025, k: 2 }, 'f32'));

  // Known bug: the kernel splits the flat output index into channel, row and
  // column with d.u32(idx / wOut), which TypeGPU compiles to u32(f32(idx) /
  // f32(wOut)). f32 counts exactly only up to 2^24, so past it outputs come
  // from the wrong pixel. Turn into a plain `it` with the fix.
  it.fails('runs past 2^24 outputs', async () => {
    expect(4098 * 4096).toBeGreaterThan(2 ** 24);
    await checkConvTranspose({ cIn: 2, cOut: 1, h: 2049, w: 2048, k: 2 }, 'f32');
  });

  it('rejects an empty input with an INVALID_ARGUMENT', () => {
    expect(() => upload(root, new Float32Array(0), tensor3d(0, 2, 2))).toThrow(
      expect.objectContaining({ code: 'INVALID_ARGUMENT' }),
    );
  });

  const weight = (rows: number, cols: number) =>
    upload(root, new Float32Array(rows * cols), { ...matrix(rows, cols), dtype: 'f16' });
  it.for([
    {
      name: 'a stride other than the kernel',
      rows: 8,
      opts: { kernelSize: 2, stride: 1 },
      err: /only kernelSize === stride/,
    },
    {
      name: 'weight rows not a multiple of k²',
      rows: 6,
      opts: { kernelSize: 2, stride: 2 },
      err: /weight rows 6 not a multiple of k²=4/,
    },
  ])('refuses $name', ({ rows, opts, err }, ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const x = upload(root, new Float32Array(2 * 4), tensor3d(2, 2, 2));
    expect(() => convTranspose2d(x, weight(rows, 2), undefined, opts)).toThrow(err);
  });

  it('refuses an f32 weight', () => {
    const x = upload(root, new Float32Array(2 * 4), tensor3d(2, 2, 2));
    const w = upload(root, new Float32Array(8), [4, 2]);
    expect(() => convTranspose2d(x, w, undefined, { kernelSize: 2, stride: 2 })).toThrow(
      /weight must be 2D f16 \[C_out·k·k, C_in\]/,
    );
  });

  it('refuses a weight for another C_in', (ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const x = upload(root, new Float32Array(2 * 4), tensor3d(2, 2, 2));
    expect(() => convTranspose2d(x, weight(4, 3), undefined, { kernelSize: 2, stride: 2 })).toThrow(
      /weight C_in 3 != input C_in 2/,
    );
  });
});
