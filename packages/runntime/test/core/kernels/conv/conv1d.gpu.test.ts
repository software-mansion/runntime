import { describe, expect, it } from 'vitest';
import { toArray } from '../../../../src/core/gpu/eval.ts';
import { conv1d, type SlotAct } from '../../../../src/core/graph/ops.ts';
import { matrix } from '../../../../src/core/graph/value.ts';
import { expectClose, gpu, randomF32, requireFeature, roundF16, upload } from '../../../gpu.ts';
import { ACTIVATION_REF } from '../matmul/checkMatmul.ts';

const { root, ex } = await gpu();

interface Case {
  tIn: number;
  cIn: number;
  cOut: number;
  kernelSize: number;
  stride: number;
  padLeft?: number;
  padRight?: number;
  bias?: boolean;
  activation?: SlotAct;
}

/** out[t, co] = act(bias[co] + Σ x[t·stride + tap − padLeft, ci] ·
 *  w[tap·C_in + ci, co]) over taps inside x, time-major, in f64. */
function conv1dRef(x: Float32Array, w: Float32Array, c: Case, bias?: Float32Array): Float64Array {
  const { tIn, cIn, cOut, kernelSize, stride, padLeft = 0, padRight = 0 } = c;
  const tOut = Math.floor((tIn + padLeft + padRight - kernelSize) / stride) + 1;
  const out = new Float64Array(tOut * cOut);
  for (let t = 0; t < tOut; t++) {
    for (let co = 0; co < cOut; co++) {
      let sum = bias ? bias[co]! : 0;
      for (let tap = 0; tap < kernelSize; tap++) {
        const ti = t * stride + tap - padLeft;
        if (ti < 0 || ti >= tIn) continue;
        for (let ci = 0; ci < cIn; ci++)
          sum += x[ti * cIn + ci]! * w[(tap * cIn + ci) * cOut + co]!;
      }
      out[t * cOut + co] = c.activation ? ACTIVATION_REF[c.activation](sum) : sum;
    }
  }
  return out;
}

async function checkConv1d(c: Case, dtype: 'f32' | 'f16') {
  const round = dtype === 'f16' ? roundF16 : (v: Float32Array) => v;
  const x = round(randomF32(c.tIn * c.cIn, 1));
  const w = round(randomF32(c.kernelSize * c.cIn * c.cOut, 2));
  const bias = c.bias ? round(randomF32(c.cOut, 3)) : undefined;
  const meta = (rows: number, cols: number) => ({ ...matrix(rows, cols), dtype });
  const got = await toArray(
    conv1d(
      upload(root, x, meta(c.tIn, c.cIn)),
      upload(root, w, meta(c.kernelSize * c.cIn, c.cOut)),
      {
        kernelSize: c.kernelSize,
        stride: c.stride,
        padLeft: c.padLeft,
        padRight: c.padRight,
        bias: bias && upload(root, bias, meta(1, c.cOut)),
        activation: c.activation,
      },
    ),
    ex,
  );
  const k = c.kernelSize * c.cIn;
  const tol = dtype === 'f16' ? { atol: 2e-3, rtol: 2e-3 } : { atol: 1e-6 * k, rtol: 1e-5 };
  expectClose(got, conv1dRef(x, w, c, bias), tol);
}

describe('conv1d kernel', () => {
  // Routes are picked in gpu/dispatch/conv.ts.
  const SHAPES: { name: string; c: Case }[] = [
    // direct: padding, so a window can run past either end
    {
      name: 'same padding (direct)',
      c: { tIn: 9, cIn: 3, cOut: 8, kernelSize: 3, stride: 1, padLeft: 1, padRight: 1 },
    },
    {
      name: 'causal padding (direct)',
      c: { tIn: 10, cIn: 4, cOut: 4, kernelSize: 4, stride: 1, padLeft: 3 },
    },
    {
      name: 'right padding only (direct)',
      c: { tIn: 10, cIn: 2, cOut: 4, kernelSize: 3, stride: 2, padRight: 2 },
    },
    // direct: C_out % 4 != 0 runs the scalar tail unit
    {
      name: 'C_out not a multiple of 4 (direct)',
      c: { tIn: 11, cIn: 5, cOut: 7, kernelSize: 3, stride: 1 },
    },
    {
      name: 'one output channel (direct)',
      c: { tIn: 6, cIn: 3, cOut: 1, kernelSize: 2, stride: 1 },
    },
    // direct: C_in % 4 != 0
    {
      name: 'stride 3, odd C_in (direct)',
      c: { tIn: 20, cIn: 3, cOut: 8, kernelSize: 5, stride: 3 },
    },
    // direct: kernelSize·C_in < 64 is too short for the tiled matmul
    {
      name: 'K one quad under the tiled minimum (direct)',
      c: { tIn: 30, cIn: 4, cOut: 8, kernelSize: 15, stride: 1 },
    },
    { name: '1×1 out (direct)', c: { tIn: 1, cIn: 1, cOut: 1, kernelSize: 1, stride: 1 } },
    // gemm: no padding, C_in % 4 == 0, kernelSize·C_in >= 64 and % 4 == 0, C_out % 4 == 0
    {
      name: 'at the tiled minimum K (gemm)',
      c: { tIn: 30, cIn: 4, cOut: 8, kernelSize: 16, stride: 1 },
    },
    { name: 'stride 2 (gemm)', c: { tIn: 70, cIn: 16, cOut: 36, kernelSize: 4, stride: 2 } },
    { name: 'one frame out (gemm)', c: { tIn: 8, cIn: 8, cOut: 4, kernelSize: 8, stride: 1 } },
    {
      name: 'Moonshine stem (gemm)',
      c: { tIn: 400, cIn: 288, cOut: 288, kernelSize: 3, stride: 2 },
    },
    // direct: same as above, but C_out % 4 != 0
    {
      name: 'gemm shape with C_out % 4 != 0 (direct)',
      c: { tIn: 70, cIn: 16, cOut: 35, kernelSize: 4, stride: 2 },
    },
    // direct: a gemm shape with one condition broken at a time. tIn 71 makes
    // padRight add an output frame; stride 1 keeps the C_in 18 row stride off
    // a multiple of 4.
    {
      name: 'gemm shape with padLeft (direct)',
      c: { tIn: 70, cIn: 16, cOut: 36, kernelSize: 4, stride: 2, padLeft: 1 },
    },
    {
      name: 'gemm shape with padRight (direct)',
      c: { tIn: 71, cIn: 16, cOut: 36, kernelSize: 4, stride: 2, padRight: 1 },
    },
    {
      name: 'gemm shape with C_in % 4 != 0 (direct)',
      c: { tIn: 70, cIn: 18, cOut: 36, kernelSize: 4, stride: 1 },
    },
  ];
  it.for((['f32', 'f16'] as const).flatMap((dtype) => SHAPES.map((s) => ({ dtype, ...s }))))(
    '$dtype $name',
    async ({ dtype, c }, ctx) => {
      if (dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
      await checkConv1d(c, dtype);
    },
  );

  // Every epilogue on both routes: act(conv + bias).
  const EPILOGUES = [
    { name: 'bias', bias: true },
    ...(['tanh', 'gelu', 'silu', 'relu', 'geluTanh'] as const).map((activation) => ({
      name: `bias and ${activation}`,
      bias: true,
      activation,
    })),
    { name: 'gelu, no bias', activation: 'gelu' as const },
  ];
  it.for(
    (['f32', 'f16'] as const).flatMap((dtype) =>
      [
        // direct: padded, with the scalar tail
        {
          route: 'direct',
          c: { tIn: 11, cIn: 3, cOut: 7, kernelSize: 3, stride: 1, padLeft: 1, padRight: 1 },
        },
        // gemm: unpadded, C_in and C_out % 4 == 0, K = 64
        { route: 'gemm', c: { tIn: 40, cIn: 16, cOut: 12, kernelSize: 4, stride: 2 } },
      ].flatMap((r) => EPILOGUES.map((e) => ({ dtype, ...r, ...e }))),
    ),
  )('$dtype $route with $name', async ({ dtype, c, bias, activation }, ctx) => {
    if (dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    await checkConv1d({ ...c, bias, activation }, dtype);
  });

  // padding 'same' with an even kernel pads one more on the right, as torch does.
  it("padding 'same' with an even kernel", async () => {
    const x = randomF32(9 * 2, 1);
    const w = randomF32(4 * 2 * 4, 2);
    const got = await toArray(
      conv1d(upload(root, x, [9, 2]), upload(root, w, [8, 4]), {
        kernelSize: 4,
        stride: 1,
        padding: 'same',
      }),
      ex,
    );
    const want = conv1dRef(x, w, {
      tIn: 9,
      cIn: 2,
      cOut: 4,
      kernelSize: 4,
      stride: 1,
      padLeft: 1,
      padRight: 2,
    });
    expect(got.length).toBe(9 * 4);
    expectClose(got, want, { atol: 1e-6, rtol: 1e-5 });
  });

  it('rejects an empty input with an INVALID_ARGUMENT', () => {
    expect(() => upload(root, new Float32Array(0), [0, 4])).toThrow(
      expect.objectContaining({ code: 'INVALID_ARGUMENT' }),
    );
  });

  const base = { kernelSize: 3, stride: 1 };
  it.each([
    {
      name: 'a pad as wide as the kernel',
      opts: { ...base, padLeft: 3 },
      err: /padLeft 3 must be an integer in \[0, kernelSize\)/,
    },
    {
      name: 'padding and padLeft together',
      opts: { ...base, padding: 1, padLeft: 1 },
      err: /padding excludes padLeft\/padRight/,
    },
    {
      name: "'same' with stride 2",
      opts: { ...base, stride: 2, padding: 'same' as const },
      err: /padding 'same' requires stride 1, got 2/,
    },
    { name: 'stride 0', opts: { ...base, stride: 0 }, err: /must be positive integers/ },
    {
      name: 'w rows that are not kernelSize·C_in',
      opts: { ...base, kernelSize: 2 },
      err: /w rows 6 != kernelSize·C_in = 2·2/,
    },
  ])('refuses $name', ({ opts, err }) => {
    const x = upload(root, new Float32Array(10), [5, 2]);
    const w = upload(root, new Float32Array(6 * 4), [6, 4]);
    expect(() => conv1d(x, w, opts)).toThrow(err);
  });

  it('refuses an input shorter than the kernel', () => {
    const x = upload(root, new Float32Array(4), [2, 2]);
    const w = upload(root, new Float32Array(6 * 4), [6, 4]);
    expect(() => conv1d(x, w, base)).toThrow(/padded frames 2\+0\+0 < kernelSize 3/);
  });

  it('refuses a weight in another dtype', (ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const x = upload(root, new Float32Array(10), [5, 2]);
    const w = upload(root, new Float32Array(24), { ...matrix(6, 4), dtype: 'f16' });
    expect(() => conv1d(x, w, base)).toThrow(/weight dtype f16 must match the input's f32/);
  });
});
