import { describe, expect, it } from 'vitest';
import { toArray } from '../../../../src/core/gpu/eval.ts';
import { swigluChunk } from '../../../../src/core/graph/ops.ts';
import { matrix } from '../../../../src/core/graph/value.ts';
import { GRID_Y_STRIDE } from '../../../../src/core/kernels/common.ts';
import { expectClose, gpu, randomF32, requireFeature, roundF16, upload } from '../../../gpu.ts';

const { root, ex } = await gpu();

/** out[r, c] = x[r, c] · silu(x[r, F + c]) for x [rows, 2F], in f64: hidden
 *  first, gate second, as HF's fc1(x).chunk(2). */
function swigluChunkRef(x: Float32Array, rows: number, half: number): Float64Array {
  const out = new Float64Array(rows * half);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < half; c++) {
      const hidden = x[r * 2 * half + c]!;
      const gate = x[r * 2 * half + half + c]!;
      out[r * half + c] = (hidden * gate) / (1 + Math.exp(-gate));
    }
  }
  return out;
}

/** f32: an exp and a division. f16 output adds 2^-11 relative. */
const TOLERANCE = { f32: { atol: 1e-6, rtol: 1e-5 }, f16: { atol: 1e-3, rtol: 2e-3 } };

async function checkSwigluChunk(rows: number, half: number, dtype: 'f32' | 'f16', scale = 4) {
  const scaled = randomF32(rows * 2 * half, 1).map((v) => scale * v);
  const x = dtype === 'f16' ? roundF16(scaled) : scaled;
  const y = swigluChunk(upload(root, x, { ...matrix(rows, 2 * half), dtype }));
  expect(y.shape).toEqual({ ...matrix(rows, half), dtype });
  expectClose(await toArray(y, ex), swigluChunkRef(x, rows, half), TOLERANCE[dtype]);
}

describe('swigluChunk kernel', () => {
  // One route: a thread per output element.
  it.for(
    (['f32', 'f16'] as const).flatMap((dtype) =>
      [
        { name: '1×1 out', rows: 1, half: 1 },
        { name: 'odd sizes', rows: 3, half: 7 },
        { name: 'one past a workgroup', rows: 1, half: 65 },
        { name: 'many rows', rows: 300, half: 96 },
      ].map((c) => ({ dtype, ...c })),
    ),
  )('$dtype $name: [$rows, 2·$half]', async ({ dtype, rows, half }, ctx) => {
    if (dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    await checkSwigluChunk(rows, half, dtype);
  });

  // The gate's exp(−gate) overflows f32 past −88; silu then goes to −0, not NaN.
  it('saturates for gates far from 0', async () => {
    const x = new Float32Array([2, -3, 1.5, 0.5, 100, -100, -1000, 0]);
    const got = await toArray(swigluChunk(upload(root, x, [1, 8])), ex);
    expectClose(got, swigluChunkRef(x, 1, 4), TOLERANCE.f32);
  });

  it('runs past GRID_Y_STRIDE outputs, in the second grid dimension', async () => {
    expect(2048 * 2049).toBeGreaterThan(GRID_Y_STRIDE);
    await checkSwigluChunk(2048, 2049, 'f32');
  });

  // Known bug: the kernel splits the flat output index into row and column
  // with d.u32(i / half), which TypeGPU compiles to u32(f32(i) / f32(half)).
  // f32 counts exactly only up to 2^24, so past 2^24 outputs it reads the
  // wrong row. f16 is the dtype whose 2^25 inputs still fit one binding.
  // Turn into a plain `it` with the fix.
  it.fails('f16 past 2^24 outputs', async (ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    expect(4100 * 4097).toBeGreaterThan(2 ** 24);
    await checkSwigluChunk(4100, 4097, 'f16');
  });

  it('rejects an empty input with an INVALID_ARGUMENT', () => {
    expect(() => upload(root, new Float32Array(0), [0, 4])).toThrow(
      expect.objectContaining({ code: 'INVALID_ARGUMENT' }),
    );
  });

  it('refuses an odd row width', () => {
    const x = upload(root, new Float32Array(10), [2, 5]);
    expect(() => swigluChunk(x)).toThrow(/cols 5 must be even/);
  });

  it('refuses an input that is not 2D', () => {
    const x = upload(root, new Float32Array(8), { elems: 8, dtype: 'f32', dims: [8] });
    expect(() => swigluChunk(x)).toThrow(/expected a 2D f32 or f16 tensor/);
  });
});
