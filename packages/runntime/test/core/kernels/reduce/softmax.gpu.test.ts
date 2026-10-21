import { describe, expect, it, type TestContext } from 'vitest';
import { toArray } from '../../../../src/core/gpu/eval.ts';
import { softmax } from '../../../../src/core/graph/ops.ts';
import { matrix } from '../../../../src/core/graph/value.ts';
import {
  expectClose,
  gpu,
  gpuWithoutSubgroups,
  randomF32,
  requireFeature,
  roundF16,
  upload,
} from '../../../gpu.ts';

const main = await gpu();
const withoutSubgroups = await gpuWithoutSubgroups();
const { root, ex } = main;

/** Softmax of each row from the definition, in f64. `keep(r, j)` masks key j
 *  out of row r: it is left out of the max and the sum and comes out 0. */
function softmaxRef(
  x: Float32Array,
  rows: number,
  cols: number,
  keep: (r: number, j: number) => boolean = () => true,
): Float64Array {
  const out = new Float64Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    let max = -Infinity;
    for (let j = 0; j < cols; j++) if (keep(r, j)) max = Math.max(max, x[r * cols + j]!);
    let sum = 0;
    for (let j = 0; j < cols; j++) if (keep(r, j)) sum += Math.exp(x[r * cols + j]! - max);
    for (let j = 0; j < cols; j++) {
      out[r * cols + j] = keep(r, j) ? Math.exp(x[r * cols + j]! - max) / sum : 0;
    }
  }
  return out;
}

/** f32: exp and the division are a few ulp each. f16 output adds 2^-11
 *  relative. */
const TOLERANCE = { f32: { atol: 1e-7, rtol: 1e-5 }, f16: { atol: 1e-5, rtol: 2e-3 } };

type Dtype = 'f32' | 'f16';

/** Every case runs on both devices: the subgroup reduction where the main
 *  device has 32-wide subgroups, the 64-thread tree on the one without. */
const DEVICES = ['subgroups', 'tree'] as const;

/** Runs on the case's device, after the skips its route needs. */
function deviceFor(ctx: TestContext, reduction: (typeof DEVICES)[number], dtype: Dtype) {
  const dev = reduction === 'subgroups' ? main : withoutSubgroups;
  if (reduction === 'subgroups') requireFeature(ctx, dev.root, 'subgroups');
  if (dtype === 'f16') requireFeature(ctx, dev.root, 'shader-f16');
  return dev;
}

describe('softmax kernel', () => {
  // Routes are picked in gpu/dispatch/reduce.ts: vec4 when cols % 4 == 0,
  // scalar otherwise, each with the subgroup or the tree reduction.
  const SHAPES = [
    // vec4: cols % 4 == 0
    { name: 'one quad (vec4)', rows: 3, cols: 4 },
    { name: 'one quad per subgroup lane (vec4)', rows: 5, cols: 128 },
    { name: 'one quad past the 64-thread span (vec4)', rows: 2, cols: 260 },
    { name: 'one wide row (vec4)', rows: 1, cols: 32768 },
    // vec4: past 65535 rows the grid is 2-D and its last row runs past the end
    { name: 'past 65535 rows (vec4)', rows: 70000, cols: 4 },
    // scalar: cols % 4 != 0
    { name: '1×1 (scalar)', rows: 1, cols: 1 },
    { name: 'odd sizes (scalar)', rows: 3, cols: 7 },
    { name: 'one past the 64-thread span (scalar)', rows: 4, cols: 65 },
    { name: 'odd row width (scalar)', rows: 37, cols: 1001 },
    { name: 'past 65535 rows (scalar)', rows: 70000, cols: 3 },
  ];
  it.for(
    DEVICES.flatMap((reduction) =>
      (['f32', 'f16'] as const).flatMap((dtype) =>
        SHAPES.map((shape) => ({ reduction, dtype, ...shape })),
      ),
    ),
  )('$reduction $dtype $name: [$rows, $cols]', async (c, ctx) => {
    const dev = deviceFor(ctx, c.reduction, c.dtype);
    // Scaled so a row spans e^±8 and the running max has to rescale the sum.
    const scaled = randomF32(c.rows * c.cols, 1).map((v) => 8 * v);
    const x = c.dtype === 'f16' ? roundF16(scaled) : scaled;
    const got = await toArray(
      softmax(upload(dev.root, x, { ...matrix(c.rows, c.cols), dtype: c.dtype })),
      dev.ex,
    );
    expectClose(got, softmaxRef(x, c.rows, c.cols), TOLERANCE[c.dtype]);
  });

  // [1, 2, 3] → e^[0, 1, 2] / (1 + e + e²) = [0.09003057, 0.24472847,
  // 0.66524096], the same as torch.softmax. 1000 more only shifts the max.
  it.for(
    DEVICES.flatMap((reduction) => [
      // scalar: cols % 4 != 0
      { reduction, name: 'scalar', x: [1, 2, 3], want: [0.09003057, 0.24472847, 0.66524096] },
      // vec4: cols % 4 == 0; -1000 is e^-1003 below the max, 0 in f32
      {
        reduction,
        name: 'vec4, logits near 1000',
        x: [1001, 1002, 1003, -1000],
        want: [0.09003057, 0.24472847, 0.66524096, 0],
      },
    ]),
  )('$reduction $name matches torch.softmax', async ({ reduction, x, want }, ctx) => {
    const dev = deviceFor(ctx, reduction, 'f32');
    const got = await toArray(
      softmax(upload(dev.root, new Float32Array(x), [1, x.length])),
      dev.ex,
    );
    expectClose(got, want, { atol: 1e-7, rtol: 1e-6 });
  });

  // With segments, row r keeps keys in [start, end) of its segment and within
  // windowLeft before and windowRight after r. The rows are two segments,
  // [0, n/2) and [n/2, n).
  const MASKS = [
    { name: 'segments only', windowLeft: Infinity, windowRight: Infinity },
    { name: 'causal', windowLeft: Infinity, windowRight: 0 },
    { name: 'a sliding window', windowLeft: 2, windowRight: 1 },
    { name: 'only the diagonal', windowLeft: 0, windowRight: 0 },
  ];
  it.for(
    DEVICES.flatMap((reduction) =>
      (['f32', 'f16'] as const).flatMap((dtype) =>
        [
          // vec4: n % 4 == 0; n / 2 = 6 puts a segment edge inside a quad
          { route: 'vec4', n: 12 },
          // scalar: n % 4 != 0
          { route: 'scalar', n: 9 },
          // vec4, wider than a subgroup's 32 quads
          { route: 'vec4', n: 136 },
        ].flatMap((shape) => MASKS.map((mask) => ({ reduction, dtype, ...shape, ...mask }))),
      ),
    ),
  )('$reduction $dtype $route, masked by $name: [$n, $n]', async (c, ctx) => {
    const dev = deviceFor(ctx, c.reduction, c.dtype);
    const { n } = c;
    const half = Math.floor(n / 2);
    const scaled = randomF32(n * n, 2).map((v) => 8 * v);
    const x = c.dtype === 'f16' ? roundF16(scaled) : scaled;
    const segs = new Float32Array(2 * n);
    for (let r = 0; r < n; r++) {
      segs[2 * r] = r < half ? 0 : half;
      segs[2 * r + 1] = r < half ? half : n;
    }
    const got = await toArray(
      softmax(upload(dev.root, x, { ...matrix(n, n), dtype: c.dtype }), {
        segments: upload(dev.root, segs, [n, 2]),
        windowLeft: c.windowLeft,
        windowRight: c.windowRight,
      }),
      dev.ex,
    );
    const keep = (r: number, j: number) =>
      j >= segs[2 * r]! && j < segs[2 * r + 1]! && j >= r - c.windowLeft && j <= r + c.windowRight;
    expectClose(got, softmaxRef(x, n, n, keep), TOLERANCE[c.dtype]);
  });

  it('runs past 2^24 elements (vec4)', async () => {
    const rows = 4100;
    const cols = 4096;
    expect(rows * cols).toBeGreaterThan(2 ** 24);
    const x = randomF32(rows * cols, 3);
    const got = await toArray(softmax(upload(root, x, [rows, cols])), ex);
    expectClose(got, softmaxRef(x, rows, cols), TOLERANCE.f32);
  });

  it('rejects an empty input with an INVALID_ARGUMENT', () => {
    expect(() => upload(root, new Float32Array(0), [0, 4])).toThrow(
      expect.objectContaining({ code: 'INVALID_ARGUMENT' }),
    );
  });

  it('refuses an input that is not 2D', () => {
    const x = upload(root, new Float32Array(8), { elems: 8, dtype: 'f32', dims: [2, 2, 2] });
    expect(() => softmax(x)).toThrow(/expected a 2D f32 or f16 tensor/);
  });

  it('refuses masked scores that are not square', () => {
    const x = upload(root, new Float32Array(6), [2, 3]);
    const segs = upload(root, new Float32Array(4), [2, 2]);
    expect(() => softmax(x, { segments: segs })).toThrow(/needs square scores, got \[2, 3\]/);
  });

  it.for([
    { name: 'f16 segments', elems: 8, dtype: 'f16' },
    { name: 'segments of the wrong length', elems: 6, dtype: 'f32' },
  ] as const)('refuses $name', ({ elems, dtype }, ctx) => {
    if (dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    const x = upload(root, new Float32Array(16), [4, 4]);
    const segs = upload(root, new Float32Array(elems), { elems, dtype, dims: [elems / 2, 2] });
    expect(() => softmax(x, { segments: segs })).toThrow(/segments must be f32 \[4, 2\]/);
  });

  it('refuses a window that is not a non-negative integer', () => {
    const x = upload(root, new Float32Array(16), [4, 4]);
    const segs = upload(root, new Float32Array(8), [4, 2]);
    expect(() => softmax(x, { segments: segs, windowLeft: 1.5 })).toThrow(
      /windowLeft must be a non-negative integer or Infinity, got 1.5/,
    );
    expect(() => softmax(x, { segments: segs, windowRight: -1 })).toThrow(
      /windowRight must be a non-negative integer or Infinity, got -1/,
    );
  });
});
