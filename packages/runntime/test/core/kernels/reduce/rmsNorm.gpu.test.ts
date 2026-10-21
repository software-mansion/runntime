import { describe, expect, it, type TestContext } from 'vitest';
import { toArray } from '../../../../src/core/gpu/eval.ts';
import { rmsNorm } from '../../../../src/core/graph/ops.ts';
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

type Dtype = 'f32' | 'f16';

/** Every case runs on both devices: the subgroup sum where the main device
 *  has 32-wide subgroups, the 64-thread tree on the one without. */
const DEVICES = ['subgroups', 'tree'] as const;

/** Runs on the case's device, after the skips its route needs. */
function deviceFor(ctx: TestContext, reduction: (typeof DEVICES)[number], dtype: Dtype) {
  const dev = reduction === 'subgroups' ? main : withoutSubgroups;
  if (reduction === 'subgroups') requireFeature(ctx, dev.root, 'subgroups');
  if (dtype === 'f16') requireFeature(ctx, dev.root, 'shader-f16');
  return dev;
}

/** (residual + x / sqrt(mean(x²) + eps) · weight) · scale over each run of
 *  `group` columns, in f64. A weight `group` long is shared by every run, one
 *  `cols` long gives each run its own slice. */
function rmsNormRef(
  x: Float32Array,
  weight: Float32Array,
  rows: number,
  cols: number,
  { eps = 1e-6, group = cols, residual = undefined as Float32Array | undefined, scale = 1 } = {},
): Float64Array {
  const out = new Float64Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    for (let g = 0; g < cols; g += group) {
      let sumSq = 0;
      for (let j = 0; j < group; j++) sumSq += x[r * cols + g + j]! ** 2;
      const inv = 1 / Math.sqrt(sumSq / group + eps);
      for (let j = 0; j < group; j++) {
        const i = r * cols + g + j;
        const w = weight.length === group ? weight[j]! : weight[g + j]!;
        out[i] = (x[i]! * inv * w + (residual ? residual[i]! : 0)) * scale;
      }
    }
  }
  return out;
}

/** f32: a long sum of squares and an inverseSqrt, a few ulp each. f16 output
 *  adds 2^-11 relative. */
const TOLERANCE = { f32: { atol: 1e-6, rtol: 1e-5 }, f16: { atol: 1e-3, rtol: 2e-3 } };

interface Case {
  rows: number;
  cols: number;
  dtype: Dtype;
  eps?: number;
  group?: number;
  weightLen?: number;
  residual?: boolean;
  scale?: number;
}

async function checkRmsNorm(dev: Awaited<ReturnType<typeof gpu>>, c: Case): Promise<void> {
  const { rows, cols, dtype, eps, group = cols, scale } = c;
  const round = dtype === 'f16' ? roundF16 : (v: Float32Array) => v;
  const x = round(randomF32(rows * cols, 1));
  const weight = round(randomF32(c.weightLen ?? group, 2));
  const residual = c.residual ? round(randomF32(rows * cols, 3)) : undefined;
  const meta = (r: number, n: number) => ({ ...matrix(r, n), dtype });
  const got = await toArray(
    rmsNorm(
      upload(dev.root, x, meta(rows, cols)),
      upload(dev.root, weight, meta(1, weight.length)),
      eps,
      {
        group: c.group,
        residual: residual && upload(dev.root, residual, meta(rows, cols)),
        scale,
      },
    ),
    dev.ex,
  );
  expectClose(
    got,
    rmsNormRef(x, weight, rows, cols, { eps, group, residual, scale }),
    TOLERANCE[dtype],
  );
}

describe('rmsNorm kernel', () => {
  // Routes are picked in gpu/dispatch/reduce.ts: vec4 when the group (the
  // row width without one) is a multiple of 4, scalar otherwise, each with
  // the subgroup or the tree sum.
  const SHAPES = [
    // vec4: cols % 4 == 0
    { name: 'one quad (vec4)', rows: 3, cols: 4 },
    { name: 'one quad per subgroup lane (vec4)', rows: 5, cols: 128 },
    { name: 'one quad past the 64-thread span (vec4)', rows: 2, cols: 260 },
    { name: 'a wide row (vec4)', rows: 2, cols: 16384 },
    // vec4: past 65535 rows the grid is 2-D and its last row runs past the end
    { name: 'past 65535 rows (vec4)', rows: 70000, cols: 4 },
    // scalar: cols % 4 != 0
    { name: '1×1 (scalar)', rows: 1, cols: 1 },
    { name: 'odd sizes (scalar)', rows: 3, cols: 7 },
    { name: 'one past the 64-thread span (scalar)', rows: 4, cols: 65 },
    { name: 'odd row width (scalar)', rows: 37, cols: 642 },
    { name: 'past 65535 rows (scalar)', rows: 70000, cols: 3 },
  ];
  it.for(
    DEVICES.flatMap((reduction) =>
      (['f32', 'f16'] as const).flatMap((dtype) =>
        SHAPES.map((shape) => ({ reduction, dtype, ...shape })),
      ),
    ),
  )('$reduction $dtype $name: [$rows, $cols]', async (c, ctx) => {
    await checkRmsNorm(deviceFor(ctx, c.reduction, c.dtype), c);
  });

  // Every option on both routes. The group picks the route: vec4 runs groups
  // of 8 in rows of 48, scalar groups of 6 in rows of 42.
  const OPTIONS: {
    name: string;
    eps?: number;
    residual?: boolean;
    scale?: number;
    grouped?: boolean;
    sliced?: boolean;
  }[] = [
    { name: 'eps 1e-2', eps: 1e-2 },
    { name: 'a residual', residual: true },
    { name: 'scale 0.5', scale: 0.5 },
    { name: 'a residual and scale 2', residual: true, scale: 2 },
    { name: 'groups sharing one weight', grouped: true },
    { name: 'groups, a weight slice per group', grouped: true, sliced: true },
    {
      name: 'groups with a residual and scale',
      grouped: true,
      sliced: true,
      residual: true,
      scale: 0.25,
    },
  ];
  it.for(
    DEVICES.flatMap((reduction) =>
      (['f32', 'f16'] as const).flatMap((dtype) =>
        [
          // vec4: group % 4 == 0
          { route: 'vec4', rows: 5, cols: 48, group: 8 },
          // scalar: group % 4 != 0
          { route: 'scalar', rows: 5, cols: 42, group: 6 },
        ].flatMap((shape) => OPTIONS.map((o) => ({ reduction, dtype, ...shape, ...o }))),
      ),
    ),
  )('$reduction $dtype $route with $name: [$rows, $cols]', async (c, ctx) => {
    await checkRmsNorm(deviceFor(ctx, c.reduction, c.dtype), {
      ...c,
      group: c.grouped ? c.group : undefined,
      weightLen: c.sliced ? c.cols : undefined,
    });
  });

  // x = [3, 4]: mean(x²) = 12.5, so out = x / sqrt(12.5 + eps) · weight.
  // With eps = 0.5 that is [3, 4] / sqrt(13) · [1, 2] = [0.83205029, 2.21880078].
  it('matches the formula worked out by hand', async () => {
    const x = upload(root, new Float32Array([3, 4]), [1, 2]);
    const w = upload(root, new Float32Array([1, 2]), [1, 2]);
    expectClose(await toArray(rmsNorm(x, w, 0.5), ex), [0.83205029, 2.21880078], {
      atol: 0,
      rtol: 1e-6,
    });
  });

  // x² of 300 overflows f16 (max 65504); the sum stays in f32.
  it('f16 with values whose square overflows f16', async (ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const x = new Float32Array([300, -300, 300, 1]);
    const w = new Float32Array([1, 1, 1, 1]);
    const f16 = { ...matrix(1, 4), dtype: 'f16' } as const;
    const got = await toArray(rmsNorm(upload(root, x, f16), upload(root, w, f16)), ex);
    expectClose(got, rmsNormRef(x, w, 1, 4), TOLERANCE.f16);
  });

  it('runs past 2^24 elements (vec4)', async () => {
    const rows = 4100;
    const cols = 4096;
    expect(rows * cols).toBeGreaterThan(2 ** 24);
    await checkRmsNorm(main, { rows, cols, dtype: 'f32' });
  });

  it('rejects an empty input with an INVALID_ARGUMENT', () => {
    expect(() => upload(root, new Float32Array(0), [0, 4])).toThrow(
      expect.objectContaining({ code: 'INVALID_ARGUMENT' }),
    );
  });

  it.for([
    {
      name: 'a group that does not divide the row',
      group: 5,
      wl: 5,
      eps: 1e-6,
      err: /group 5 must divide the row width 12/,
    },
    {
      name: 'a weight neither group nor row long',
      group: 4,
      wl: 6,
      eps: 1e-6,
      err: /weight must be f32 with 4 or 12 elems, got \(6,f32\)/,
    },
    { name: 'eps 0', group: 12, wl: 12, eps: 0, err: /eps must be positive, got 0/ },
  ])('refuses $name', ({ group, wl, eps, err }) => {
    const x = upload(root, new Float32Array(24), [2, 12]);
    const w = upload(root, new Float32Array(wl), [1, wl]);
    expect(() => rmsNorm(x, w, eps, { group })).toThrow(err);
  });

  it('refuses an f16 weight on f32 x', (ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const x = upload(root, new Float32Array(8), [2, 4]);
    const w = upload(root, new Float32Array(4), { ...matrix(1, 4), dtype: 'f16' });
    expect(() => rmsNorm(x, w)).toThrow(/weight must be f32 with 4 elems, got \(4,f16\)/);
  });

  it('refuses a residual of another shape', () => {
    const x = upload(root, new Float32Array(8), [2, 4]);
    const w = upload(root, new Float32Array(4), [1, 4]);
    const residual = upload(root, new Float32Array(4), [1, 4]);
    expect(() => rmsNorm(x, w, 1e-6, { residual })).toThrow(/residual must be f32 \[2, 4\]/);
  });
});
