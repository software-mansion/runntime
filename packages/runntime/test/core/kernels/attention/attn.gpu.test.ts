import { describe, expect, it, type TestContext } from 'vitest';
import { sdpa } from '../../../../src/core/graph/ops.ts';
import { matrix } from '../../../../src/core/graph/value.ts';
import { GRID_Y_STRIDE } from '../../../../src/core/kernels/common.ts';
import { gpu, gpuWithoutSubgroups, requireFeature, upload } from '../../../gpu.ts';
import { checkSdpa, SDPA_OPTIONS, type SdpaCase } from './checkSdpa.ts';

const main = await gpu();
const withoutSubgroups = await gpuWithoutSubgroups();
const { root } = main;

/** `main` is the production device, where a case for a subgroup route skips
 *  without 32-wide subgroups. `noSubgroups` is the device without them. */
type Device = 'main' | 'noSubgroups';

function deviceFor(ctx: TestContext, device: Device, dtype: 'f32' | 'f16', subgroups: boolean) {
  const dev = device === 'main' ? main : withoutSubgroups;
  if (subgroups) requireFeature(ctx, dev.root, 'subgroups');
  if (dtype === 'f16') requireFeature(ctx, dev.root, 'shader-f16');
  return dev;
}

// Routes are picked in gpu/dispatch/attention.ts. qLen > 1, and headDim 40
// is not a multiple of 16, so flash never takes these.
const ROUTES = [
  // rowsSubgroupSplit: f32, subgroups, row-heads <= 65535, the split scratch fits
  { route: 'rowsSubgroupSplit', device: 'main', dtype: 'f32', subgroups: true, t: 37 },
  // rowsSubgroupSplit: as above, 300 keys make 4 chunks of 96
  { route: 'rowsSubgroupSplit', device: 'main', dtype: 'f32', subgroups: true, t: 300 },
  // rowsSplit: f32, no subgroups, more than 96 keys
  { route: 'rowsSplit', device: 'noSubgroups', dtype: 'f32', subgroups: false, t: 200 },
  // rows: f32, no subgroups, at most 96 keys
  { route: 'rows', device: 'noSubgroups', dtype: 'f32', subgroups: false, t: 37 },
  // rows: f16 never splits
  { route: 'rows', device: 'main', dtype: 'f16', subgroups: false, t: 200 },
  { route: 'rows', device: 'noSubgroups', dtype: 'f16', subgroups: false, t: 37 },
] as const;

describe('attn kernels', () => {
  it.for(ROUTES.flatMap((r) => SDPA_OPTIONS.map((o) => ({ ...r, ...o }))))(
    '$route $dtype on $device with $name: t = $t, headDim 40',
    async (c, ctx) => {
      const dev = deviceFor(ctx, c.device, c.dtype, c.subgroups);
      const shape = { qLen: c.t, kvLen: c.t, qHeads: 4, kvHeads: 2, headDim: 40 };
      await checkSdpa(dev, { ...shape, ...c.opts }, c.dtype);
    },
  );

  const SHAPES: {
    name: string;
    device: Device;
    dtype: 'f32' | 'f16';
    subgroups: boolean;
    c: SdpaCase;
  }[] = [
    // rowsSubgroupSplit: one column per lane, the other 31 lanes hold 0
    {
      name: 'headDim 1 (rowsSubgroupSplit)',
      device: 'main',
      dtype: 'f32',
      subgroups: true,
      c: { qLen: 5, kvLen: 5, qHeads: 2, kvHeads: 1, headDim: 1 },
    },
    // rowsSubgroupSplit: headDim 200 is 7 columns per lane, the last one partial
    {
      name: 'headDim 200 (rowsSubgroupSplit)',
      device: 'main',
      dtype: 'f32',
      subgroups: true,
      c: { qLen: 9, kvLen: 9, qHeads: 2, kvHeads: 2, headDim: 200, windowRight: 0 },
    },
    // rowsSubgroupSplit: 1700 keys pass 16 chunks of 96, so the chunks grow
    {
      name: 'cross-attention past the chunk cap (rowsSubgroupSplit)',
      device: 'main',
      dtype: 'f32',
      subgroups: true,
      c: { qLen: 5, kvLen: 1700, qHeads: 2, kvHeads: 1, headDim: 8 },
    },
    // rowsSubgroupSplit: queries at the end of a longer cache
    {
      name: 'qPosOffset into a longer cache (rowsSubgroupSplit)',
      device: 'main',
      dtype: 'f32',
      subgroups: true,
      c: { qLen: 6, kvLen: 20, qHeads: 2, kvHeads: 1, headDim: 8, windowRight: 0, qPosOffset: 14 },
    },
    // rowsSubgroupSplit: rows past the keys with windowLeft 0 see none and come out 0
    {
      name: 'rows that see no key, with sinks (rowsSubgroupSplit)',
      device: 'main',
      dtype: 'f32',
      subgroups: true,
      c: { qLen: 6, kvLen: 4, qHeads: 2, kvHeads: 1, headDim: 8, windowLeft: 0, sinks: true },
    },
    // rowsSplit: the same empty rows on the per-thread split
    {
      name: 'rows that see no key, with sinks (rowsSplit)',
      device: 'noSubgroups',
      dtype: 'f32',
      subgroups: false,
      c: { qLen: 120, kvLen: 100, qHeads: 2, kvHeads: 1, headDim: 8, windowLeft: 0, sinks: true },
    },
    // rows: the same empty rows on the rows kernel
    {
      name: 'rows that see no key, with sinks (rows)',
      device: 'noSubgroups',
      dtype: 'f32',
      subgroups: false,
      c: { qLen: 6, kvLen: 4, qHeads: 2, kvHeads: 1, headDim: 8, windowLeft: 0, sinks: true },
    },
    // rows: 96 keys is the most rows takes without subgroups
    {
      name: 'at the 96-key chunk (rows)',
      device: 'noSubgroups',
      dtype: 'f32',
      subgroups: false,
      c: { qLen: 96, kvLen: 96, qHeads: 2, kvHeads: 1, headDim: 8, windowRight: 0 },
    },
    // rowsSplit: 97 keys is one past it
    {
      name: 'one key past the 96-key chunk (rowsSplit)',
      device: 'noSubgroups',
      dtype: 'f32',
      subgroups: false,
      c: { qLen: 97, kvLen: 97, qHeads: 2, kvHeads: 1, headDim: 8, windowRight: 0 },
    },
    // rowsSplit: 1700 keys pass 16 chunks of 96, so the chunks grow
    {
      name: 'cross-attention past the chunk cap (rowsSplit)',
      device: 'noSubgroups',
      dtype: 'f32',
      subgroups: false,
      c: { qLen: 5, kvLen: 1700, qHeads: 2, kvHeads: 1, headDim: 8 },
    },
    // rows: maxSegment 90 sizes the split from the segment, at most 96 keys
    {
      name: 'segments, maxSegment under the chunk (rows)',
      device: 'noSubgroups',
      dtype: 'f32',
      subgroups: false,
      c: {
        qLen: 180,
        kvLen: 180,
        qHeads: 2,
        kvHeads: 1,
        headDim: 8,
        segments: true,
        maxSegment: 90,
      },
    },
    // rowsSplit: maxSegment 100 is past it
    {
      name: 'segments, maxSegment past the chunk (rowsSplit)',
      device: 'noSubgroups',
      dtype: 'f32',
      subgroups: false,
      c: {
        qLen: 200,
        kvLen: 200,
        qHeads: 2,
        kvHeads: 1,
        headDim: 8,
        segments: true,
        maxSegment: 100,
      },
    },
    // rows: the widest head sdpa takes with more than one query row
    {
      name: 'headDim 512 (rows)',
      device: 'noSubgroups',
      dtype: 'f32',
      subgroups: false,
      c: { qLen: 7, kvLen: 7, qHeads: 1, kvHeads: 1, headDim: 512, windowRight: 0 },
    },
    {
      name: 'f16 headDim 512 (rows)',
      device: 'noSubgroups',
      dtype: 'f16',
      subgroups: false,
      c: { qLen: 7, kvLen: 7, qHeads: 1, kvHeads: 1, headDim: 512, windowRight: 0 },
    },
    // rows: 70000 row-heads is past the subgroup split's 65535 workgroups, and
    // 8 keys are under the 96-key chunk
    {
      name: 'past 65535 row-heads, few keys (rows)',
      device: 'main',
      dtype: 'f32',
      subgroups: false,
      c: { qLen: 70000, kvLen: 8, qHeads: 1, kvHeads: 1, headDim: 4 },
    },
    // rowsSplit: the same with 100 keys
    {
      name: 'past 65535 row-heads, more than 96 keys (rowsSplit)',
      device: 'main',
      dtype: 'f32',
      subgroups: false,
      c: { qLen: 66000, kvLen: 100, qHeads: 1, kvHeads: 1, headDim: 4 },
    },
    // rows: a split scratch of 20000 row-heads × 16 chunks × 128 floats is past
    // maxStorageBufferBindingSize, so neither split route runs
    ...(['main', 'noSubgroups'] as const).map((device) => ({
      name: 'split scratch past the binding limit (rows)',
      device,
      dtype: 'f32' as const,
      subgroups: false,
      c: {
        qLen: 20000,
        kvLen: 20000,
        qHeads: 1,
        kvHeads: 1,
        headDim: 126,
        windowLeft: 2,
        windowRight: 0,
      },
    })),
    // batch: past GRID_Y_STRIDE row-heads the row kernels' 1-D grid runs out
    ...(
      [
        ['main', 'f32'],
        ['main', 'f16'],
        ['noSubgroups', 'f32'],
      ] as const
    ).map(([device, dtype]) => ({
      name: 'past GRID_Y_STRIDE row-heads (batch)',
      device,
      dtype,
      subgroups: false,
      c: { qLen: GRID_Y_STRIDE + 5, kvLen: 4, qHeads: 1, kvHeads: 1, headDim: 2 },
    })),
    // batch: with GQA, a window, sinks and segments on the serial kernel
    {
      name: 'past GRID_Y_STRIDE row-heads with every option (batch)',
      device: 'noSubgroups',
      dtype: 'f32',
      subgroups: false,
      c: {
        qLen: GRID_Y_STRIDE / 2 + 2,
        kvLen: GRID_Y_STRIDE / 2 + 2,
        qHeads: 2,
        kvHeads: 1,
        headDim: 1,
        windowLeft: 2,
        windowRight: 0,
        sinks: true,
        segments: true,
      },
    },
  ];
  it.for(SHAPES)('$dtype $name on $device', async ({ device, dtype, subgroups, c }, ctx) => {
    await checkSdpa(deviceFor(ctx, device, dtype, subgroups), c, dtype);
  });

  // Known bug: the batch kernel splits its flat index into row and head with
  // d.u32(idx / qHeads), which TypeGPU compiles to u32(f32(idx) /
  // f32(qHeads)). f32 counts exactly only up to 2^24, so past 2^24 row-heads
  // rows come out wrong. Turn into a plain `it` with the fix.
  // batch: past GRID_Y_STRIDE row-heads
  it.fails('past 2^24 row-heads (batch)', () =>
    checkSdpa(withoutSubgroups, { qLen: 2 ** 24 + 3, kvLen: 2, qHeads: 1, kvHeads: 1, headDim: 1 }),
  );

  it('rejects qLen = 0 with an INVALID_ARGUMENT', () => {
    expect(() => upload(root, new Float32Array(0), [0, 8])).toThrow(
      expect.objectContaining({ code: 'INVALID_ARGUMENT' }),
    );
  });

  const heads = { qHeads: 2, kvHeads: 1, headDim: 8, windowLeft: Infinity, windowRight: 0 };
  it.each([
    {
      name: 'a head wider than 512',
      q: [4, 2 * 520],
      kv: [4, 520],
      opts: { ...heads, headDim: 520 },
      err: /headDim 520 exceeds the kernel's max \(512\)/,
    },
    {
      name: 'query heads that kv heads do not divide',
      q: [4, 24],
      kv: [4, 16],
      opts: { ...heads, qHeads: 3, kvHeads: 2 },
      err: /qHeads 3 not divisible by kvHeads 2/,
    },
    {
      name: 'queries past the keys under a bounded right window',
      q: [4, 16],
      kv: [3, 8],
      opts: heads,
      err: /qPosOffset\+qLen 4 exceeds key rows kvLen 3/,
    },
    {
      name: 'a window that is not an integer',
      q: [4, 16],
      kv: [4, 8],
      opts: { ...heads, windowLeft: 1.5 },
      err: /windowLeft 1.5 must be a non-negative integer or Infinity/,
    },
    {
      name: 'maxSegment without segments',
      q: [4, 16],
      kv: [4, 8],
      opts: { ...heads, maxSegment: 2 },
      err: /maxSegment needs segments/,
    },
  ])('refuses $name', ({ q, kv, opts, err }) => {
    const mat = (dims: number[]) =>
      upload(root, new Float32Array(dims[0]! * dims[1]!), [dims[0]!, dims[1]!]);
    expect(() => sdpa(mat(q), mat(kv), mat(kv), opts)).toThrow(err);
  });

  it('refuses k in another dtype than q', (ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const q = upload(root, new Float32Array(32), [2, 16]);
    const k = upload(root, new Float32Array(16), { ...matrix(2, 8), dtype: 'f16' });
    const v = upload(root, new Float32Array(16), [2, 8]);
    expect(() => sdpa(q, k, v, heads)).toThrow(/q, k and v must share a dtype, got f32\/f16\/f32/);
  });
});
