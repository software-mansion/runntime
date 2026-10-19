import { describe, it } from 'vitest';
import { MAX_WORKGROUPS_PER_DIM } from '../../../../src/core/kernels/common.ts';
import { gpu, gpuWithoutSubgroups, requireFeature } from '../../../gpu.ts';
import { checkSdpa, SDPA_OPTIONS, type SdpaCase } from './checkSdpa.ts';

const main = await gpu();
const withoutSubgroups = await gpuWithoutSubgroups();
const { root } = main;

/** Rows per flash workgroup. */
const TILE_Q = 16;

describe('attnFlash kernel', () => {
  // flash: qLen > 1, subgroups, headDim % 16 == 0. Picked in
  // gpu/dispatch/attention.ts before every other prefill route.

  // 8 lanes per row when headDim % 32 == 0, 4 otherwise. Keys per block: 32,
  // or 16 for f32 heads wider than 32, fewer past 128.
  const HEAD_DIMS = [
    { headDim: 16, lanes: 4 },
    { headDim: 32, lanes: 8 },
    { headDim: 48, lanes: 4 },
    { headDim: 64, lanes: 8 },
    { headDim: 128, lanes: 8 },
    { headDim: 256, lanes: 8 },
    { headDim: 512, lanes: 8 },
  ];
  it.for(
    (['f32', 'f16'] as const).flatMap((dtype) =>
      HEAD_DIMS.flatMap((h) =>
        [
          // 37 rows leave a partial last workgroup, 37 keys a partial last block
          { name: 'odd sizes', qLen: 37, kvLen: 37 },
          { name: 'exactly one workgroup of rows', qLen: TILE_Q, kvLen: TILE_Q },
          { name: 'one row past a workgroup', qLen: TILE_Q + 1, kvLen: TILE_Q + 1 },
          { name: 'two rows', qLen: 2, kvLen: 2 },
        ].map((shape) => ({ dtype, ...h, ...shape })),
      ),
    ),
  )(
    '$dtype headDim $headDim ($lanes lanes per row), $name: qLen $qLen',
    async ({ dtype, headDim, qLen, kvLen }, ctx) => {
      requireFeature(ctx, root, 'subgroups');
      if (dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
      await checkSdpa(main, { qLen, kvLen, qHeads: 2, kvHeads: 1, headDim, windowRight: 0 }, dtype);
    },
  );

  // Every option, with 4 and 8 lanes per row.
  it.for(
    (['f32', 'f16'] as const).flatMap((dtype) =>
      [16, 64].flatMap((headDim) => SDPA_OPTIONS.map((o) => ({ dtype, headDim, ...o }))),
    ),
  )('$dtype headDim $headDim with $name: t = 37', async ({ dtype, headDim, opts }, ctx) => {
    requireFeature(ctx, root, 'subgroups');
    if (dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    const shape = { qLen: 37, kvLen: 37, qHeads: 4, kvHeads: 2, headDim };
    await checkSdpa(main, { ...shape, ...opts }, dtype);
  });

  const SHAPES: { name: string; c: SdpaCase }[] = [
    {
      name: 'cross-attention over many key blocks',
      c: { qLen: 20, kvLen: 1000, qHeads: 2, kvHeads: 1, headDim: 32 },
    },
    {
      name: 'qPosOffset into a longer cache',
      c: {
        qLen: 20,
        kvLen: 70,
        qHeads: 2,
        kvHeads: 1,
        headDim: 32,
        windowRight: 0,
        qPosOffset: 50,
      },
    },
    // Rows in one workgroup see different key ranges, past the block size
    {
      name: 'a sliding window wider than a key block',
      c: {
        qLen: 100,
        kvLen: 100,
        qHeads: 2,
        kvHeads: 1,
        headDim: 64,
        windowLeft: 40,
        windowRight: 0,
      },
    },
    // Rows past the keys with windowLeft 0 see none and come out 0
    {
      name: 'rows that see no key, with sinks',
      c: { qLen: 40, kvLen: 20, qHeads: 2, kvHeads: 1, headDim: 16, windowLeft: 0, sinks: true },
    },
    {
      name: 'segments with maxSegment',
      c: {
        qLen: 64,
        kvLen: 64,
        qHeads: 2,
        kvHeads: 1,
        headDim: 16,
        segments: true,
        maxSegment: 32,
      },
    },
  ];
  it.for((['f32', 'f16'] as const).flatMap((dtype) => SHAPES.map((s) => ({ dtype, ...s }))))(
    '$dtype $name',
    async ({ dtype, c }, ctx) => {
      requireFeature(ctx, root, 'subgroups');
      if (dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
      await checkSdpa(main, c, dtype);
    },
  );

  // Just past flash's conditions, so a row kernel runs on the same shapes.
  it.for([
    // rowsSplit: headDim % 16 == 0 but no subgroups, more than 96 keys
    {
      name: 'no subgroups (rowsSplit)',
      c: { qLen: 100, kvLen: 100, qHeads: 2, kvHeads: 1, headDim: 64 },
    },
    // rows: no subgroups, at most 96 keys
    { name: 'no subgroups (rows)', c: { qLen: 37, kvLen: 37, qHeads: 2, kvHeads: 1, headDim: 64 } },
  ])('$name: headDim 64', ({ c }) => checkSdpa(withoutSubgroups, c));

  // Known bug: flash dispatches [ceil(qLen / 16), qHeads] workgroups with no
  // 2-D split, so past 65535 × 16 rows its grid passes the per-dimension
  // limit: the dispatch fails validation and the output comes back all 0,
  // with no error. The batch fallback only checks row-heads past
  // GRID_Y_STRIDE. Turn into `it` with the fix.
  // flash: qLen > 1, subgroups, headDim % 16 == 0
  it.fails('past 65535 workgroups of rows', async (ctx) => {
    requireFeature(ctx, root, 'subgroups');
    const qLen = MAX_WORKGROUPS_PER_DIM * TILE_Q + 1;
    await checkSdpa(main, { qLen, kvLen: 4, qHeads: 1, kvHeads: 1, headDim: 16 });
  });
});
