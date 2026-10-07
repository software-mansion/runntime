import { describe, expect, it } from 'vitest';
import {
  SSM_DIRS,
  SSM_STATE,
  ssmScanMerge,
  ssmScanProject,
  ssmSelectiveScan,
} from '../../../../src/core/graph/ops.ts';
import { input, meta } from '../values.ts';

// A [C=4, H=2, W=3] map, scanned in 4 directions with 8 state values
const C = 4;
const H = 2;
const W = 3;
const P = H * W;
const RANK = 2;
const x = input([C, H, W]);

describe('ssmScanProject', () => {
  const xw = input([SSM_DIRS * (RANK + 2 * SSM_STATE) * C]);
  const dw = input([SSM_DIRS * C * RANK]);

  it('gives delta, B, C and dt-scratch rows per direction, one column per pixel', () => {
    const c = ssmScanProject(x, xw, dw, { rank: RANK });
    expect(c.shape).toEqual(meta([SSM_DIRS * (C + 2 * SSM_STATE + RANK), P]));
  });

  it.each([
    { name: 'rank 0', opts: { rank: 0 }, error: /rank must be a positive integer, got 0/ },
    {
      name: 'a projection weight of the wrong size',
      xwIn: input([10]),
      error: /xw must be f32 with 288 elems, got 10/,
    },
    {
      name: 'a dt weight of the wrong size',
      dwIn: input([10]),
      error: /dw must be f32 with 32 elems, got 10/,
    },
  ])('rejects $name', ({ opts = { rank: RANK }, xwIn = xw, dwIn = dw, error }) => {
    expect(() => ssmScanProject(x, xwIn, dwIn, opts)).toThrow(error);
  });

  it('needs an f32 input', () => {
    expect(() => ssmScanProject(input([C, H, W], 'f16'), xw, dw, { rank: RANK })).toThrow(
      /expected a 3D f32 tensor/,
    );
  });
});

describe('ssmSelectiveScan', () => {
  const proj = input([SSM_DIRS * (C + 2 * SSM_STATE + RANK), P]);
  const a = input([SSM_DIRS * C * SSM_STATE]);
  const dSkip = input([SSM_DIRS * C]);
  const deltaBias = input([SSM_DIRS * C]);

  it('gives one row per channel and direction', () => {
    const c = ssmSelectiveScan(x, proj, a, dSkip, deltaBias);
    expect(c.shape).toEqual(meta([SSM_DIRS * C, P]));
    expect(c.pending?.attrs).toEqual([C, H, W]);
  });

  it.each([
    {
      name: 'a projection too small for the map',
      args: { proj: input([4, P]) },
      error: /proj must be a scan-project output/,
    },
    { name: 'A of the wrong size', args: { a: input([4]) }, error: /a must be f32 with 128 elems/ },
    {
      name: 'D of the wrong size',
      args: { dSkip: input([4]) },
      error: /dSkip must be f32 with 16 elems/,
    },
    {
      name: 'a delta bias of the wrong size',
      args: { deltaBias: input([4]) },
      error: /deltaBias must be f32 with 16 elems/,
    },
  ])('rejects $name', ({ args, error }) => {
    const all = { proj, a, dSkip, deltaBias, ...args };
    expect(() => ssmSelectiveScan(x, all.proj, all.a, all.dSkip, all.deltaBias)).toThrow(error);
  });
});

describe('ssmScanMerge', () => {
  it('sums the directions back into [C, H, W]', () => {
    const c = ssmScanMerge(input([SSM_DIRS * C, P]), { c: C, h: H, w: W });
    expect(c.shape).toEqual(meta([C, H, W]));
  });

  it.each([
    {
      name: 'a channel count that does not match',
      d: input([SSM_DIRS * C, P]),
      opts: { c: 5, h: H, w: W },
    },
    {
      name: 'a pixel count that does not match',
      d: input([SSM_DIRS * C, P]),
      opts: { c: C, h: 3, w: 3 },
    },
    { name: 'a flat input', d: input([SSM_DIRS * C * P]), opts: { c: C, h: H, w: W } },
  ])('rejects $name', ({ d, opts }) => {
    expect(() => ssmScanMerge(d, opts)).toThrow(/input must be f32 \[4·/);
  });
});
