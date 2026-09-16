/** SS2D selective-scan ops (VMamba/DepthART): project → scan → merge. */

import { pending, Value } from '../value.ts';
import { SSM_DIRS, SSM_STATE } from '../../kernels/ssm.ts';
import { want3d } from './shared.ts';

export { SSM_DIRS, SSM_STATE };

/** SS2D scan projection: projects each pixel to [dtRank | B | C], expands dt to
 *  all channels, and emits one matrix of delta, B and C rows followed by dt
 *  scratch rows that consumers never read. */
export function ssmScanProject(x: Value, xw: Value, dw: Value, opts: { rank: number }): Value {
  const [c, h, w] = want3d('ssmScanProject', x, ['f32']);
  const { rank } = opts;
  if (!Number.isInteger(rank) || rank < 1) {
    throw new Error(`ssmScanProject: rank must be a positive integer, got ${rank}`);
  }
  const xwElems = SSM_DIRS * (rank + 2 * SSM_STATE) * c;
  const dwElems = SSM_DIRS * c * rank;
  if (xw.shape.dtype !== 'f32' || xw.shape.elems !== xwElems) {
    throw new Error(`ssmScanProject: xw must be f32 with ${xwElems} elems, got ${xw.shape.elems}`);
  }
  if (dw.shape.dtype !== 'f32' || dw.shape.elems !== dwElems) {
    throw new Error(`ssmScanProject: dw must be f32 with ${dwElems} elems, got ${dw.shape.elems}`);
  }
  const rows = SSM_DIRS * (c + 2 * SSM_STATE + rank);
  return pending(
    { elems: rows * h * w, dtype: 'f32', dims: [rows, h * w] },
    'ssmScanProject',
    [x, xw, dw],
    undefined,
    [rank],
  );
}

/** SS2D selective-scan recurrence over the four directions, giving outputs in
 *  traversal order for ssmScanMerge to restore. */
export function ssmSelectiveScan(
  x: Value,
  proj: Value,
  a: Value,
  dSkip: Value,
  deltaBias: Value,
): Value {
  const [c, h, w] = want3d('ssmSelectiveScan', x, ['f32']);
  const p = h * w;
  if (proj.shape.dtype !== 'f32' || proj.shape.elems < SSM_DIRS * (c + 2 * SSM_STATE) * p) {
    throw new Error(
      `ssmSelectiveScan: proj must be a scan-project output with at least [4C+8N, P] rows, got ${proj.shape.elems} elems`,
    );
  }
  for (const [t, name, elems] of [
    [a, 'a', SSM_DIRS * c * SSM_STATE],
    [dSkip, 'dSkip', SSM_DIRS * c],
    [deltaBias, 'deltaBias', SSM_DIRS * c],
  ] as const) {
    if (t.shape.dtype !== 'f32' || t.shape.elems !== elems) {
      throw new Error(`ssmSelectiveScan: ${name} must be f32 with ${elems} elems`);
    }
  }
  return pending(
    { elems: SSM_DIRS * c * p, dtype: 'f32', dims: [SSM_DIRS * c, p] },
    'ssmSelectiveScan',
    [x, proj, a, dSkip, deltaBias],
    undefined,
    [c, h, w],
  );
}

/** Sums the four directional scan outputs back into spatial order. */
export function ssmScanMerge(directional: Value, opts: { c: number; h: number; w: number }): Value {
  const { c, h, w } = opts;
  const dims = directional.shape.dims;
  if (
    directional.shape.dtype !== 'f32' ||
    !dims ||
    dims.length !== 2 ||
    dims[0] !== SSM_DIRS * c ||
    dims[1] !== h * w
  ) {
    throw new Error(
      `ssmScanMerge: input must be f32 [${SSM_DIRS}·${c}, ${h * w}], got ${JSON.stringify(dims)}`,
    );
  }
  return pending(
    { elems: c * h * w, dtype: 'f32', dims: [c, h, w] },
    'ssmScanMerge',
    [directional],
    undefined,
    [c, h, w],
  );
}
