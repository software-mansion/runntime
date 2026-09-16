/** Input-validation helpers shared by the op modules. */

import type { EagerDtype, Value } from '../value.ts';

export { ACT_CODE, type ActName, type SlotAct } from '../../kernels/activations.ts';

/** hwc4-aware ops branch on this; everything else refuses a tagged Value,
 *  since reading hwc4 bytes row-major produces silent garbage. */
export function isHwc4(x: Value): boolean {
  return x.shape.layout === 'hwc4';
}

function rejectHwc4(name: string, x: Value): void {
  if (x.shape.layout !== undefined) {
    throw new Error(
      `${name}: input is stored '${x.shape.layout}' — this op reads row-major storage; convert with toChw() first`,
    );
  }
}

/** Requires hwc4 storage, returning the logical [C, H, W]. */
export function wantHwc4(name: string, x: Value): [number, number, number] {
  if (!isHwc4(x)) {
    throw new Error(
      `${name}: input must be hwc4-stored (got layout=${x.shape.layout ?? 'row-major'}, dtype=${x.shape.dtype}) — convert with toHwc4() at the model boundary`,
    );
  }
  const [c, h, w] = x.shape.dims! as [number, number, number];
  return [c, h, w];
}

/** The float storage dtypes an activation may carry. An op whose kernels bind
 *  one dtype passes that one alone instead. */
export const FLOAT_DTYPES = ['f32', 'f16'] as const;

/** Require an n-D row-major tensor whose dtype is one of `allow`; returns its
 *  dims and that dtype.
 *
 *  The dtype comes back because the caller must carry it into the result meta
 *  — an op that hardcoded 'f32' would mislabel an f16 buffer and the next
 *  kernel would read it as f32. `allow` is generic so the returned dtype is
 *  narrowed to what the op actually admits, not widened to every float. */
export function wantND<D extends EagerDtype>(
  name: string,
  x: Value,
  n: number,
  allow: readonly D[],
): [number[], D] {
  rejectHwc4(name, x);
  const { dims, dtype } = x.shape;
  if (dims?.length !== n || !allow.includes(dtype as D)) {
    throw new Error(
      `${name}: expected a ${n}D ${allow.join(' or ')} tensor, got dims=${JSON.stringify(dims)} dtype=${dtype}`,
    );
  }
  return [[...dims], dtype as D];
}

/** wantND flattened to [rows, cols, dtype]. */
export function want2d<D extends EagerDtype>(
  name: string,
  x: Value,
  allow: readonly D[],
): [number, number, D] {
  const [dims, dtype] = wantND(name, x, 2, allow);
  return [dims[0]!, dims[1]!, dtype];
}

/** wantND flattened to [c, h, w, dtype]. */
export function want3d<D extends EagerDtype>(
  name: string,
  x: Value,
  allow: readonly D[],
): [number, number, number, D] {
  const [dims, dtype] = wantND(name, x, 3, allow);
  return [dims[0]!, dims[1]!, dims[2]!, dtype];
}

/** Resolves a negative dim against rank, torch-style. */
export function normDim(op: string, dim: number, rank: number): number {
  if (!Number.isInteger(dim)) throw new Error(`${op}: dim must be an integer, got ${dim}`);
  const d = dim < 0 ? dim + rank : dim;
  if (d < 0 || d >= rank) {
    throw new Error(`${op}: dim ${dim} out of range for a rank-${rank} operand`);
  }
  return d;
}
