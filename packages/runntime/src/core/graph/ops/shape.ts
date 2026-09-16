/** Layout ops: cat, slice, chunk and split with their pairwise kernel nodes,
 *  plus reshape, transpose, row writes and row gathers. */

import { hwc4Meta, pending, Value } from '../value.ts';
import { FLOAT_DTYPES, isHwc4, normDim, want2d, want3d, wantHwc4 } from './shared.ts';

export function reshape(x: Value, newDims: number[]): Value {
  if (x.shape.layout !== undefined) {
    throw new Error(`reshape: input is stored '${x.shape.layout}' — convert with toChw() first`);
  }
  const newElems = newDims.reduce((a, b) => a * b, 1);
  if (newElems !== x.shape.elems) {
    throw new Error(
      `reshape: number of elements mismatch inputs has - ${x.shape.elems}, newDims imply: ${newElems}`,
    );
  }
  return pending({ elems: newElems, dtype: x.shape.dtype, dims: newDims }, 'reshape', [x]);
}

/** Storage-width conversion, like torch's x.to(dtype). A no-op when the dtype
 *  already matches. */
export function astype(x: Value, dtype: 'f32' | 'f16'): Value {
  const dims = x.shape.dims;
  const from = x.shape.dtype;
  if (from !== 'f32' && from !== 'f16') {
    throw new Error(`astype: needs a float input, got ${from}`);
  }
  if (x.shape.layout !== undefined) {
    throw new Error(
      `astype: input is stored '${x.shape.layout}' — toChw() converts dtype and layout in one dispatch`,
    );
  }
  if (from === dtype) return x;
  return pending({ elems: x.shape.elems, dtype, dims: dims ? [...dims] : undefined }, 'astype', [
    x,
  ]);
}

/** 2D transpose, [M,N] to [N,M]. torch's x.T. */
export function transpose(x: Value): Value {
  const [m, n, dtype] = want2d('transpose', x, FLOAT_DTYPES);
  return pending({ elems: m * n, dtype, dims: [n, m] }, 'transpose', [x]);
}

/** One pairwise concat step along a resolved dim; cat() folds over this. */
function catPair(a: Value, b: Value, d: number, rank: number): Value {
  if (rank === 3) return concatChannels(a, b);
  return d === 0 ? concatRows(a, b) : concatCols(a, b);
}

/** Concatenation along dim, like torch.cat. Folds left to right into pairwise
 *  kernel nodes, which is what lets the evaluator elide the channel case. */
export function cat(tensors: readonly Value[], dim: number): Value {
  if (tensors.length === 0) throw new Error('cat: empty tensor list');
  const first = tensors[0]!;
  const rank = first.shape.dims?.length ?? 0;
  if (rank !== 2 && rank !== 3) {
    throw new Error(`cat: needs 2D or 3D operands, got dims ${first.shape.dims}`);
  }
  const d = normDim('cat', dim, rank);
  if (rank === 3 && d !== 0) {
    throw new Error(`cat: only the channel dim (0) is supported on 3D, got ${dim}`);
  }
  let acc = first;
  for (let i = 1; i < tensors.length; i++) acc = catPair(acc, tensors[i]!, d, rank);
  return acc;
}

/** Narrows x to [start, end) along dim, like torch's narrow. */
export function slice(x: Value, dim: number, start: number, end: number): Value {
  const rank = x.shape.dims?.length ?? 0;
  if (rank !== 2 && rank !== 3) {
    throw new Error(`slice: needs a 2D or 3D operand, got dims ${x.shape.dims}`);
  }
  const d = normDim('slice', dim, rank);
  if (rank === 3) {
    if (d !== 0) throw new Error(`slice: only the channel dim (0) is supported on 3D, got ${dim}`);
    return sliceChannels(x, start, end);
  }
  return d === 0 ? sliceRows(x, start, end) : sliceCols(x, start, end);
}

/** Splits x into n equal parts along dim, like torch.chunk. Exact division
 *  only. */
export function chunk(x: Value, n: number, dim: number): Value[] {
  const rank = x.shape.dims?.length ?? 0;
  const d = normDim('chunk', dim, rank);
  const size = x.shape.dims![d]!;
  if (!Number.isInteger(n) || n < 1)
    throw new Error(`chunk: n must be a positive integer, got ${n}`);
  if (size % n !== 0) {
    throw new Error(`chunk: dim ${dim} size ${size} does not divide into ${n} equal parts`);
  }
  const step = size / n;
  return Array.from({ length: n }, (_, i) => slice(x, d, i * step, (i + 1) * step));
}

/** Splits x along dim into parts of the given sizes, which must cover the dim
 *  exactly. */
export function split(x: Value, sizes: readonly number[], dim: number): Value[] {
  const rank = x.shape.dims?.length ?? 0;
  const d = normDim('split', dim, rank);
  const size = x.shape.dims![d]!;
  const total = sizes.reduce((s, v) => s + v, 0);
  if (total !== size) {
    throw new Error(`split: sizes [${sizes}] sum to ${total}, but dim ${dim} has ${size}`);
  }
  const out: Value[] = [];
  let off = 0;
  for (const s of sizes) {
    out.push(slice(x, d, off, off + s));
    off += s;
  }
  return out;
}

/** Column slice x[:, start:end), giving [M, end−start]. */
function sliceCols(x: Value, start: number, end: number): Value {
  const [m, n, dtype] = want2d('slice', x, FLOAT_DTYPES);
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end > n || start >= end) {
    throw new Error(`slice: bad range [${start}, ${end}) for ${n} cols`);
  }
  const w = end - start;
  return pending({ elems: m * w, dtype, dims: [m, w] }, 'sliceCols', [x], undefined, [start, end]);
}

/** Row slice x[start:end, :], giving [end−start, N]. */
function sliceRows(x: Value, start: number, end: number): Value {
  const [m, n, dtype] = want2d('slice', x, FLOAT_DTYPES);
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end > m || start >= end) {
    throw new Error(`slice: bad range [${start}, ${end}) for ${m} rows`);
  }
  const h = end - start;
  return pending({ elems: h * n, dtype, dims: [h, n] }, 'sliceRows', [x], undefined, [start, end]);
}

/** Channel slice, [C,H,W] to [start:end, H, W]. Reuses the sliceRows kernel,
 *  since channels are outermost. */
function sliceChannels(x: Value, start: number, end: number): Value {
  if (isHwc4(x)) return sliceChannelsHwc4(x, start, end);
  const [c, h, w] = want3d('slice', x, FLOAT_DTYPES);
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end > c || start >= end) {
    throw new Error(`slice: bad range [${start}, ${end}) for ${c} channels`);
  }
  const cOut = end - start;
  return pending(
    { elems: cOut * h * w, dtype: x.shape.dtype, dims: [cOut, h, w] },
    'sliceChannels',
    [x],
    undefined,
    [start, end],
  );
}

/** Row concat: [A,N] and [B,N] give [A+B,N]. */
function concatRows(a: Value, b: Value): Value {
  const [ma, ca, dtype] = want2d('cat', a, FLOAT_DTYPES);
  const [mb, cb, bDtype] = want2d('cat', b, FLOAT_DTYPES);
  if (ca !== cb) {
    throw new Error(`cat: col mismatch — a has ${ca}, b has ${cb}`);
  }
  if (bDtype !== dtype) {
    throw new Error(`cat: dtype mismatch — a is ${dtype}, b is ${bDtype}`);
  }
  return pending({ elems: (ma + mb) * ca, dtype, dims: [ma + mb, ca] }, 'concatRows', [a, b]);
}

/** Column concat: [M,A] and [M,B] give [M,A+B]. */
function concatCols(a: Value, b: Value): Value {
  const [ma, ca, dtype] = want2d('cat', a, FLOAT_DTYPES);
  const [mb, cb, bDtype] = want2d('cat', b, FLOAT_DTYPES);
  if (ma !== mb) {
    throw new Error(`cat: row mismatch — a has ${ma}, b has ${mb}`);
  }
  if (bDtype !== dtype) {
    throw new Error(`cat: dtype mismatch — a is ${dtype}, b is ${bDtype}`);
  }
  return pending({ elems: ma * (ca + cb), dtype, dims: [ma, ca + cb] }, 'concatCols', [a, b]);
}

/** Channel concat: [A,H,W] and [B,H,W] give [A+B,H,W]. */
function concatChannels(x: Value, y: Value): Value {
  if (isHwc4(x) || isHwc4(y)) return concatChannelsHwc4(x, y);
  const [c1, h1, w1] = want3d('cat', x, FLOAT_DTYPES);
  const [c2, h2, w2] = want3d('cat', y, FLOAT_DTYPES);
  if (h1 !== h2 || w1 !== w2) {
    throw new Error(`cat: spatial mismatch - a=${x.shape.dims}, b=${y.shape.dims}`);
  }
  if (x.shape.dtype !== y.shape.dtype) {
    throw new Error(`cat: dtype mismatch - a=${x.shape.dtype}, b=${y.shape.dtype}`);
  }
  return pending(
    {
      elems: (c1 + c2) * h1 * w1,
      dtype: x.shape.dtype,
      dims: [c1 + c2, h1, w1],
    },
    'concatChannels',
    [x, y],
  );
}

/** HWC4 channel slice; start and width must both be multiples of 4. */
function sliceChannelsHwc4(x: Value, start: number, end: number): Value {
  const [c, h, w] = x.shape.dims! as [number, number, number];
  if (!Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end > c || start >= end) {
    throw new Error(`slice: bad range [${start}, ${end}) for ${c} channels`);
  }
  if (start % 4 !== 0 || (end - start) % 4 !== 0) {
    throw new Error(
      `slice: hwc4 channel ranges must align to 4 — got [${start}, ${end}); convert with toChw() for odd splits`,
    );
  }
  return pending(hwc4Meta(end - start, h, w), 'sliceChannelsHwc4', [x], undefined, [start, end]);
}

/** HWC4 channel concat; each side's channel count must be a multiple of 4. */
function concatChannelsHwc4(x: Value, y: Value): Value {
  const [c1, h1, w1] = wantHwc4('cat.a', x);
  const [c2, h2, w2] = wantHwc4('cat.b', y);
  if (h1 !== h2 || w1 !== w2) {
    throw new Error(`cat: spatial mismatch - a=${x.shape.dims}, b=${y.shape.dims}`);
  }
  if (c1 % 4 !== 0 || c2 % 4 !== 0) {
    throw new Error(`cat: hwc4 channel counts must be multiples of 4, got ${c1} ++ ${c2}`);
  }
  return pending(hwc4Meta(c1 + c2, h1, w1), 'concatChannelsHwc4', [x, y]);
}

/** In-place row write: copies src into rows [startRow, startRow + r) of the
 *  caller-owned buffer behind dst, returning a view of that same buffer.
 *
 *  The result aliases dst, so the driver never allocates or pool-releases for
 *  it. Built for KV-cache appends: existing rows are never copied, and rows
 *  past the view keep whatever they held. */
export function writeRows(dst: Value, src: Value, startRow: number): Value {
  const [dstRows, dstCols, dtype] = want2d('writeRows.dst', dst, FLOAT_DTYPES);
  const [srcRows, srcCols, srcDtype] = want2d('writeRows.src', src, FLOAT_DTYPES);
  if (srcDtype !== dtype) {
    throw new Error(`writeRows: dtype mismatch — dst is ${dtype}, src is ${srcDtype}`);
  }
  if (dst.state !== 'materialized') {
    throw new Error('writeRows: dst must be a materialized (caller-owned) buffer');
  }
  if (srcCols !== dstCols) {
    throw new Error(`writeRows: col mismatch — dst has ${dstCols}, src has ${srcCols}`);
  }
  if (!Number.isInteger(startRow) || startRow < 0 || startRow + srcRows > dstRows) {
    throw new Error(
      `writeRows: rows [${startRow}, ${startRow + srcRows}) exceed dst capacity ${dstRows}`,
    );
  }
  const viewRows = startRow + srcRows;
  return pending(
    { elems: viewRows * dstCols, dtype, dims: [viewRows, dstCols] },
    'writeRows',
    [dst, src],
    undefined,
    [startRow],
  );
}

/** Row gather, like torch's index_select(0, idx). Repeats allowed; indices are
 *  validated here and travel as attrs. */
export function gatherRows(x: Value, idx: readonly number[]): Value {
  const [m, n, dtype] = want2d('gatherRows', x, FLOAT_DTYPES);
  if (idx.length === 0) throw new Error('gatherRows: empty index list');
  for (const i of idx) {
    if (!Number.isInteger(i) || i < 0 || i >= m) {
      throw new Error(`gatherRows: index ${i} out of range for ${m} rows`);
    }
  }
  // Copy: mutation after validation could inject out-of-range indices.
  return pending(
    { elems: idx.length * n, dtype, dims: [idx.length, n] },
    'gatherRows',
    [x],
    undefined,
    [...idx],
  );
}

/** Row gather with a GPU-resident index vector, so token selection can feed
 *  the next decode step without a CPU readback.
 *
 *  The caller guarantees idx lands in [0, x rows); it is data, unverifiable at
 *  graph-build time. */
export function gatherRowsFrom(x: Value, idx: Value): Value {
  const [, n, dtype] = want2d('gatherRowsFrom', x, FLOAT_DTYPES);
  if (idx.shape.dtype !== 'f32') {
    throw new Error(`gatherRowsFrom: idx must be f32, got ${idx.shape.dtype}`);
  }
  const rows = idx.shape.elems;
  return pending({ elems: rows * n, dtype, dims: [rows, n] }, 'gatherRowsFrom', [x, idx]);
}
