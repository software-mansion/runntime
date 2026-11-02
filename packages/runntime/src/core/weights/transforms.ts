/** Lazy reshaping helpers for `transformStateDict` hooks: concatenate, split,
 *  permute, scale, transpose and reshape checkpoint tensors without fetching
 *  them.
 *
 *  F16 sources give f16 halfwords and F32 sources give f32, except
 *  `scaleWeight`, which always gives f32. BF16 decodes to
 *  f32, which is exact; a derived tensor's Uint16Array means f16, so bf16 bits
 *  have no derived form. Each output's byteLength is its share of the source
 *  bytes, so load progress still sums to the checkpoint size. */
import { RunntimeError } from '../error.ts';
import { f16ToF32 } from './convert.ts';
import { derivedTensor, type LazyTensor } from './safetensors.ts';

/** f32 values, or f16 halfwords. */
type Floats = Float32Array | Uint16Array;

const numel = (shape: readonly number[]): number => shape.reduce((a, b) => a * b, 1);

const mismatch = (helper: string, why: string) =>
  new RunntimeError('CHECKPOINT_MISMATCH', `${helper}: ${why}`);

/** Reads `t` as f16 halfwords when it holds F16 and as f32 otherwise. Throws
 *  CHECKPOINT_MISMATCH, naming `helper`, on packed u32 words or on a derived
 *  tensor whose data does not fill its shape. A file tensor's length is
 *  checked by the header parser. */
export async function readFloats(helper: string, t: LazyTensor): Promise<Floats> {
  if (t.kind === 'file') return t.dtype === 'F16' ? t.u16() : t.f32();
  const data = await t.data();
  if (data instanceof Uint32Array) throw mismatch(helper, 'source holds packed u32 words');
  if (data.length !== numel(t.shape)) {
    throw mismatch(helper, `source holds ${data.length} values for shape [${t.shape.join(', ')}]`);
  }
  return data;
}

const allocLike = (like: Floats, n: number): Floats =>
  like instanceof Uint16Array ? new Uint16Array(n) : new Float32Array(n);

/** One read of `t` for `consumers` outputs, dropped once the last has it. */
function sharedRead(helper: string, t: LazyTensor, consumers: number): () => Promise<Floats> {
  let memo: Promise<Floats> | undefined;
  let reads = 0;
  return () => {
    const data = (memo ??= readFloats(helper, t));
    if (++reads === consumers) memo = undefined;
    return data;
  };
}

function checkSizes(helper: string, t: LazyTensor, sizes: readonly number[], axis: number): void {
  const total = sizes.reduce((a, b) => a + b, 0);
  if (total !== t.shape[axis]) {
    throw mismatch(
      helper,
      `sizes [${sizes.join(', ')}] sum to ${total}, not dim ${axis} of shape [${t.shape.join(', ')}]`,
    );
  }
}

/** Stacks `parts` along dim 0. Every part needs the same trailing dims. The
 *  result is f16 when every part is, f32 otherwise. A part listed more than
 *  once (a norm tiled over every head) is read once. */
export function concatRows(parts: readonly LazyTensor[]): LazyTensor {
  const shapes = parts.map((p) => `[${p.shape.join(', ')}]`).join(', ');
  const first = parts[0];
  if (!first || first.shape.length === 0) throw mismatch('concatRows', `cannot stack ${shapes}`);
  const rest = first.shape.slice(1);
  for (const p of parts) {
    if (p.shape.length !== first.shape.length || p.shape.slice(1).join() !== rest.join()) {
      throw mismatch('concatRows', `cannot stack ${shapes}`);
    }
  }
  const rows = parts.reduce((sum, p) => sum + p.shape[0]!, 0);
  const shape = [rows, ...rest];
  return derivedTensor(
    shape,
    async () => {
      const read = new Map<LazyTensor, Floats>();
      for (const p of parts) {
        if (!read.has(p)) read.set(p, await readFloats('concatRows', p));
      }
      const datas = parts.map((p) => read.get(p)!);
      const half = datas.every((d) => d instanceof Uint16Array);
      const out = half ? new Uint16Array(numel(shape)) : new Float32Array(numel(shape));
      let base = 0;
      for (const d of datas) {
        out.set(half || d instanceof Float32Array ? d : f16ToF32(d), base);
        base += d.length;
      }
      return out;
    },
    parts.reduce((sum, p) => sum + p.byteLength, 0),
  );
}

/** Splits `t` along dim 0 into parts of `sizes` rows. The source is read once
 *  for all parts. */
export function splitRows(t: LazyTensor, sizes: readonly number[]): LazyTensor[] {
  if (t.shape.length === 0) throw mismatch('splitRows', 'cannot split a scalar');
  checkSizes('splitRows', t, sizes, 0);
  const rest = t.shape.slice(1);
  const rowElems = numel(rest);
  const source = sharedRead('splitRows', t, sizes.length);
  let start = 0;
  return sizes.map((rows) => {
    const from = start * rowElems;
    start += rows;
    return derivedTensor(
      [rows, ...rest],
      async () => (await source()).slice(from, from + rows * rowElems),
      (t.byteLength * rows) / t.shape[0]!,
    );
  });
}

/** Splits `t` along its last dim into parts of `sizes` columns, keeping the
 *  leading dims. The source is read once for all parts. */
export function splitCols(t: LazyTensor, sizes: readonly number[]): LazyTensor[] {
  const axis = t.shape.length - 1;
  if (axis < 0) throw mismatch('splitCols', 'cannot split a scalar');
  checkSizes('splitCols', t, sizes, axis);
  const lead = t.shape.slice(0, axis);
  const rows = numel(lead);
  const width = t.shape[axis]!;
  const source = sharedRead('splitCols', t, sizes.length);
  let start = 0;
  return sizes.map((cols) => {
    const from = start;
    start += cols;
    return derivedTensor(
      [...lead, cols],
      async () => {
        const data = await source();
        const out = allocLike(data, rows * cols);
        for (let r = 0; r < rows; r++) {
          out.set(data.subarray(r * width + from, r * width + from + cols), r * cols);
        }
        return out;
      },
      (t.byteLength * cols) / width,
    );
  });
}

/** Reorders the rows of dim 0: row i of the result is row `order[i]` of `t`,
 *  like torch `t[order]`. Throws CHECKPOINT_MISMATCH when `order` is not a
 *  permutation of 0 … rows − 1. */
export function permuteRows(t: LazyTensor, order: readonly number[]): LazyTensor {
  if (t.shape.length === 0) throw mismatch('permuteRows', 'cannot permute a scalar');
  const rows = t.shape[0]!;
  const inRange = order.every((r) => Number.isInteger(r) && r >= 0 && r < rows);
  if (order.length !== rows || new Set(order).size !== rows || !inRange) {
    throw mismatch(
      'permuteRows',
      `order is not a permutation of the ${rows} rows of shape [${t.shape.join(', ')}]`,
    );
  }
  const rowElems = numel(t.shape) / rows;
  return derivedTensor(
    t.shape,
    async () => {
      const data = await readFloats('permuteRows', t);
      const out = allocLike(data, data.length);
      for (const [i, r] of order.entries()) {
        out.set(data.subarray(r * rowElems, (r + 1) * rowElems), i * rowElems);
      }
      return out;
    },
    t.byteLength,
  );
}

/** Multiplies `t` by `scale`: one number for every element, or one per row
 *  of dim 0. The result is f32 even for an F16 source: rounding the product
 *  back to f16 would change what an f32 model loads. */
export function scaleWeight(t: LazyTensor, scale: number | readonly number[]): LazyTensor {
  const rows = t.shape[0] ?? 1;
  if (typeof scale !== 'number' && scale.length !== rows) {
    throw mismatch(
      'scaleWeight',
      `${scale.length} row scales for shape [${t.shape.join(', ')}], need ${rows}`,
    );
  }
  const rowElems = numel(t.shape) / rows;
  const scaleAt = (i: number) =>
    typeof scale === 'number' ? scale : scale[Math.floor(i / rowElems)]!;
  return derivedTensor(
    t.shape,
    async () => {
      const data = await readFloats('scaleWeight', t);
      const values = data instanceof Float32Array ? data : f16ToF32(data);
      return values.map((v, i) => v * scaleAt(i));
    },
    t.byteLength,
  );
}

/** Swaps the two dims of a 2D tensor: [rows, cols] becomes [cols, rows].
 *  Named apart from the `transpose` graph op. */
export function transposeWeight(t: LazyTensor): LazyTensor {
  if (t.shape.length !== 2) {
    throw mismatch('transposeWeight', `needs a 2D tensor, got shape [${t.shape.join(', ')}]`);
  }
  const rows = t.shape[0]!;
  const cols = t.shape[1]!;
  return derivedTensor(
    [cols, rows],
    async () => {
      const data = await readFloats('transposeWeight', t);
      const out = allocLike(data, data.length);
      for (let r = 0; r < rows; r++) {
        for (let c = 0; c < cols; c++) out[c * rows + r] = data[r * cols + c]!;
      }
      return out;
    },
    t.byteLength,
  );
}

/** Gives `t` a new shape with the same element count, like torch `reshape`.
 *  The data is untouched: a file tensor stays a file tensor of its dtype, and
 *  byteLength stays the same. Throws CHECKPOINT_MISMATCH when the element
 *  counts differ. */
export function reshapeWeight(t: LazyTensor, shape: readonly number[]): LazyTensor {
  if (numel(shape) !== numel(t.shape)) {
    throw mismatch(
      'reshapeWeight',
      `cannot reshape [${t.shape.join(', ')}] to [${shape.join(', ')}]`,
    );
  }
  // Every LazyTensor is a plain object whose readers close over their own
  // source, so a shallow copy shares the data and reads it the same way.
  return { ...t, shape: [...shape] };
}
