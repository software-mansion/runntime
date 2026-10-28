/** Lazy reshaping helpers for `transformStateDict` hooks: concatenate, split,
 *  permute, scale, transpose and reshape checkpoint tensors without fetching
 *  them.
 *
 *  F16 sources give f16 halfwords and F32 sources give f32, except
 *  `scaleWeight`, which always gives f32. BF16 decodes to
 *  f32, which is exact; a derived tensor's Uint16Array means f16, so bf16 bits
 *  have no derived form. Each output's byteLength is its share of the source
 *  bytes, so load progress still sums to the checkpoint size.
 *
 *  A QuantTensor source gives a QuantTensor: the helpers move its ints and
 *  scales together and never dequantize. Whatever would split a scale group
 *  (a column split off the group grid, a reshape of the last dim, a
 *  transpose) throws CHECKPOINT_MISMATCH. */
import { RunntimeError } from '../error.ts';
import { f16ToF32 } from './convert.ts';
import { derivedTensor, quantTensor, type LazyTensor, type QuantTensor } from './safetensors.ts';

/** f32 values, or f16 halfwords. */
type Floats = Float32Array | Uint16Array;

/** Floats, or the ints of a QuantTensor. */
type Rows = Floats | Int8Array;

const numel = (shape: readonly number[]): number => shape.reduce((a, b) => a * b, 1);

const mismatch = (helper: string, why: string) =>
  new RunntimeError('CHECKPOINT_MISMATCH', `${helper}: ${why}`);

/** Reads `t` as f16 halfwords when it holds F16 and as f32 otherwise. Throws
 *  CHECKPOINT_MISMATCH, naming `helper`, on packed u32 words or on a derived
 *  tensor whose data does not fill its shape. A file tensor's length is
 *  checked by the header parser. */
export async function readFloats(helper: string, t: LazyTensor): Promise<Floats> {
  if (t.kind === 'quant') throw mismatch(helper, `${t.name} is quantized, not float`);
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

/** One `read` for `consumers` outputs, dropped once the last has it. */
function sharedRead<T>(read: () => Promise<T>, consumers: number): () => Promise<T> {
  let memo: Promise<T> | undefined;
  let reads = 0;
  return () => {
    const data = (memo ??= read());
    if (++reads === consumers) memo = undefined;
    return data;
  };
}

/** `parts` narrowed to QuantTensors, or undefined when every part is float.
 *  Throws CHECKPOINT_MISMATCH when float and quant parts mix or the quant
 *  parts differ in bits or group size. */
function quantParts(helper: string, parts: readonly LazyTensor[]): QuantTensor[] | undefined {
  const quant = parts.filter((p): p is QuantTensor => p.kind === 'quant');
  if (quant.length === 0) return undefined;
  if (quant.length !== parts.length) throw mismatch(helper, 'cannot mix quantized and float parts');
  const { bits, groupSize } = quant[0]!;
  if (quant.some((p) => p.bits !== bits || p.groupSize !== groupSize)) {
    throw mismatch(helper, 'quantized parts differ in bits or group size');
  }
  return quant;
}

/** Each part read once, even when listed several times, in `parts` order. */
async function readEachOnce<P, T>(parts: readonly P[], read: (p: P) => Promise<T>): Promise<T[]> {
  const done = new Map<P, T>();
  for (const p of parts) {
    if (!done.has(p)) done.set(p, await read(p));
  }
  return parts.map((p) => done.get(p)!);
}

/** `datas` end to end: f16 when every one is, f32 otherwise. */
function stackFloats(datas: readonly Floats[]): Floats {
  const n = datas.reduce((sum, d) => sum + d.length, 0);
  const half = datas.every((d) => d instanceof Uint16Array);
  const out = half ? new Uint16Array(n) : new Float32Array(n);
  let base = 0;
  for (const d of datas) {
    out.set(half || d instanceof Float32Array ? d : f16ToF32(d), base);
    base += d.length;
  }
  return out;
}

/** Fills `out` with row `order[i]` of `data` at row i, rows of `rowElems`. */
function reorderRows<T extends Rows>(
  data: T,
  out: T,
  order: readonly number[],
  rowElems: number,
): T {
  for (const [i, r] of order.entries()) {
    out.set(data.subarray(r * rowElems, (r + 1) * rowElems), i * rowElems);
  }
  return out;
}

/** Fills `out` with columns [from, from + cols) of every row of `data`, rows
 *  of `width`. */
function copyCols<T extends Rows>(data: T, out: T, width: number, from: number, cols: number): T {
  const rows = data.length / width;
  for (let r = 0; r < rows; r++) {
    out.set(data.subarray(r * width + from, r * width + from + cols), r * cols);
  }
  return out;
}

const toF32 = (data: Floats): Float32Array =>
  data instanceof Float32Array ? data : f16ToF32(data);

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
  const byteLength = parts.reduce((sum, p) => sum + p.byteLength, 0);
  const quant = quantParts('concatRows', parts);
  if (quant) {
    return quantTensor({
      name: `concatRows(${[...new Set(quant.map((p) => p.name))].join(', ')})`,
      shape,
      bits: quant[0]!.bits,
      groupSize: quant[0]!.groupSize,
      byteLength,
      async ints() {
        const out = new Int8Array(numel(shape));
        let base = 0;
        for (const d of await readEachOnce(quant, (p) => p.ints())) {
          out.set(d, base);
          base += d.length;
        }
        return out;
      },
      scales: async () => stackFloats(await readEachOnce(quant, (p) => p.scales())),
    });
  }
  return derivedTensor(
    shape,
    async () => stackFloats(await readEachOnce(parts, (p) => readFloats('concatRows', p))),
    byteLength,
  );
}

/** Splits `t` along dim 0 into parts of `sizes` rows. The source is read once
 *  for all parts. */
export function splitRows(t: LazyTensor, sizes: readonly number[]): LazyTensor[] {
  if (t.shape.length === 0) throw mismatch('splitRows', 'cannot split a scalar');
  checkSizes('splitRows', t, sizes, 0);
  const rest = t.shape.slice(1);
  const rowElems = numel(rest);
  const share = (rows: number) => (t.byteLength * rows) / t.shape[0]!;
  if (t.kind === 'quant') {
    const ints = sharedRead(() => t.ints(), sizes.length);
    const scales = sharedRead(() => t.scales(), sizes.length);
    const scaleRowElems = rowElems / t.groupSize;
    let start = 0;
    return sizes.map((rows) => {
      const first = start;
      start += rows;
      return quantTensor({
        name: `splitRows(${t.name})`,
        shape: [rows, ...rest],
        bits: t.bits,
        groupSize: t.groupSize,
        byteLength: share(rows),
        ints: async () => (await ints()).slice(first * rowElems, (first + rows) * rowElems),
        scales: async () =>
          (await scales()).slice(first * scaleRowElems, (first + rows) * scaleRowElems),
      });
    });
  }
  const source = sharedRead(() => readFloats('splitRows', t), sizes.length);
  let start = 0;
  return sizes.map((rows) => {
    const from = start * rowElems;
    start += rows;
    return derivedTensor(
      [rows, ...rest],
      async () => (await source()).slice(from, from + rows * rowElems),
      share(rows),
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
  const share = (cols: number) => (t.byteLength * cols) / width;
  if (t.kind === 'quant') {
    const { groupSize } = t;
    if (sizes.some((cols) => cols % groupSize !== 0)) {
      throw mismatch(
        'splitCols',
        `sizes [${sizes.join(', ')}] split the groups of ${groupSize} columns of ${t.name}`,
      );
    }
    const ints = sharedRead(() => t.ints(), sizes.length);
    const scales = sharedRead(() => t.scales(), sizes.length);
    let start = 0;
    return sizes.map((cols) => {
      const from = start;
      start += cols;
      return quantTensor({
        name: `splitCols(${t.name})`,
        shape: [...lead, cols],
        bits: t.bits,
        groupSize,
        byteLength: share(cols),
        ints: async () => copyCols(await ints(), new Int8Array(rows * cols), width, from, cols),
        async scales() {
          const data = await scales();
          const out = allocLike(data, (rows * cols) / groupSize);
          return copyCols(data, out, width / groupSize, from / groupSize, cols / groupSize);
        },
      });
    });
  }
  const source = sharedRead(() => readFloats('splitCols', t), sizes.length);
  let start = 0;
  return sizes.map((cols) => {
    const from = start;
    start += cols;
    return derivedTensor(
      [...lead, cols],
      async () => {
        const data = await source();
        return copyCols(data, allocLike(data, rows * cols), width, from, cols);
      },
      share(cols),
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
  if (t.kind === 'quant') {
    return quantTensor({
      ...t,
      name: `permuteRows(${t.name})`,
      async ints() {
        const data = await t.ints();
        return reorderRows(data, new Int8Array(data.length), order, rowElems);
      },
      async scales() {
        const data = await t.scales();
        return reorderRows(data, allocLike(data, data.length), order, rowElems / t.groupSize);
      },
    });
  }
  return derivedTensor(
    t.shape,
    async () => {
      const data = await readFloats('permuteRows', t);
      return reorderRows(data, allocLike(data, data.length), order, rowElems);
    },
    t.byteLength,
  );
}

/** Multiplies `t` by `scale`: one number for every element, or one per row
 *  of dim 0. The result is f32 even for an F16 source: rounding the product
 *  back to f16 would change what an f32 model loads. A QuantTensor keeps its
 *  ints and gets f32 scales multiplied instead. */
export function scaleWeight(t: LazyTensor, scale: number | readonly number[]): LazyTensor {
  const rows = t.shape[0] ?? 1;
  if (typeof scale !== 'number' && scale.length !== rows) {
    throw mismatch(
      'scaleWeight',
      `${scale.length} row scales for shape [${t.shape.join(', ')}], need ${rows}`,
    );
  }
  const rowElems = numel(t.shape) / rows;
  /** Multiplies element i of `values`, rows of `rowElems`. */
  const scaled = (values: Float32Array, rowElems: number) =>
    values.map(
      (v, i) => v * (typeof scale === 'number' ? scale : scale[Math.floor(i / rowElems)]!),
    );
  if (t.kind === 'quant') {
    return quantTensor({
      ...t,
      name: `scaleWeight(${t.name})`,
      scales: async () => scaled(toF32(await t.scales()), rowElems / t.groupSize),
    });
  }
  return derivedTensor(
    t.shape,
    async () => scaled(toF32(await readFloats('scaleWeight', t)), rowElems),
    t.byteLength,
  );
}

/** Swaps the two dims of a 2D tensor: [rows, cols] becomes [cols, rows].
 *  Named apart from the `transpose` graph op. */
export function transposeWeight(t: LazyTensor): LazyTensor {
  if (t.shape.length !== 2) {
    throw mismatch('transposeWeight', `needs a 2D tensor, got shape [${t.shape.join(', ')}]`);
  }
  if (t.kind === 'quant') {
    throw mismatch('transposeWeight', `${t.name} is quantized: a transpose would break its groups`);
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
 *  counts differ, or when a QuantTensor's last dim (and so its groups)
 *  would change. */
export function reshapeWeight(t: LazyTensor, shape: readonly number[]): LazyTensor {
  if (numel(shape) !== numel(t.shape)) {
    throw mismatch(
      'reshapeWeight',
      `cannot reshape [${t.shape.join(', ')}] to [${shape.join(', ')}]`,
    );
  }
  if (t.kind === 'quant' && shape.at(-1) !== t.shape.at(-1)) {
    throw mismatch(
      'reshapeWeight',
      `${t.name} is quantized: [${t.shape.join(', ')}] to [${shape.join(', ')}] changes the last dim`,
    );
  }
  // Every LazyTensor is a plain object whose readers close over their own
  // source, so a shallow copy shares the data and reads it the same way.
  return { ...t, shape: [...shape] };
}
