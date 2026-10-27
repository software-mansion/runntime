/** Lazy reshaping helpers for `transformStateDict` hooks: concatenate, split,
 *  scale and transpose checkpoint tensors without fetching them.
 *
 *  F16 sources give f16 halfwords and F32 sources give f32. BF16 decodes to
 *  f32, which is exact; a derived tensor's Uint16Array means f16, so bf16 bits
 *  have no derived form. Each output's byteLength is its share of the source
 *  bytes, so load progress still sums to the checkpoint size. */
import { RunntimeError } from '../error.ts';
import { f16ToF32, f32ToF16Bits } from './convert.ts';
import { derivedTensor, type LazyTensor } from './safetensors.ts';

/** f32 values, or f16 halfwords. */
type Floats = Float32Array | Uint16Array;

const numel = (shape: readonly number[]): number => shape.reduce((a, b) => a * b, 1);

const mismatch = (helper: string, why: string) =>
  new RunntimeError('CHECKPOINT_MISMATCH', `${helper}: ${why}`);

async function readFloats(helper: string, t: LazyTensor): Promise<Floats> {
  if (t.kind === 'file') return t.dtype === 'F16' ? t.u16() : t.f32();
  const data = await t.data();
  if (data instanceof Uint32Array) throw mismatch(helper, 'source holds packed u32 words');
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
 *  result is f16 when every part is, f32 otherwise. */
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
      const datas: Floats[] = [];
      for (const p of parts) datas.push(await readFloats('concatRows', p));
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

/** Multiplies `t` by `scale`: one number for every element, or one per row
 *  of dim 0. F16 rounds each product back to f16. */
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
      if (data instanceof Float32Array) return data.map((v, i) => v * scaleAt(i));
      const values = f16ToF32(data);
      return Uint16Array.from(values, (v, i) => f32ToF16Bits(v * scaleAt(i)));
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
