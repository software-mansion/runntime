/** Synthetic checkpoints for the weight helper and state dict hook tests. */
import { f16ToF32, f32ToF16Bits } from '../../../src/core/weights/convert.ts';
import {
  bufferSource,
  fromSafetensors,
  memoryStateDict,
  type LazyStateDict,
  type LazyTensor,
  type QuantTensor,
} from '../../../src/core/weights/safetensors.ts';

export type Entry = { values: number[]; shape: number[] };

/** The array types an entry's data can come in. */
type EntryData = Float32Array | Uint16Array | Uint32Array | Int8Array | Uint8Array;

/** Safetensors dtype by the array type that holds an entry's data. */
const dtypeOf = (data: EntryData) => {
  if (data instanceof Float32Array) return 'F32';
  if (data instanceof Uint16Array) return 'F16';
  if (data instanceof Uint32Array) return 'U32';
  return data instanceof Int8Array ? 'I8' : 'U8';
};

/** A safetensors file built from the format spec: an 8-byte little-endian
 *  header length, the JSON header, then the payload. An entry's dtype follows
 *  its data: F32 for a Float32Array, F16 for a Uint16Array of halfwords, U32
 *  for a Uint32Array, I8 for an Int8Array, U8 for a Uint8Array. `metadata` goes in `__metadata__`. */
export function safetensorsFile(
  entries: Record<string, { data: EntryData; shape: number[] }>,
  metadata?: Record<string, string>,
): Uint8Array {
  const header: Record<string, unknown> = metadata ? { __metadata__: metadata } : {};
  const chunks: Uint8Array[] = [];
  let offset = 0;
  for (const [name, { data, shape }] of Object.entries(entries)) {
    const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
    const dtype = dtypeOf(data);
    header[name] = { dtype, shape, data_offsets: [offset, offset + bytes.length] };
    chunks.push(bytes);
    offset += bytes.length;
  }
  const json = new TextEncoder().encode(JSON.stringify(header));
  const out = new Uint8Array(8 + json.length + offset);
  new DataView(out.buffer).setBigUint64(0, BigInt(json.length), true);
  out.set(json, 8);
  let at = 8 + json.length;
  for (const bytes of chunks) {
    out.set(bytes, at);
    at += bytes.length;
  }
  return out;
}

/** An F16 safetensors file of `entries`, each value rounded to f16. */
export const f16File = (entries: Record<string, Entry>): Uint8Array =>
  safetensorsFile(
    Object.fromEntries(
      Object.entries(entries).map(([name, { values, shape }]) => [
        name,
        { data: Uint16Array.from(values, f32ToF16Bits), shape },
      ]),
    ),
  );

/** `entries` as an F16 safetensors file, or as f32 in memory. memoryStateDict
 *  reads a Uint16Array as BF16, so F16 has to go through a real file. */
export async function checkpoint(
  dtype: 'F32' | 'F16',
  entries: Record<string, Entry>,
): Promise<LazyStateDict> {
  if (dtype === 'F16') return fromSafetensors(bufferSource(f16File(entries)), { chunkBytes: 0 });
  return memoryStateDict(
    Object.fromEntries(
      Object.entries(entries).map(([name, { values, shape }]) => [
        name,
        { data: new Float32Array(values), shape },
      ]),
    ),
  );
}

/** What a derived tensor holds: f16 halfwords or f32, decoded to numbers. */
export async function contents(t: LazyTensor): Promise<{ type: 'f16' | 'f32'; values: number[] }> {
  if (t.kind !== 'derived') throw new Error('expected a derived tensor');
  const data = await t.data();
  if (data instanceof Uint16Array) return { type: 'f16', values: [...f16ToF32(data)] };
  if (data instanceof Float32Array) return { type: 'f32', values: [...data] };
  throw new Error('expected floats');
}

/** A quantized linear weight: `ints` row-major `[out, in]`, `scales` row-major
 *  `[out, in / groupSize]`. */
export type QuantEntry = { ints: number[]; scales: number[]; shape: [number, number] };

/** `values`, a torch-layout `[out, in]` weight, quantized the way the format
 *  spec's symmetric scheme reads it: per group of `groupSize` along `in`, the
 *  scale is the group's largest |value| over the top int (127 for int8, 7 for
 *  int4) rounded to f16, and each int is value / scale rounded to nearest. */
export function quantizeWeight(
  values: Float32Array,
  shape: [number, number],
  bits: 4 | 8,
  groupSize: number,
): QuantEntry {
  const top = 2 ** (bits - 1) - 1;
  const ints = new Array<number>(values.length);
  const scales = new Array<number>(values.length / groupSize);
  for (let g = 0; g < scales.length; g++) {
    const group = values.subarray(g * groupSize, (g + 1) * groupSize);
    const largest = group.reduce((most, v) => Math.max(most, Math.abs(v)), 0);
    const scale = f16ToF32(Uint16Array.of(f32ToF16Bits(largest / top)))[0]!;
    scales[g] = scale;
    // An all-zero group has scale 0 and ints 0.
    group.forEach((v, i) => (ints[g * groupSize + i] = scale === 0 ? 0 : Math.round(v / scale)));
  }
  return { ints, scales, shape };
}

/** A quantized safetensors file in the format spec: each entry's ints as
 *  `name` (I8, or U8 with two int4 per byte, low nibble first), its scales
 *  rounded to f16 as `name_scale`, and the setting in `runntime.quant`.
 *  `floats` go in as F32 tensors next to them. */
export function quantFile(
  bits: 4 | 8,
  groupSize: number,
  entries: Record<string, QuantEntry>,
  floats: Record<string, { data: Float32Array; shape: number[] }> = {},
): Uint8Array {
  const tensors: Parameters<typeof safetensorsFile>[0] = { ...floats };
  for (const [name, { ints, scales, shape }] of Object.entries(entries)) {
    const [out, cols] = shape;
    tensors[name] =
      bits === 8
        ? { data: Int8Array.from(ints), shape }
        : {
            data: Uint8Array.from({ length: ints.length / 2 }, (_, i) => {
              return (ints[2 * i]! & 0x0f) | ((ints[2 * i + 1]! & 0x0f) << 4);
            }),
            shape: [out, cols / 2],
          };
    tensors[`${name}_scale`] = {
      data: Uint16Array.from(scales, f32ToF16Bits),
      shape: [out, cols / groupSize],
    };
  }
  const setting = { version: 1, bits, groupSize, scheme: 'symmetric' };
  return safetensorsFile(tensors, { 'runntime.quant': JSON.stringify(setting) });
}

/** The quantized tensor `name` of `entries`, read back from a quantFile. */
export async function quantSource(
  bits: 4 | 8,
  groupSize: number,
  name: string,
  entry: QuantEntry,
): Promise<QuantTensor> {
  const file = quantFile(bits, groupSize, { [name]: entry });
  const t = (await fromSafetensors(bufferSource(file), { chunkBytes: 0 })).tensors.get(name)!;
  if (t.kind !== 'quant') throw new Error('expected a quantized tensor');
  return t;
}

/** What a QuantTensor holds: its ints, and its scales decoded to numbers with
 *  whether they are f16 halfwords or f32. */
export async function quantContents(
  t: LazyTensor,
): Promise<{ ints: number[]; scaleType: 'f16' | 'f32'; scales: number[] }> {
  if (t.kind !== 'quant') throw new Error('expected a quantized tensor');
  const [ints, scales] = await Promise.all([t.ints(), t.scales()]);
  return {
    ints: [...ints],
    scaleType: scales instanceof Uint16Array ? 'f16' : 'f32',
    scales: [...(scales instanceof Uint16Array ? f16ToF32(scales) : scales)],
  };
}
