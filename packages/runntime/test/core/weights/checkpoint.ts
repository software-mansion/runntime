/** Synthetic checkpoints for the weight helper and state dict hook tests. */
import { f16ToF32, f32ToF16Bits } from '../../../src/core/weights/convert.ts';
import {
  bufferSource,
  fromSafetensors,
  memoryStateDict,
  type LazyStateDict,
  type LazyTensor,
} from '../../../src/core/weights/safetensors.ts';

export type Entry = { values: number[]; shape: number[] };

/** An F16 safetensors file, built from the format spec: an 8-byte
 *  little-endian header length, the JSON header, then the payload. */
export function f16File(entries: Record<string, Entry>): Uint8Array {
  const header: Record<string, unknown> = {};
  const halves: number[] = [];
  for (const [name, { values, shape }] of Object.entries(entries)) {
    const begin = halves.length * 2;
    halves.push(...values.map(f32ToF16Bits));
    header[name] = { dtype: 'F16', shape, data_offsets: [begin, halves.length * 2] };
  }
  const json = new TextEncoder().encode(JSON.stringify(header));
  const payload = new Uint8Array(new Uint16Array(halves).buffer);
  const out = new Uint8Array(8 + json.length + payload.length);
  new DataView(out.buffer).setBigUint64(0, BigInt(json.length), true);
  out.set(json, 8);
  out.set(payload, 8 + json.length);
  return out;
}

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
