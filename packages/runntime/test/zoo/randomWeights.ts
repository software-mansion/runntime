/** Random weights for a zoo model as a safetensors file in memory, so a test
 *  can load a real create<Task>() runner without a download. */
import { bufferSource, type RangeSource } from '../../src/core/weights/safetensors.ts';
import type { AnyModule } from '../../src/core/nn/module.ts';
import { randomF32 } from '../gpu.ts';

/** Every parameter and buffer of `model` under its own name and shape,
 *  random in [-scale, scale), with `shapes` overriding the shape a
 *  checkpoint stores differently. */
export function randomSafetensors(
  model: AnyModule,
  { scale = 1, shapes = {} }: { scale?: number; shapes?: Record<string, number[]> } = {},
): RangeSource {
  const tensors: { name: string; shape: number[]; data: Float32Array }[] = [];
  let seed = 1;
  for (const [name, p] of [...model.namedParameters(), ...model.namedBuffers()]) {
    const shape = shapes[name] ?? [...(p.shape.dims ?? [p.shape.elems])];
    const elems = shape.reduce((a, b) => a * b, 1);
    tensors.push({ name, shape, data: randomF32(elems, seed++).map((v) => v * scale) });
  }
  // The format: a u64 header length, the JSON header, then the raw bytes.
  const header: Record<string, unknown> = {};
  let offset = 0;
  for (const { name, shape, data } of tensors) {
    header[name] = { dtype: 'F32', shape, data_offsets: [offset, offset + data.byteLength] };
    offset += data.byteLength;
  }
  const json = new TextEncoder().encode(JSON.stringify(header));
  const bytes = new Uint8Array(8 + json.length + offset);
  new DataView(bytes.buffer).setBigUint64(0, BigInt(json.length), true);
  bytes.set(json, 8);
  let at = 8 + json.length;
  for (const { data } of tensors) {
    bytes.set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength), at);
    at += data.byteLength;
  }
  return bufferSource(bytes);
}
