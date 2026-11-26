/** Random weights for a zoo model: as a safetensors file in memory, so a test
 *  can load a real create<Task>() runner without a download, or bound
 *  straight to the model's slots. */
import type { TgpuRoot } from 'typegpu';
import { onTestFinished } from 'vitest';
import { gpuUpload } from '../../src/core/nn/loadStateDict.ts';
import { CpuParameter, type AnyModule } from '../../src/core/nn/module.ts';
import { f32ArrayToF16Bits } from '../../src/core/weights/convert.ts';
import { bufferSource, type RangeSource } from '../../src/core/weights/safetensors.ts';
import { randomF32 } from '../gpu.ts';

/** One tensor of a random file. `offset` is added to every value: 1 keeps a
 *  BatchNorm variance positive. */
export interface RandomTensor {
  name: string;
  shape: number[];
  offset?: number;
}

/** Every parameter and buffer of `model` under its own name and shape,
 *  random in [-scale, scale), with `shapes` overriding the shape a
 *  checkpoint stores differently. */
export function randomSafetensors(
  model: AnyModule,
  { scale = 1, shapes = {} }: { scale?: number; shapes?: Record<string, number[]> } = {},
): RangeSource {
  const slots = [...model.namedParameters(), ...model.namedBuffers()];
  return randomTensorsFile(
    slots.map(([name, p]) => ({
      name,
      shape: shapes[name] ?? [...(p.shape.dims ?? [p.shape.elems])],
    })),
    scale,
  );
}

/** `list` as a safetensors file, each tensor random in [-scale, scale) plus
 *  its offset. */
export function randomTensorsFile(list: readonly RandomTensor[], scale = 1): RangeSource {
  const tensors = list.map(({ name, shape, offset = 0 }, i) => {
    const elems = shape.reduce((a, b) => a * b, 1);
    return { name, shape, data: randomF32(elems, i + 1).map((v) => v * scale + offset) };
  });
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

/** Binds every parameter and buffer the way loadStateDict does, with random
 *  values. Skips the checkpoint transforms: the names and layouts they adapt
 *  are the checkpoint's, and the model's own shapes are what runs. */
export function loadRandomWeights(model: AnyModule, root: TgpuRoot): void {
  const upload = gpuUpload(root);
  let seed = 1;
  // The stored parameters: a FusedLinear's stacked weight, not its parts.
  for (const [, p] of model.storageParameters()) {
    const f32 = randomF32(p.shape.elems, seed++);
    if (p instanceof CpuParameter) {
      p.bindData(f32);
      continue;
    }
    // f16 slots get their bits before preprocess, as from a checkpoint.
    const data = p.shape.dtype === 'f16' ? f32ArrayToF16Bits(f32).data : f32;
    p.bind(upload(p.preprocess ? p.preprocess(data) : data, p.shape));
  }
  for (const [, b] of model.namedBuffers()) b.bind(upload(b.data, b.shape));
}

/** The URL of a safetensors file with no tensors, for a loader that fetches
 *  its weights and a test that binds them with loadRandomWeights(). */
export function emptyWeightsUrl(): string {
  const header = new TextEncoder().encode('{}');
  const bytes = new Uint8Array(8 + header.length);
  new DataView(bytes.buffer).setBigUint64(0, BigInt(header.length), true);
  bytes.set(header, 8);
  const url = URL.createObjectURL(new Blob([bytes]));
  onTestFinished(() => URL.revokeObjectURL(url));
  return url;
}
