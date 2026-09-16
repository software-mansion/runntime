import { d } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import type { F16Buffer, F32Buffer, U32Buffer } from '../kernels/common.ts';
import {
  elemBytes,
  type GpuBufferRef,
  materialized,
  matrix,
  type Value,
  type ValueMeta,
} from '../graph/value.ts';
import { f16ToF32, f32ArrayToF16Bits } from '../weights/convert.ts';
import { defaultRoot } from './context.ts';

/** Raw upload and readback utilities. Uploads go through `queue.writeBuffer`
 *  rather than `buffer.write()`, which retains a full-size host-side staging
 *  copy and would double host memory at weight sizes. */

/** Refuses a storage buffer past `maxStorageBufferBindingSize`. One is created
 *  happily and only fails inside a bind group, where nobody reads the error, so
 *  the model runs at full speed and returns garbage. */
export function checkBindable(root: TgpuRoot, bytes: number, what: string): void {
  const cap = root.device?.limits?.maxStorageBufferBindingSize;
  if (cap !== undefined && bytes > cap) {
    throw new Error(
      `${what}: ${(bytes / 1e6).toFixed(1)} MB exceeds this device's ` +
        `maxStorageBufferBindingSize of ${(cap / 1e6).toFixed(1)} MB — no shader can bind it. ` +
        'Halve the model (nn.Module.half()) or use a quantized weight format.',
    );
  }
}

/** writeBuffer via the backing buffer; TS rejects the typed array itself. */
function rawWrite(
  root: TgpuRoot,
  dst: GPUBuffer,
  data: Uint32Array | Float32Array | Uint16Array,
): void {
  root.device.queue.writeBuffer(
    dst,
    0,
    data.buffer as ArrayBuffer,
    data.byteOffset,
    data.byteLength,
  );
}

/** Overwrites an f32 storage buffer in place. */
export function writeF32(root: TgpuRoot, dst: F32Buffer, data: Float32Array): void {
  rawWrite(root, root.unwrap(dst), data);
}

/** First arg of the root-optional overloads. */
const isRoot = (x: unknown): x is TgpuRoot =>
  typeof x === 'object' && x !== null && 'device' in x && 'createBuffer' in x;

/** Creates a storage buffer and uploads a u32 typed array. */
export function uploadU32(data: Uint32Array): U32Buffer;
export function uploadU32(root: TgpuRoot, data: Uint32Array): U32Buffer;
export function uploadU32(a: TgpuRoot | Uint32Array, b?: Uint32Array): U32Buffer {
  const [root, data] = isRoot(a) ? [a, b!] : [defaultRoot(), a];
  checkBindable(root, data.byteLength, 'uploadU32');
  const buf = root.createBuffer(d.arrayOf(d.u32, data.length)).$usage('storage');
  rawWrite(root, root.unwrap(buf), data);
  return buf;
}

/** Creates a storage buffer and uploads an f32 typed array. */
export function uploadF32(data: Float32Array): F32Buffer;
export function uploadF32(root: TgpuRoot, data: Float32Array): F32Buffer;
export function uploadF32(a: TgpuRoot | Float32Array, b?: Float32Array): F32Buffer {
  const [root, data] = isRoot(a) ? [a, b!] : [defaultRoot(), a];
  checkBindable(root, data.byteLength, 'uploadF32');
  const buf = root.createBuffer(d.arrayOf(d.f32, data.length)).$usage('storage');
  rawWrite(root, root.unwrap(buf), data);
  return buf;
}

/** Makes a materialized f32 Value from CPU data — the way to get model input
 *  onto the device. */
export function tensor(data: Float32Array, shape: readonly [number, number] | ValueMeta): Value;
export function tensor(
  root: TgpuRoot,
  data: Float32Array,
  shape: readonly [number, number] | ValueMeta,
): Value;
export function tensor(
  a: TgpuRoot | Float32Array,
  b: Float32Array | readonly [number, number] | ValueMeta,
  c?: readonly [number, number] | ValueMeta,
): Value {
  const [root, data, shape] = isRoot(a)
    ? [a, b as Float32Array, c!]
    : [defaultRoot(), a, b as readonly [number, number] | ValueMeta];
  const meta = Array.isArray(shape)
    ? matrix(shape[0] as number, shape[1] as number)
    : (shape as ValueMeta);
  // Every kernel indexes by the shape, so a count mismatch is a Value whose
  // meta lies about its own buffer.
  if (data.length !== meta.elems) {
    throw new Error(
      `tensor: data has ${data.length} elements but the shape needs ${meta.elems}` +
        (meta.dims ? ` (dims [${meta.dims.join(', ')}])` : ''),
    );
  }
  // The meta decides the buffer: an f32 buffer under an f16 label would read
  // as garbage in every kernel.
  if (meta.dtype === 'f16') {
    return materialized(meta, uploadF16(root, f32ArrayToF16Bits(data).data));
  }
  if (meta.dtype !== 'f32') {
    throw new Error(`tensor: cannot build a '${meta.dtype}' tensor from f32 data`);
  }
  return materialized(meta, uploadF32(root, data));
}

/** Rounded up to even, so the byte size stays a multiple of 4;
 *  copyBufferToBuffer rejects other sizes. */
const evenElems = (elems: number): number => elems + (elems & 1);

/** Overwrites an f16 storage buffer in place from f32 numbers. */
export function writeF16(root: TgpuRoot, dst: F16Buffer, data: Float32Array): void {
  const bits = f32ArrayToF16Bits(data).data;
  const n = evenElems(bits.length);
  let src = bits;
  if (n !== bits.length) {
    src = new Uint16Array(n);
    src.set(bits);
  }
  rawWrite(root, root.unwrap(dst), src);
}

/** Creates a storage buffer and uploads raw f16 halfwords. An odd-length
 *  tensor is padded by one halfword, which is never read. */
export function uploadF16(root: TgpuRoot, bits: Uint16Array): F16Buffer {
  const n = evenElems(bits.length);
  checkBindable(root, n * 2, 'uploadF16');
  const buf = root.createBuffer(d.arrayOf(d.f16, n)).$usage('storage');
  let src = bits;
  if (n !== bits.length) {
    src = new Uint16Array(n);
    src.set(bits);
  }
  rawWrite(root, root.unwrap(buf), src);
  return buf as F16Buffer;
}

/** Reads `count` f16 values back, widened to f32. */
export async function readbackF16(
  root: TgpuRoot,
  src: GPUBuffer,
  count: number,
): Promise<Float32Array> {
  const bytes = evenElems(count) * 2;
  return withStaging(
    root,
    bytes,
    (enc, staging) => enc.copyBufferToBuffer(src, 0, staging, 0, bytes),
    (mapped) => f16ToF32(new Uint16Array(mapped.slice(0, bytes)).subarray(0, count)),
  );
}

/** dtype to storage buffer for op outputs. Weight-only dtypes are rejected:
 *  a request for one means an output shape is wrong. */
export function createStorageFor(root: TgpuRoot, shape: ValueMeta): GpuBufferRef {
  const { dtype, elems } = shape;
  if (dtype === 'quantW') {
    throw new Error(`eager: ${dtype} buffers are upload-only — no op produces them`);
  }
  checkBindable(root, elems * elemBytes(dtype), `allocating a ${dtype} [${shape.dims ?? elems}]`);
  if (dtype === 'f16') {
    return root.createBuffer(d.arrayOf(d.f16, evenElems(elems))).$usage('storage') as F16Buffer;
  }
  return root.createBuffer(d.arrayOf(d.f32, elems)).$usage('storage');
}

/** Persistent staging buffer per root, grown to the largest readback seen. An
 *  overlapping readback falls back to a transient buffer. */
const stagingPerRoot = new WeakMap<TgpuRoot, { buf: GPUBuffer; size: number; busy: boolean }>();

/** A leased MAP_READ staging buffer, so the copy can ride an existing encoder.
 *  Encode into `staging`, submit, then `finish(read)`. withStaging() below is
 *  the standalone encode-and-submit form. */
export interface StagingLease {
  staging: GPUBuffer;
  finish<T>(read: (mapped: ArrayBuffer) => T): Promise<T>;
}

export function leaseStaging(root: TgpuRoot, byteSize: number): StagingLease {
  const device = root.device;
  let entry = stagingPerRoot.get(root);
  let staging: GPUBuffer;
  let transient = false;
  if (entry === undefined || entry.busy) {
    staging = device.createBuffer({
      size: byteSize,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    if (entry === undefined) {
      entry = { buf: staging, size: byteSize, busy: true };
      stagingPerRoot.set(root, entry);
    } else {
      transient = true; // a readback is already in flight on the shared buffer
    }
  } else {
    if (entry.size < byteSize) {
      entry.buf.destroy();
      entry.buf = device.createBuffer({
        size: byteSize,
        usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
      });
      entry.size = byteSize;
    }
    entry.busy = true;
    staging = entry.buf;
  }
  const shared = entry;
  return {
    staging,
    async finish<T>(read: (mapped: ArrayBuffer) => T): Promise<T> {
      // On every path: a throw must not leave the slot marked busy, which
      // would push every later readback onto a transient buffer.
      try {
        await staging.mapAsync(GPUMapMode.READ, 0, byteSize);
        try {
          return read(staging.getMappedRange(0, byteSize));
        } finally {
          staging.unmap();
        }
      } finally {
        if (transient) staging.destroy();
        else shared.busy = false;
      }
    },
  };
}

async function withStaging<T>(
  root: TgpuRoot,
  byteSize: number,
  encode: (enc: GPUCommandEncoder, staging: GPUBuffer) => void,
  read: (mapped: ArrayBuffer) => T,
): Promise<T> {
  const device = root.device;
  const lease = leaseStaging(root, byteSize);
  const enc = device.createCommandEncoder();
  encode(enc, lease.staging);
  device.queue.submit([enc.finish()]);
  return lease.finish(read);
}

/** Reads `count` f32 values through the shared staging buffer. */
export async function readbackF32(
  root: TgpuRoot,
  src: GPUBuffer,
  count: number,
): Promise<Float32Array> {
  return withStaging(
    root,
    count * 4,
    (enc, staging) => enc.copyBufferToBuffer(src, 0, staging, 0, count * 4),
    (mapped) => new Float32Array(mapped.slice(0, count * 4)),
  );
}

/** Bytes a Value's buffer occupies, and how to widen them on the CPU. Copy
 *  sizes round up to a multiple of 4, which copyBufferToBuffer requires. */
export function shapeSpan(shape: ValueMeta): {
  bytes: number;
  decode: (mapped: ArrayBuffer, offset: number) => Float32Array;
} {
  if (shape.layout !== undefined) {
    throw new Error(
      `readback: Value is stored '${shape.layout}' — convert with toChw() before reading`,
    );
  }
  if (shape.dtype === 'f16') {
    const bytes = (shape.elems + (shape.elems & 1)) * 2;
    return {
      bytes,
      decode: (mapped, offset) =>
        f16ToF32(new Uint16Array(mapped.slice(offset, offset + bytes)).subarray(0, shape.elems)),
    };
  }
  if (shape.dtype !== 'f32') {
    throw new Error(`eager: cannot read back a ${shape.dtype} Value — it is weight storage`);
  }
  const bytes = shape.elems * 4;
  return {
    bytes,
    decode: (mapped, offset) => new Float32Array(mapped.slice(offset, offset + bytes)),
  };
}

/** Copies one materialized buffer back as f32s, with its own submit. */
export function readbackShape(
  root: TgpuRoot,
  buffer: GpuBufferRef,
  shape: ValueMeta,
): Promise<Float32Array> {
  return readbackShapes(root, [{ buffer, shape }]).then((outs) => outs[0]!);
}

/** Several buffers through one staging buffer, submit and map. */
export function readbackShapes(
  root: TgpuRoot,
  items: readonly { buffer: GpuBufferRef; shape: ValueMeta }[],
): Promise<Float32Array[]> {
  return readbackManySpans(
    root,
    items.map(({ buffer, shape }) => ({
      buffer: root.unwrap(buffer),
      ...shapeSpan(shape),
    })),
  );
}

/** One entry of a batched readback. */
export interface ReadbackSpan {
  buffer: GPUBuffer;
  bytes: number;
  decode: (mapped: ArrayBuffer, offset: number) => Float32Array;
}

/** Batched readback of mixed dtypes through one staging buffer. The fixed
 *  per-readback cost dominates multi-output reads, so this stays one round-trip
 *  however many values it reads. */
export async function readbackManySpans(
  root: TgpuRoot,
  spans: readonly ReadbackSpan[],
): Promise<Float32Array[]> {
  const total = spans.reduce((a, s) => a + s.bytes, 0);
  return withStaging(
    root,
    total,
    (enc, staging) => {
      let off = 0;
      for (const s of spans) {
        enc.copyBufferToBuffer(s.buffer, 0, staging, off, s.bytes);
        off += s.bytes;
      }
    },
    (mapped) => {
      const outs: Float32Array[] = [];
      let off = 0;
      for (const s of spans) {
        outs.push(s.decode(mapped, off));
        off += s.bytes;
      }
      return outs;
    },
  );
}

/** Reads several f32 buffers through one staging buffer. */
export async function readbackManyF32(
  root: TgpuRoot,
  srcs: readonly { buffer: GPUBuffer; count: number }[],
): Promise<Float32Array[]> {
  const total = srcs.reduce((a, s) => a + s.count, 0);
  return withStaging(
    root,
    total * 4,
    (enc, staging) => {
      let off = 0;
      for (const s of srcs) {
        enc.copyBufferToBuffer(s.buffer, 0, staging, off * 4, s.count * 4);
        off += s.count;
      }
    },
    (mapped) => {
      const outs: Float32Array[] = [];
      let off = 0;
      for (const s of srcs) {
        outs.push(new Float32Array(mapped.slice(off * 4, (off + s.count) * 4)));
        off += s.count;
      }
      return outs;
    },
  );
}
