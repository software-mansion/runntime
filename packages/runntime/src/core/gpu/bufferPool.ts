import type { GpuBufferRef, ValueMeta } from '../graph/value.ts';

/** Recycles GPU buffers for op intermediates. Once an intermediate's last
 *  consumer has been dispatched its buffer is dead and can back a later op, so
 *  a long chain runs in about two buffers instead of one per op.
 *
 *  Buffers are interchangeable only within the same size and dtype, hence the
 *  bucket key. Allocation is injected, to keep this class free of WebGPU. */
export class BufferPool {
  private free = new Map<string, GpuBufferRef[]>();

  constructor(private readonly createBuffer: (shape: ValueMeta) => GpuBufferRef) {}

  private key(shape: ValueMeta): string {
    return `${shape.dtype}:${shape.elems}`;
  }

  acquire(shape: ValueMeta): GpuBufferRef {
    const bucket = this.free.get(this.key(shape));
    const reused = bucket?.pop();
    return reused ?? this.createBuffer(shape);
  }

  release(buffer: GpuBufferRef, shape: ValueMeta): void {
    const k = this.key(shape);
    const bucket = this.free.get(k);
    if (bucket) bucket.push(buffer);
    else this.free.set(k, [buffer]);
  }

  dispose(): void {
    for (const bucket of this.free.values()) for (const buf of bucket) buf.destroy();
    this.free.clear();
  }
}
