import type { GpuBufferRef, ValueMeta } from '../graph/value.ts';

/** Recycles GPU buffers for op intermediates. Once an intermediate's last
 *  consumer has been dispatched its buffer is dead and can back a later op, so
 *  a long chain runs in about two buffers instead of one per op.
 *
 *  Buffers are interchangeable only within the same size and dtype, hence the
 *  bucket key. A model whose input length varies leaves a set of free buckets
 *  behind for every length it has seen; trim() drops the ones its last call
 *  did not use. Allocation is injected, to keep this class free of WebGPU. */
export class BufferPool {
  private free = new Map<string, GpuBufferRef[]>();
  /** Bucket keys acquired since the last trim(). */
  private used = new Set<string>();

  constructor(private readonly createBuffer: (shape: ValueMeta) => GpuBufferRef) {}

  private key(shape: ValueMeta): string {
    return `${shape.dtype}:${shape.elems}`;
  }

  acquire(shape: ValueMeta): GpuBufferRef {
    const k = this.key(shape);
    this.used.add(k);
    const reused = this.free.get(k)?.pop();
    return reused ?? this.createBuffer(shape);
  }

  release(buffer: GpuBufferRef, shape: ValueMeta): void {
    const k = this.key(shape);
    const bucket = this.free.get(k);
    if (bucket) bucket.push(buffer);
    else this.free.set(k, [buffer]);
  }

  /** Destroys the free buffers of every bucket not acquired since the last
   *  trim. Returns whether it destroyed any. */
  trim(): boolean {
    let destroyed = false;
    for (const [k, bucket] of this.free) {
      if (this.used.has(k)) continue;
      for (const buf of bucket) buf.destroy();
      this.free.delete(k);
      destroyed ||= bucket.length > 0;
    }
    this.used.clear();
    return destroyed;
  }

  dispose(): void {
    for (const bucket of this.free.values()) for (const buf of bucket) buf.destroy();
    this.free.clear();
    this.used.clear();
  }
}
