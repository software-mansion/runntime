import type { GpuBufferRef, ValueMeta } from '../graph/value.ts';

/** Recycles GPU buffers for op intermediates. Once an intermediate's last
 *  consumer has been dispatched its buffer is dead and can back a later op, so
 *  a long chain runs in about two buffers instead of one per op.
 *
 *  Buffers are interchangeable only within the same size and dtype, hence the
 *  bucket key. Allocation is injected, to keep this class free of WebGPU. */
export class BufferPool {
  private free = new Map<string, GpuBufferRef[]>();
  /** Every buffer this pool made, free or in use, so dispose() frees the ones
   *  a captured frame or a caller still holds. */
  private made: GpuBufferRef[] = [];

  constructor(private readonly createBuffer: (shape: ValueMeta) => GpuBufferRef) {}

  private key(shape: ValueMeta): string {
    return `${shape.dtype}:${shape.elems}`;
  }

  acquire(shape: ValueMeta): GpuBufferRef {
    const bucket = this.free.get(this.key(shape));
    const reused = bucket?.pop();
    if (reused !== undefined) return reused;
    const made = this.createBuffer(shape);
    this.made.push(made);
    return made;
  }

  release(buffer: GpuBufferRef, shape: ValueMeta): void {
    const k = this.key(shape);
    const bucket = this.free.get(k);
    if (bucket) bucket.push(buffer);
    else this.free.set(k, [buffer]);
  }

  /** Destroys every buffer the pool made, including those still in use. */
  dispose(): void {
    for (const buf of this.made.splice(0)) buf.destroy();
    this.free.clear();
  }
}
