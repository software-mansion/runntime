/** Readback fused into a submit: the staging copy for a set of Values
 *  rides the same command buffer as the compute pass that produces them.
 *  One request may be pending at a time; the next submit consumes it. */

import type { TgpuRoot } from 'typegpu';
import type { Value } from '../graph/value.ts';
import { leaseStaging, shapeSpan } from './buffers.ts';

interface PendingReadback {
  values: readonly Value[];
  resolve: (a: Float32Array[]) => void;
  reject: (e: unknown) => void;
}

export class ReadbackQueue {
  private pending: PendingReadback | undefined;

  constructor(private readonly root: TgpuRoot) {}

  request(values: readonly Value[]): Promise<Float32Array[]> {
    if (this.pending !== undefined) {
      throw new Error('readbackManyOnSubmit: a readback is already queued for the next submit');
    }
    return new Promise<Float32Array[]>((resolve, reject) => {
      this.pending = { values, resolve, reject };
    });
  }

  isPendingFor(value: Value): boolean {
    return this.pending?.values[0] === value;
  }

  rejectPending(err: unknown): void {
    if (this.pending === undefined) return;
    const rb = this.pending;
    this.pending = undefined;
    rb.reject(err);
  }

  submit(build: (encoder: GPUCommandEncoder) => void): void {
    const encoder = this.root.device.createCommandEncoder();
    let finish: ((err?: unknown) => void) | undefined;
    try {
      build(encoder);
      finish = this.encodePending(encoder);
      this.root.device.queue.submit([encoder.finish()]);
    } catch (err) {
      if (finish !== undefined) finish(err);
      else this.rejectPending(err);
      throw err;
    }
    finish?.();
  }

  private encodePending(encoder: GPUCommandEncoder): ((err?: unknown) => void) | undefined {
    if (this.pending === undefined) return undefined;
    const rb = this.pending;
    this.pending = undefined;
    let lease: ReturnType<typeof leaseStaging>;
    const spans: ReturnType<typeof shapeSpan>[] = [];
    let total = 0;
    try {
      for (const v of rb.values) {
        const span = shapeSpan(v.shape);
        spans.push(span);
        total += span.bytes;
      }
      lease = leaseStaging(this.root, total);
      let offset = 0;
      rb.values.forEach((v, i) => {
        encoder.copyBufferToBuffer(
          this.root.unwrap(v.buffer),
          0,
          lease.staging,
          offset,
          spans[i]!.bytes,
        );
        offset += spans[i]!.bytes;
      });
    } catch (err) {
      rb.reject(err);
      return undefined;
    }
    return (err?: unknown) => {
      if (err !== undefined) {
        rb.reject(err);
        void lease.finish(() => undefined).catch(() => undefined);
        return;
      }
      lease
        .finish((mapped) => {
          const outs: Float32Array[] = [];
          let offset = 0;
          for (const span of spans) {
            outs.push(span.decode(mapped, offset));
            offset += span.bytes;
          }
          return outs;
        })
        .then(rb.resolve, rb.reject);
    };
  }
}
