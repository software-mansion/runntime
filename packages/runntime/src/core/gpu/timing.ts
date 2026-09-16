/** Timestamp-query timing: the per-submit timing a GpuPerfSink receives,
 *  and the per-kernel profile of a captured frame. Both need the
 *  'timestamp-query' device feature. */

import type { KernelHandle } from '../kernels/common.ts';
import { encodeHandle, encodePass } from './encode.ts';
import type { GpuOpTime, GpuSubmitTiming } from './perf.ts';

interface TimestampQuery {
  set: GPUQuerySet;
  resolve: GPUBuffer;
  staging: GPUBuffer;
  pairs: number;
}

function createQuery(device: GPUDevice, pairs: number): TimestampQuery {
  return {
    set: device.createQuerySet({ type: 'timestamp', count: pairs * 2 }),
    resolve: device.createBuffer({
      size: pairs * 16,
      usage: GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC,
    }),
    staging: device.createBuffer({
      size: pairs * 16,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    }),
    pairs,
  };
}

function timestampWrites(query: TimestampQuery, pair: number): GPUComputePassDescriptor {
  return {
    timestampWrites: {
      querySet: query.set,
      beginningOfPassWriteIndex: pair * 2,
      endOfPassWriteIndex: pair * 2 + 1,
    },
  };
}

function encodeResolve(encoder: GPUCommandEncoder, query: TimestampQuery): void {
  encoder.resolveQuerySet(query.set, 0, query.pairs * 2, query.resolve, 0);
  encoder.copyBufferToBuffer(query.resolve, 0, query.staging, 0, query.pairs * 16);
}

async function readSpans(query: TimestampQuery): Promise<number[]> {
  await query.staging.mapAsync(GPUMapMode.READ);
  const ts = new BigUint64Array(query.staging.getMappedRange().slice(0));
  query.staging.unmap();
  query.staging.destroy();
  query.resolve.destroy();
  query.set.destroy();
  const spans: number[] = [];
  for (let i = 0; i < query.pairs; i++) {
    spans.push(Math.max(0, Number(ts[i * 2 + 1]! - ts[i * 2]!)));
  }
  return spans;
}

const MAX_TIMED_PASSES = 2048;

function encodePerDispatch(
  device: GPUDevice,
  encoder: GPUCommandEncoder,
  handles: readonly KernelHandle[],
): { timed: number; read: () => Promise<number[]> } {
  const timed = Math.min(handles.length, MAX_TIMED_PASSES);
  const query = createQuery(device, timed);
  handles.forEach((h, i) => {
    const pass = encoder.beginComputePass(i < timed ? timestampWrites(query, i) : {});
    encodeHandle(pass, h);
    pass.end();
  });
  encodeResolve(encoder, query);
  return { timed, read: () => readSpans(query) };
}

export function encodeTimed(
  device: GPUDevice,
  encoder: GPUCommandEncoder,
  handles: readonly KernelHandle[],
  perOp: boolean,
): () => Promise<GpuSubmitTiming> {
  if (!perOp) {
    const query = createQuery(device, 1);
    encodePass(encoder, handles, timestampWrites(query, 0));
    encodeResolve(encoder, query);
    return () => readSpans(query).then((spans) => ({ gpuNs: spans[0]! }));
  }
  const { timed, read } = encodePerDispatch(device, encoder, handles);
  return () =>
    read().then((spans) => {
      let gpuNs = 0;
      const opNs = new Map<string, GpuOpTime>();
      for (let i = 0; i < timed; i++) {
        const name = handles[i]!.name;
        const ns = spans[i]!;
        gpuNs += ns;
        const entry = opNs.get(name);
        if (entry) {
          entry.ns += ns;
          entry.count += 1;
        } else {
          opNs.set(name, { ns, count: 1 });
        }
      }
      return { gpuNs, opNs };
    });
}

export interface KernelTiming {
  name: string;
  workgroups: number | readonly [number, number];
  ns: number;
}

export async function profileHandles(
  device: GPUDevice,
  handles: readonly KernelHandle[],
): Promise<KernelTiming[] | undefined> {
  if (!device.features.has('timestamp-query')) return undefined;
  const timings: KernelTiming[] = [];
  for (let start = 0; start < handles.length; start += MAX_TIMED_PASSES) {
    const chunk = handles.slice(start, start + MAX_TIMED_PASSES);
    const encoder = device.createCommandEncoder();
    const { read } = encodePerDispatch(device, encoder, chunk);
    device.queue.submit([encoder.finish()]);
    const spans = await read();
    chunk.forEach((h, i) =>
      timings.push({ name: h.name, workgroups: h.workgroups, ns: spans[i]! }),
    );
  }
  return timings;
}
