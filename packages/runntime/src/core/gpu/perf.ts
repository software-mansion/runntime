/** Diagnostic instrumentation for the GPU executor. A GpuPerfSink handed to
 *  gpuExecutor() receives one onSubmit per queue.submit with the dispatch
 *  count (synchronously) and, when the device has 'timestamp-query', a
 *  promise resolving to the submitted work's GPU time. `perOp: true` switches
 *  submit() from one pass to one pass per dispatch with its own timestamp
 *  pair — inflated totals (pass boundaries aren't free) but a faithful
 *  per-kernel ranking. Without a sink the executor's hot path does no perf
 *  work at all. */

export interface GpuOpTime {
  ns: number;
  count: number;
}

export interface GpuSubmitTiming {
  gpuNs: number;
  opNs?: Map<string, GpuOpTime>;
}

export interface GpuPerfSink {
  readonly perOp?: boolean;
  onSubmit(dispatches: number, timing?: Promise<GpuSubmitTiming>): void;
}
