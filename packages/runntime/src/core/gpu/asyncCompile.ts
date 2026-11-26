/** Async pipeline compile for the load-time warm-up. A TypeGPU pipeline
 *  compiles on its first use in a pass, which blocks the page. During a
 *  warm-up the executor compiles with `initAsync()` instead and submits once
 *  that resolves. */

import { RunntimeError } from '../error.ts';

/** A TypeGPU compute or render pipeline. */
interface Pipeline {
  initAsync(): Promise<void>;
}

let enabled = true;
/** Warm-ups running on each device. */
const warmUps = new WeakMap<GPUDevice, number>();
/** Pipelines compilePipelines() has compiled. */
const compiled = new WeakSet<object>();
/** Pipelines compilePipelines() is compiling. */
const compiling = new Set<object>();

/** Internal switch for load-time comparisons: with `false`, a warm-up
 *  compiles each pipeline on its first use, blocking. Default `true`. */
export function setAsyncCompile(on: boolean): void {
  enabled = on;
}

/** Runs `fn` as warm-up work on `device`: a submit made before `fn` returns,
 *  or before its promise settles, compiles its pipelines without blocking. */
export function withAsyncCompile<T>(device: GPUDevice, fn: () => Promise<T>): Promise<T>;
export function withAsyncCompile<T>(device: GPUDevice, fn: () => T): T;
export function withAsyncCompile(device: GPUDevice, fn: () => unknown): unknown {
  if (!enabled) return fn();
  warmUps.set(device, (warmUps.get(device) ?? 0) + 1);
  const leave = () => {
    warmUps.set(device, warmUps.get(device)! - 1);
  };
  let result: unknown;
  try {
    result = fn();
  } catch (err) {
    leave();
    throw err;
  }
  if (result instanceof Promise) return result.finally(leave);
  leave();
  return result;
}

/** Whether a submit of `handles` on `device` has to wait for
 *  compilePipelines(): in a warm-up when a pipeline is not compiled yet, and
 *  at any time when one is mid-compile, since TypeGPU throws on using it. */
export function needsAsyncCompile(
  device: GPUDevice,
  handles: readonly { readonly pipeline: object }[],
): boolean {
  if ((warmUps.get(device) ?? 0) > 0) return handles.some((h) => !compiled.has(h.pipeline));
  return compiling.size > 0 && handles.some((h) => compiling.has(h.pipeline));
}

async function compile(pipeline: Pipeline): Promise<void> {
  compiling.add(pipeline);
  try {
    await pipeline.initAsync();
    compiled.add(pipeline);
  } finally {
    compiling.delete(pipeline);
  }
}

/** Resolves in a later task, so the page handles input and paints first. A
 *  message, not a timer: a hidden tab runs timers once a second. */
function nextTask(): Promise<void> {
  return new Promise((resolve) => {
    const { port1, port2 } = new MessageChannel();
    port1.onmessage = () => {
      port1.close();
      resolve();
    };
    port2.postMessage(undefined);
  });
}

/** Compiles each pipeline once, without blocking. The compiles overlap, but
 *  each starts in a task of its own: initAsync() resolves the pipeline's WGSL
 *  before it returns, and that is the part that runs on the main thread.
 *  @throws RunntimeError `EXECUTION_FAILED` when a pipeline does not compile. */
export async function compilePipelines(pipelines: Iterable<Pipeline>): Promise<void> {
  if (!enabled) return;
  const pending = [...new Set(pipelines)].filter((p) => !compiled.has(p));
  const compiles: Promise<void>[] = [];
  for (const [i, pipeline] of pending.entries()) {
    if (i > 0) await nextTask();
    const started = compile(pipeline);
    // Handled by the Promise.all below, which a later task attaches.
    started.catch(() => undefined);
    compiles.push(started);
  }
  try {
    await Promise.all(compiles);
  } catch (err) {
    throw new RunntimeError(
      'EXECUTION_FAILED',
      `GPU error during pipeline compile: ${err instanceof Error ? err.message : String(err)}`,
      { cause: err },
    );
  }
}
