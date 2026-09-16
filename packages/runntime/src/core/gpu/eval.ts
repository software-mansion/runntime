/** The WebGPU Executor for the eager eval driver: every node's dispatch is
 *  batched into one compute pass and submitted once. Per-op kernel selection
 *  lives in gpu/dispatch. */

import { d } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import type { F16Buffer, F32Buffer, FloatBuffer, KernelHandle } from '../kernels/common.ts';
import { checkBindable, createStorageFor, readbackShape, readbackShapes } from './buffers.ts';
import { BufferPool } from './bufferPool.ts';
import { elemFor } from '../kernels/elem.ts';
import {
  type DispatchExtras,
  evalValues as evalValuesWith,
  toArray as toArrayWith,
  type Executor,
  type Readback,
  type Releaser,
} from '../graph/evalCore.ts';
import type { GpuPerfSink, GpuSubmitTiming } from './perf.ts';
import { encodePass } from './encode.ts';
import { encodeTimed, type KernelTiming, profileHandles } from './timing.ts';
import { ReadbackQueue } from './readbackQueue.ts';
import type { GpuBufferRef, Value } from '../graph/value.ts';
import { defaultRoot, executorPerRoot, setDefaultRoot } from './context.ts';
import { copyChHwc4Handle } from '../kernels/conv/hwc4.ts';
import { sliceRowsHandle } from '../kernels/shape/rows.ts';
import { specFor } from './dispatch/registry.ts';
import { concatChannelsHwc4Spec } from './dispatch/conv.ts';
import { reshapeSpec } from './dispatch/shape.ts';
import {
  type DispatchCtx,
  type FloatDtype,
  narrow,
  narrowFloat,
  pipelineFor,
} from './dispatch/spec.ts';

export type { Executor, Readback, Releaser };

/** Evaluates pending Values now, on the initRunntime() executor or on `ex`. */
export function evalValues(targets: readonly Value[], ex?: Executor): void {
  evalValuesWith(targets, ex ?? defaultExecutor());
}

/** Readback fused into the next submit, so a step costs one submit and one map
 *  round-trip rather than two submits. */
export interface FusedReadback {
  /** Resolves after the next submit() or replay(); one request at a time. */
  readbackOnSubmit(value: Value): Promise<Float32Array>;
  /** Several values through one staging copy, resolved together. */
  readbackManyOnSubmit(values: readonly Value[]): Promise<Float32Array[]>;
  /** toArray with the copy fused into the eval's own submit. */
  toArrayFused(value: Value): Promise<Float32Array>;
}

export type RunntimeExecutor = Executor & Readback & Releaser & CaptureExecutor & FusedReadback;

/** Sets the default device and returns its executor. Call it once at startup;
 *  tensor(), toArray() and loadStateDict() then need no root. */
export function initRunntime(root: TgpuRoot): RunntimeExecutor {
  setDefaultRoot(root);
  let ex = executorPerRoot.get(root) as RunntimeExecutor | undefined;
  if (!ex) {
    ex = gpuExecutor(root);
    executorPerRoot.set(root, ex);
  }
  return ex;
}

export function defaultExecutor(): RunntimeExecutor {
  return initRunntime(defaultRoot());
}

/** Read a Value's numbers on the CPU, evaluating it first if still pending. */
export async function toArray(value: Value, ex?: Executor & Readback): Promise<Float32Array> {
  const executor = ex ?? defaultExecutor();
  return 'toArrayFused' in executor
    ? (executor as RunntimeExecutor).toArrayFused(value)
    : toArrayWith(value, executor);
}

export type { KernelTiming };

/** Capture and replay: records one frame's dispatches, then re-submits them
 *  without walking the graph again. Fixed-shape forwards only. */
export interface CaptureExecutor {
  /** Runs `fn` once, recording every dispatch. The executor is replay-only
   *  afterwards. Keep `targets` — their buffers stay valid across replays — and
   *  overwrite inputs in place before each replay. */
  captureFrame<T>(fn: () => T): {
    targets: T;
    replay: () => void;
    /** Replays once with a timestamp pair per dispatch. Per-dispatch passes
     *  remove overlap, so read the sum as relative attribution only. */
    profileReplay: () => Promise<KernelTiming[] | undefined>;
  };
}

/** Builds the executor for one device. Create it once and reuse it; pooled
 *  buffers and memoized bind groups only pay off across calls. */
export function gpuExecutor(root: TgpuRoot, opts: { perf?: GpuPerfSink } = {}): RunntimeExecutor {
  const perf = opts.perf;
  const canTime = root.device.features.has('timestamp-query');
  // The feature alone is not enough: a 16-wide subgroup folds half the sums.
  const subgroupsOk =
    root.device.features.has('subgroups') &&
    ((root.device as GPUDevice & { adapterInfo?: { subgroupMinSize?: number } }).adapterInfo
      ?.subgroupMinSize ?? 0) >= 32;
  const pool = new BufferPool((shape) => createStorageFor(root, shape));
  const handles: KernelHandle[] = [];
  // Stable small ints so buffers and pipelines can key the handle memo.
  const objectIds = new WeakMap<object, number>();
  let nextObjectId = 1;
  const idOf = (o: object): number => {
    let id = objectIds.get(o);
    if (id === undefined) {
      id = nextObjectId++;
      objectIds.set(o, id);
    }
    return id;
  };
  /** Memo key to the handles a dispatch encoded. */
  const handleCache = new Map<string, KernelHandle[]>();

  // Filler for bind slots a pipeline never reads. One per dtype, 4 elements,
  // since vec4 layouts need 16 bytes minimum.
  let dummyF32: F32Buffer | undefined;
  let dummyF16: F16Buffer | undefined;
  function dummy(dtype: 'f32'): F32Buffer;
  function dummy(dtype: 'f16'): F16Buffer;
  function dummy(dtype: FloatDtype): FloatBuffer;
  function dummy(dtype: FloatDtype): FloatBuffer {
    if (dtype === 'f16') {
      dummyF16 ??= root.createBuffer(d.arrayOf(d.f16, 4)).$usage('storage');
      return dummyF16;
    }
    dummyF32 ??= root.createBuffer(d.arrayOf(d.f32, 4)).$usage('storage');
    return dummyF32;
  }

  // Replay-only once captured: a fresh eval would acquire pooled buffers
  // aliasing the captured frame's intermediates.
  let capturing: KernelHandle[] | undefined;
  let frozen = false;

  const readbacks = new ReadbackQueue(root);
  const scratches = new Map<string, { buf: F32Buffer; elems: number }>();
  const ctx: DispatchCtx = {
    root,
    subgroupsOk,
    dummy,
    scratch: (key, elems) => {
      let s = scratches.get(key);
      if (s === undefined || s.elems < elems) {
        checkBindable(root, 4 * elems, `${key} scratch`);
        s = { buf: root.createBuffer(d.arrayOf(d.f32, elems)).$usage('storage'), elems };
        scratches.set(key, s);
      }
      return s.buf;
    },
  };
  /** Everything a dispatch's uniforms and bind group derive from. A decode
   *  loop re-issues the same dispatches against pool-cycled buffers, so a hit
   *  skips every per-step allocation. */
  const memoKey = (
    node: Value,
    pipeline: object,
    inputs: readonly GpuBufferRef[],
    out: GpuBufferRef,
    extras: DispatchExtras | undefined,
  ): string => {
    const p = node.pending!;
    const parts: (string | number)[] = [p.op, idOf(pipeline), idOf(out)];
    for (const b of inputs) parts.push(idOf(b));
    for (const input of p.inputs) parts.push(-1, ...(input.shape.dims ?? [input.shape.elems]));
    parts.push(-2, ...(node.shape.dims ?? [node.shape.elems]));
    if (p.attrs) parts.push(-3, ...p.attrs);
    if (p.scalar !== undefined) parts.push(-4, p.scalar);
    if (extras !== undefined) {
      parts.push(
        -5,
        extras.outBase ?? 0,
        extras.outElems ?? 0,
        extras.addend !== undefined ? idOf(extras.addend) : -1,
      );
    }
    return parts.join(':');
  };

  const executor: RunntimeExecutor = {
    captureFrame: <T>(fn: () => T) => {
      if (frozen || capturing) throw new Error('captureFrame: executor already captured');
      capturing = [];
      const targets = fn();
      const recorded = capturing;
      capturing = undefined;
      frozen = true;
      if (recorded.length === 0) throw new Error('captureFrame: fn() submitted no GPU work');
      return {
        targets,
        replay: () => readbacks.submit((encoder) => encodePass(encoder, recorded)),
        profileReplay: () => profileHandles(root.device, recorded),
      };
    },
    acquireOutput: (shape) => {
      if (frozen) {
        throw new Error(
          'gpuExecutor: executor is replay-only after captureFrame() — use replay(), or a separate executor for other work',
        );
      }
      return pool.acquire(shape);
    },
    releaseOutput: (buffer, shape) => pool.release(buffer, shape),
    dispose: () => pool.dispose(),
    /** Queues one node's kernel handles. Nothing runs until submit(). */
    dispatch: (node, inputs, out, extras) => {
      const spec = specFor(node.pending!.op);
      const attrs = spec.attrs(node.pending!);
      const cfg = spec.cfg(node, attrs, ctx, extras);
      const pipeline = pipelineFor(root, spec, node.shape.dtype, cfg);
      const key =
        spec.memo === false ? undefined : memoKey(node, pipeline as object, inputs, out, extras);
      const cached = key === undefined ? undefined : handleCache.get(key);
      if (cached !== undefined) {
        handles.push(...cached);
        return;
      }
      const encoded = spec.encode({ node, attrs, cfg, pipeline, inputs, out, extras, ctx });
      handles.push(...encoded);
      if (key !== undefined) handleCache.set(key, encoded);
    },
    /** Encodes every queued handle into one pass and submits once. */
    submit: () => {
      if (handles.length === 0) return;
      const n = handles.length;
      let readTiming: (() => Promise<GpuSubmitTiming>) | undefined;
      readbacks.submit((encoder) => {
        if (perf !== undefined && canTime) {
          readTiming = encodeTimed(root.device, encoder, handles, perf.perOp === true);
        } else {
          encodePass(encoder, handles);
        }
      });
      if (capturing) capturing.push(...handles);
      handles.length = 0;
      perf?.onSubmit(n, readTiming?.());
    },
    /** Copies one concat input the driver could not elide; hwc4 needs a
     *  strided copy. */
    copyInto: (src, dst, srcShape, dstShape, dstBase) => {
      if (srcShape.layout === 'hwc4') {
        const [, h, w] = srcShape.dims! as number[];
        const pElems = h! * w!;
        handles.push(
          copyChHwc4Handle(
            root,
            pipelineFor(root, concatChannelsHwc4Spec, 'f16', undefined),
            {
              pElems,
              srcC4: srcShape.elems / (4 * pElems),
              dstC4: dstShape.elems / (4 * pElems),
              srcOffB: 0,
              dstOffB: dstBase / (4 * pElems),
              nB: srcShape.elems / (4 * pElems),
            },
            { x: narrow(src, 'f16'), out: narrow(dst, 'f16') },
          ),
        );
        return;
      }
      handles.push(
        sliceRowsHandle(
          root,
          pipelineFor(root, reshapeSpec, srcShape.dtype, undefined),
          { total: srcShape.elems, cols: srcShape.elems, start: 0, outBase: dstBase },
          { x: narrowFloat(src, srcShape.dtype), out: narrowFloat(dst, srcShape.dtype) },
          elemFor(srcShape.dtype),
        ),
      );
    },
    releaseValues: (values) => {
      if (frozen) {
        throw new Error(
          'releaseValues: captured targets keep their buffers — a replay-only executor never releases',
        );
      }
      for (const v of values) {
        if (v.state !== 'materialized') continue;
        pool.release(v.buffer, v.shape);
        v.markReleased();
      }
    },
    readback: (buffer, shape) => readbackShape(root, buffer, shape),
    readbackMany: (items) => readbackShapes(root, items),
    readbackManyOnSubmit: (values) => readbacks.request(values),
    readbackOnSubmit: (value) => executor.readbackManyOnSubmit([value]).then((outs) => outs[0]!),
    toArrayFused: (value) => {
      if (value.state === 'materialized') {
        return executor.readback(value.buffer, value.shape);
      }
      const result = executor.readbackOnSubmit(value);
      // Without a handler, the rejection below would surface alongside a throw
      // from the eval itself.
      result.catch(() => undefined);
      try {
        evalValuesWith([value], executor);
      } catch (err) {
        if (readbacks.isPendingFor(value)) readbacks.rejectPending(err);
        throw err;
      }
      return result;
    },
  };
  return executor;
}
