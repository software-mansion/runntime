/** One recorded GPU forward per input length, or bucket. A bucket records its
 *  DAG once; later calls rewrite the input buffers and re-submit, with the
 *  result copy in the same submit, so a call costs one submit and no graph
 *  build.
 *
 *  Calls on one bucket run one at a time, since they share its buffers. */

import type { TgpuRoot } from 'typegpu';
import { type Value } from '../graph/value.ts';
import type { F32Buffer } from '../kernels/common.ts';
import { tensor, writeF16, writeF32 } from './buffers.ts';
import { defaultRoot } from './context.ts';
import { evalValues, gpuExecutor, type RunntimeExecutor } from './eval.ts';

/** Named input planes of one bucket, each a row-major f32 matrix [rows, cols]. */
export type ReplayInputShapes = Record<string, readonly [number, number]>;

export interface ReplayCacheConfig<K extends string> {
  /** The GPU to run on; defaults to the initRunntime() device. */
  root?: TgpuRoot;
  /** Padded lengths with a recording each, strictly ascending. */
  buckets: readonly number[];
  /** Input planes for bucket length T. Same keys for every T. */
  inputs: (T: number) => Record<K, readonly [number, number]>;
  /** Writes valid data for the recording run. Warm-up validation reads the
   *  output, which must be finite and not all-zero. */
  captureFill: (T: number, scratch: Record<K, Float32Array>) => void;
  /** Traces the DAG to record for bucket length T. Runs once per bucket. */
  build: (T: number, inputs: Record<K, Value>) => Value;
  /** Storage dtype per input plane, default f32. Per-plane because a plane
   *  feeding f16 arithmetic must be f16, while index planes stay f32. */
  planeDtype?: Partial<Record<K, 'f32' | 'f16'>>;
}

export interface ReplayCache<K extends string> {
  /** Smallest bucket that fits `len`, or undefined past the largest. */
  pick(len: number): number | undefined;
  /** Replay bucket T, which must be a configured one — route through pick(). */
  run(
    T: number,
    fill: (T: number, scratch: Record<K, Float32Array>) => void,
  ): Promise<Float32Array>;
  /** Builds and validates every bucket. Call it from load-time warm-up so
   *  runtime replays never hit a first-use compile. */
  warmAll(): Promise<void>;
  /** Destroys every bucket's input buffers. The cache is unusable after. */
  dispose(): void;
}

interface Bucket<K extends string> {
  ex: RunntimeExecutor;
  buffers: Record<K, F32Buffer>;
  scratch: Record<K, Float32Array>;
  target: Value;
  replay: () => void;
  /** Tail of the serialization chain — run() appends to it. */
  busy: Promise<unknown>;
}

export function createReplayCache<K extends string>(cfg: ReplayCacheConfig<K>): ReplayCache<K> {
  const { buckets: lengths, inputs, captureFill, build, planeDtype } = cfg;
  if (lengths.length === 0) throw new Error('replayCache: no buckets');
  for (let i = 0; i < lengths.length; i++) {
    const T = lengths[i]!;
    if (!Number.isInteger(T) || T < 1 || (i > 0 && T <= lengths[i - 1]!)) {
      throw new Error(
        `replayCache: buckets must be strictly ascending positive integers, got [${lengths}]`,
      );
    }
  }
  const root = cfg.root ?? defaultRoot();
  const buckets = new Map<number, Bucket<K>>();
  let disposed = false;

  const makeBucket = (T: number): Bucket<K> => {
    // One executor per bucket: capture freezes its buffer pool, so eager work
    // needs a different one. Pipelines are cached per root regardless.
    const ex = gpuExecutor(root);
    const shapes = inputs(T);
    const keys = Object.keys(shapes) as K[];
    const scratch = {} as Record<K, Float32Array>;
    for (const k of keys) {
      const [rows, cols] = shapes[k];
      scratch[k] = new Float32Array(rows * cols);
    }
    captureFill(T, scratch);
    const buffers = {} as Record<K, F32Buffer>;
    const values = {} as Record<K, Value>;
    for (const k of keys) {
      const [rows, cols] = shapes[k];
      const dtype = planeDtype?.[k] ?? 'f32';
      const v = tensor(root, scratch[k], { elems: rows * cols, dtype, dims: [rows, cols] });
      values[k] = v;
      buffers[k] = v.buffer as F32Buffer;
    }
    const cap = ex.captureFrame(() => {
      const target = build(T, values);
      evalValues([target], ex);
      return target;
    });
    return {
      ex,
      buffers,
      scratch,
      target: cap.targets,
      replay: cap.replay,
      busy: Promise.resolve(),
    };
  };

  const bucketFor = (T: number): Bucket<K> => {
    if (disposed) throw new Error('replayCache: disposed');
    if (!lengths.includes(T)) throw new Error(`replayCache: ${T} is not a configured bucket`);
    let b = buckets.get(T);
    if (b === undefined) {
      b = makeBucket(T);
      buckets.set(T, b);
    }
    return b;
  };

  const run: ReplayCache<K>['run'] = (T, fill) => {
    const b = bucketFor(T);
    const job = b.busy.then(() => {
      fill(T, b.scratch);
      for (const k of Object.keys(b.buffers) as K[]) {
        if ((planeDtype?.[k] ?? 'f32') === 'f16')
          writeF16(root, b.buffers[k] as never, b.scratch[k]);
        else writeF32(root, b.buffers[k], b.scratch[k]);
      }
      const result = b.ex.readbackOnSubmit(b.target);
      b.replay();
      return result;
    });
    // The chain must survive a rejected job, or one failure wedges the bucket.
    b.busy = job.catch(() => undefined);
    return job;
  };

  return {
    pick(len) {
      for (const T of lengths) if (len <= T) return T;
      return undefined;
    },
    run,
    async warmAll() {
      for (const T of lengths) await run(T, captureFill);
    },
    dispose() {
      disposed = true;
      for (const b of buckets.values()) {
        for (const k of Object.keys(b.buffers) as K[]) b.buffers[k].destroy();
      }
      buckets.clear();
    },
  };
}
