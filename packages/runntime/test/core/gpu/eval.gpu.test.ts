/** gpuExecutor's buffer lifetimes: intermediates go back to its pool, a warm
 *  graph allocates nothing, dispose() frees every buffer it made, and trim()
 *  drops what the last call did not use. */
import { describe, expect, it, onTestFinished } from 'vitest';
import { writeF32 } from '../../../src/core/gpu/buffers.ts';
import {
  evalValues,
  gpuExecutor,
  toArray,
  type RunntimeExecutor,
} from '../../../src/core/gpu/eval.ts';
import {
  add,
  gatherRows,
  layerNorm,
  matmul,
  mul,
  softmax,
  tanh,
} from '../../../src/core/graph/ops.ts';
import { matrix, type Value } from '../../../src/core/graph/value.ts';
import type { F32Buffer } from '../../../src/core/kernels/common.ts';
import {
  buffersSince,
  expectClose,
  gpu,
  liveBuffers,
  randomF32,
  repeated,
  upload,
} from '../../gpu.ts';

const { root } = await gpu();

/** Input rows and columns: small, since only buffer counts matter here. */
const ROWS = 8;
const COLS = 16;
/** Runs before the count must hold still: the first fills the pool. */
const WARM_RUNS = 3;
/** Runs after warm-up that must not allocate a single buffer. */
const STEADY_RUNS = 20;

interface Inputs {
  x: Value;
  /** [1, COLS], a norm weight. */
  w: Value;
  /** [COLS, COLS], a matmul weight. */
  m: Value;
}

interface GraphCase {
  name: string;
  build(inputs: Inputs): Value;
  /** Buffers a known leak, not fixed yet, leaves behind on every warm run. */
  leftPerRun?: string[];
  /** Buffers a known leak, not fixed yet, leaves behind after dispose. */
  leftOnDispose?: string[];
  /** Buffers a known leak, not fixed yet, leaves behind after a captured
   *  frame is disposed. */
  leftAfterCapture?: string[];
}

/** A 16 B dummy the executor binds for a missing optional input. */
const DUMMY = '16 B, usage 140';
/** A kernel's uniform buffer, made by the kernel, not the executor. */
const UNIFORM = '16 B, usage 76';
/** The [ROWS, COLS] f32 result a captured frame holds. */
const TARGET = '512 B, usage 140';
/** The 3-row gatherRows index list. */
const INDEX_LIST = '12 B, usage 140';

const GRAPHS: GraphCase[] = [
  {
    name: 'an elementwise chain',
    build: ({ x }) => tanh(add(mul(x, x), 1)),
    // dispose() keeps the captured target, which never went back to the pool.
    leftAfterCapture: [TARGET],
  },
  {
    name: 'layerNorm without a bias',
    build: ({ x, w }) => layerNorm(add(x, 1), w),
    // dispose() keeps the dummy bias, the layerNorm uniforms and a captured target.
    leftOnDispose: [DUMMY, UNIFORM],
    leftAfterCapture: [DUMMY, TARGET, UNIFORM],
  },
  {
    name: 'softmax',
    build: ({ x }) => softmax(add(x, 1)),
    // dispose() keeps the dummy segments buffer and a captured target.
    leftOnDispose: [DUMMY],
    leftAfterCapture: [DUMMY, TARGET],
  },
  {
    name: 'matmul',
    build: ({ x, m }) => tanh(matmul(x, m)),
    leftAfterCapture: [TARGET],
  },
  {
    name: 'gatherRows',
    build: ({ x }) => gatherRows(add(x, 1), [3, 0, 3]),
    // gatherRows uploads its index list on every dispatch and never destroys it.
    leftPerRun: [INDEX_LIST],
    leftOnDispose: [INDEX_LIST],
    // The captured target is the 3 gathered rows.
    leftAfterCapture: [INDEX_LIST, '192 B, usage 140'],
  },
];

function inputs(): Inputs {
  return {
    x: upload(root, randomF32(ROWS * COLS, 1), [ROWS, COLS]),
    w: upload(root, randomF32(COLS, 2), [1, COLS]),
    m: upload(root, randomF32(COLS * COLS, 3), [COLS, COLS]),
  };
}

/** One eval, read back and handed to the pool, so only intermediates and
 *  executor state can be left over. */
async function run(ex: RunntimeExecutor, out: Value): Promise<void> {
  evalValues([out], ex);
  await ex.readback(out.buffer, out.shape);
  ex.releaseValues([out]);
}

describe.for(GRAPHS)('gpuExecutor with $name', (graph) => {
  it('allocates nothing once warm', async () => {
    const values = inputs();
    const ex = gpuExecutor(root);
    try {
      for (let i = 0; i < WARM_RUNS; i++) await run(ex, graph.build(values));
      const before = liveBuffers();
      for (let i = 0; i < STEADY_RUNS; i++) await run(ex, graph.build(values));
      expect(buffersSince(before)).toEqual(repeated(graph.leftPerRun ?? [], STEADY_RUNS));
    } finally {
      ex.dispose();
    }
  });

  it('frees every buffer it made on dispose', async () => {
    const values = inputs();
    // A throwaway executor first: pipelines and the readback staging buffer
    // belong to the root and outlive every executor.
    const first = gpuExecutor(root);
    await run(first, graph.build(values));
    first.dispose();

    const before = liveBuffers();
    const ex = gpuExecutor(root);
    await run(ex, graph.build(values));
    ex.dispose();
    expect(buffersSince(before)).toEqual(graph.leftOnDispose ?? []);
  });

  it('frees a captured frame on dispose', async () => {
    const values = inputs();
    const first = gpuExecutor(root);
    await run(first, graph.build(values));
    first.dispose();

    const before = liveBuffers();
    const ex = gpuExecutor(root);
    const { targets, replay } = ex.captureFrame(() => {
      const out = graph.build(values);
      evalValues([out], ex);
      return out;
    });
    const read = ex.readbackOnSubmit(targets);
    replay();
    await read;
    ex.dispose();
    expect(buffersSince(before)).toEqual(graph.leftAfterCapture ?? []);
  });
});

const K = 8;
const a = randomF32(K * K, 2);
const b = randomF32(K * K, 3);

/** (x · a) · b in f64, x [rows, K]. */
function ref(x: Float32Array, rows: number): Float64Array {
  const mul = (l: ArrayLike<number>, r: ArrayLike<number>) => {
    const out = new Float64Array(rows * K);
    for (let i = 0; i < rows; i++) {
      for (let j = 0; j < K; j++) {
        let s = 0;
        for (let k = 0; k < K; k++) s += l[i * K + k]! * r[k * K + j]!;
        out[i * K + j] = s;
      }
    }
    return out;
  };
  return mul(mul(x, a), b);
}

/** (x · a) · b on the GPU. x · a is a pooled intermediate. */
function forward(x: Value): Value {
  return matmul(matmul(x, upload(root, a, matrix(K, K))), upload(root, b, matrix(K, K)));
}

function freshExecutor() {
  const ex = gpuExecutor(root);
  onTestFinished(() => ex.dispose());
  return ex;
}

describe('gpuExecutor trim', () => {
  // Each call trims after it, the way a model with varying input lengths
  // runs. A length unused by the previous call has had its buffers destroyed.
  it.for<{ name: string; lengths: number[] }>([
    { name: 'the same length again', lengths: [5, 5, 5] },
    { name: 'a new length, then the first again', lengths: [5, 9, 5] },
    { name: 'a new length every call', lengths: [3, 7, 12, 1, 20] },
  ])('$name', async ({ lengths }) => {
    const ex = freshExecutor();
    for (const [i, rows] of lengths.entries()) {
      const x = randomF32(rows * K, 10 + i);
      const got = await toArray(forward(upload(root, x, matrix(rows, K))), ex);
      ex.trim();
      expectClose(got, ref(x, rows));
    }
  });

  it('does nothing once captured, so the replay keeps its buffers', async () => {
    const ex = freshExecutor();
    const rows = 4;
    const x = upload(root, randomF32(rows * K, 20), matrix(rows, K));
    const { targets: y, replay } = ex.captureFrame(() => {
      const y = forward(x);
      evalValues([y], ex);
      return y;
    });
    // Two trims: the first keeps what the capture used, the second would
    // destroy the captured intermediate if trim ran on a captured executor.
    ex.trim();
    ex.trim();
    const next = randomF32(rows * K, 21);
    writeF32(root, x.buffer as F32Buffer, next);
    root.device.pushErrorScope('validation');
    const got = ex.readbackOnSubmit(y);
    replay();
    expect(await root.device.popErrorScope()).toBeNull();
    expectClose(await got, ref(next, rows));
  });
});
