/** gpuExecutor's buffer lifetimes: intermediates go back to its pool, a warm
 *  graph allocates nothing, dispose() frees every buffer it made, and trim()
 *  drops what the last call did not use. And its submits in a warm-up, which
 *  wait for an async pipeline compile and keep their order. */
import tgpu from 'typegpu';
import { afterEach, describe, expect, it, onTestFinished, vi } from 'vitest';
import { setAsyncCompile, withAsyncCompile } from '../../../src/core/gpu/asyncCompile.ts';
import { tensor, writeF32 } from '../../../src/core/gpu/buffers.ts';
import { warmUp } from '../../../src/core/gpu/errorScopes.ts';
import {
  evalValues,
  gpuExecutor,
  toArray,
  type RunntimeExecutor,
} from '../../../src/core/gpu/eval.ts';
import {
  add,
  argmaxDot,
  conv1d,
  gatherRows,
  gatherRowsFrom,
  layerNorm,
  matmul,
  mul,
  sdpa,
  softmax,
  swigluChunk,
  tanh,
  writeRows,
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
  requireFeature,
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
  /** [3, 1] f32 row indices, as gatherRowsFrom takes them. */
  idx: Value;
}

interface GraphCase {
  name: string;
  /** The lists below are for this feature's route. Without it, the case
   *  takes another route and leaves other buffers. */
  needs?: GPUFeatureName;
  build(inputs: Inputs): Value;
  /** Buffers a known leak, not fixed yet, leaves behind on every warm run. */
  leftPerRun?: string[];
  /** Buffers a known leak, not fixed yet, leaves behind after dispose. */
  leftOnDispose?: string[];
  /** Buffers a known leak, not fixed yet, leaves behind after a captured
   *  frame is disposed. */
  leftAfterCapture?: string[];
}

/** The 3-row gatherRows index list. */
const INDEX_LIST = '12 B, usage 140';

/** Attention with every key visible, q [T, COLS] against k and v
 *  [COLS, COLS]: one head over all COLS columns. */
const FULL_ATTENTION = {
  qHeads: 1,
  kvHeads: 1,
  headDim: COLS,
  windowLeft: Infinity,
  windowRight: Infinity,
};
/** The same over two heads of COLS / 2, a head size flash does not take. */
const TWO_HEAD_ATTENTION = { ...FULL_ATTENTION, qHeads: 2, kvHeads: 2, headDim: COLS / 2 };

const GRAPHS: GraphCase[] = [
  {
    name: 'an elementwise chain',
    build: ({ x }) => tanh(add(mul(x, x), 1)),
  },
  {
    name: 'layerNorm without a bias',
    build: ({ x, w }) => layerNorm(add(x, 1), w),
  },
  {
    name: 'softmax',
    build: ({ x }) => softmax(add(x, 1)),
  },
  {
    name: 'matmul',
    build: ({ x, m }) => tanh(matmul(x, m)),
  },
  {
    name: 'gatherRows',
    build: ({ x }) => gatherRows(add(x, 1), [3, 0, 3]),
    // gatherRows uploads its index list on every dispatch and never destroys it.
    leftPerRun: [INDEX_LIST],
    leftOnDispose: [INDEX_LIST],
    leftAfterCapture: [INDEX_LIST],
  },
  {
    name: 'flash attention',
    needs: 'subgroups',
    // flash: qLen > 1, subgroups, headDim % 16 == 0
    build: ({ x, m }) => sdpa(x, m, m, FULL_ATTENTION),
  },
  {
    name: 'rows split attention',
    needs: 'subgroups',
    // rowsSubgroupSplit: qLen > 1, f32, subgroups, headDim % 16 != 0
    build: ({ x, m }) => sdpa(x, m, m, TWO_HEAD_ATTENTION),
  },
  {
    name: 'decode attention',
    // decode: qLen == 1, kvLen <= ATTN_DECODE_CHUNK
    build: ({ w, m }) => sdpa(w, m, m, FULL_ATTENTION),
  },
  {
    name: 'conv1d',
    build: ({ x, m }) => conv1d(x, m, { kernelSize: 1, stride: 1 }),
  },
  {
    name: 'swigluChunk',
    build: ({ x }) => swigluChunk(add(x, 1)),
  },
  {
    name: 'argmaxDot',
    build: ({ w, m }) => argmaxDot(m, w),
  },
  {
    name: 'gatherRowsFrom',
    build: ({ x, idx }) => gatherRowsFrom(add(x, 1), idx),
  },
  {
    name: 'writeRows',
    // Writes into x in place, as a KV-cache append does, then reads it: a
    // released writeRows result would hand x's buffer to the pool.
    build: ({ x, w }) => tanh(writeRows(x, add(w, 1), 2)),
  },
];

function inputs(): Inputs {
  return {
    x: upload(root, randomF32(ROWS * COLS, 1), [ROWS, COLS]),
    w: upload(root, randomF32(COLS, 2), [1, COLS]),
    m: upload(root, randomF32(COLS * COLS, 3), [COLS, COLS]),
    idx: upload(root, new Float32Array([3, 0, 3]), [3, 1]),
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
  it('allocates nothing once warm', async (ctx) => {
    if (graph.needs) requireFeature(ctx, root, graph.needs);
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

  it('frees every buffer it made on dispose', async (ctx) => {
    if (graph.needs) requireFeature(ctx, root, graph.needs);
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

  it('runs again after dispose', async () => {
    const values = inputs();
    const ex = gpuExecutor(root);
    try {
      await run(ex, graph.build(values));
      ex.dispose();
      await run(ex, graph.build(values));
    } finally {
      ex.dispose();
    }
  });

  it('runs again after a captured frame is disposed', async () => {
    const values = inputs();
    const ex = gpuExecutor(root);
    try {
      const { replay } = ex.captureFrame(() => {
        const out = graph.build(values);
        evalValues([out], ex);
        return out;
      });
      replay();
      ex.dispose();
      expect(replay).toThrow('replay after dispose()');
      await run(ex, graph.build(values));
    } finally {
      ex.dispose();
    }
  });

  it('frees a captured frame on dispose', async (ctx) => {
    if (graph.needs) requireFeature(ctx, root, graph.needs);
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

describe('gpuExecutor in a warm-up', () => {
  afterEach(() => setAsyncCompile(true));

  /** [[1, 2], [3, 4]] times [[5, 6], [7, 8]]. */
  const PRODUCT = [19, 22, 43, 50];

  /** A device of its own, so no pipeline is compiled yet, with the two
   *  inputs of PRODUCT and a spy on each way a pipeline can compile. */
  async function freshDevice() {
    const root = await tgpu.init({ device: { optionalFeatures: ['subgroups', 'shader-f16'] } });
    onTestFinished(() => root.destroy());
    return {
      root,
      device: root.device,
      ex: gpuExecutor(root),
      a: tensor(root, new Float32Array([1, 2, 3, 4]), [2, 2]),
      b: tensor(root, new Float32Array([5, 6, 7, 8]), [2, 2]),
      blocking: vi.spyOn(root.device, 'createComputePipeline'),
      async: vi.spyOn(root.device, 'createComputePipelineAsync'),
    };
  }

  it('compiles without blocking in warmUp()', async () => {
    const { device, ex, a, b, blocking, async } = await freshDevice();
    const got = await warmUp(device, () => toArray(matmul(a, b), ex));
    expectClose(got, PRODUCT);
    expect(async).toHaveBeenCalledTimes(1);
    expect(blocking).not.toHaveBeenCalled();
  });

  it('compiles on first use outside a warm-up', async () => {
    const { ex, a, b, blocking, async } = await freshDevice();
    expectClose(await toArray(matmul(a, b), ex), PRODUCT);
    expect(blocking).toHaveBeenCalledTimes(1);
    expect(async).not.toHaveBeenCalled();
  });

  it('compiles on first use in a warm-up after setAsyncCompile(false)', async () => {
    setAsyncCompile(false);
    const { device, ex, a, b, blocking, async } = await freshDevice();
    expectClose(await warmUp(device, () => toArray(matmul(a, b), ex)), PRODUCT);
    expect(blocking).toHaveBeenCalledTimes(1);
    expect(async).not.toHaveBeenCalled();
  });

  it('runs two submits in order, the second reading the first', async () => {
    const { device, ex, a, b, blocking } = await freshDevice();
    await withAsyncCompile(device, async () => {
      const y = matmul(a, b);
      evalValues([y], ex);
      // mul is a pipeline the first submit did not compile.
      const squares = mul(y, y);
      evalValues([squares], ex);
      expectClose(
        await ex.readback(squares.buffer, squares.shape),
        PRODUCT.map((v) => v * v),
      );
      expectClose(await ex.readback(y.buffer, y.shape), PRODUCT);
    });
    expect(blocking).not.toHaveBeenCalled();
  });

  it('replays a frame captured in a warm-up once its pipelines compiled', async () => {
    const { device, ex, a, b, blocking } = await freshDevice();
    const { targets: y, replay } = withAsyncCompile(device, () =>
      ex.captureFrame(() => {
        const y = matmul(a, b);
        evalValues([y], ex);
        return y;
      }),
    );
    // Outside the warm-up, and the capture's pipeline is still compiling.
    const got = ex.readbackOnSubmit(y);
    replay();
    expectClose(await got, PRODUCT);
    expect(blocking).not.toHaveBeenCalled();
  });

  it('encodes a replay once ready, captured while another warm-up compiles its pipeline', async () => {
    const { root, device, ex, a, b, blocking } = await freshDevice();
    const other = gpuExecutor(root);
    const warming = warmUp(device, () => toArray(matmul(a, b), other));
    const cap = ex.captureFrame(() => {
      const y = matmul(a, b);
      evalValues([y], ex);
      return y;
    });
    // Before this, encodeReplay() throws: matmul is still compiling.
    await cap.ready;
    const encoder = device.createCommandEncoder();
    cap.encodeReplay(encoder);
    device.queue.submit([encoder.finish()]);
    expectClose(await ex.readback(cap.targets.buffer, cap.targets.shape), PRODUCT);
    expectClose(await warming, PRODUCT);
    expect(blocking).not.toHaveBeenCalled();
  });

  it('rejects with EXECUTION_FAILED when a pipeline does not compile, then runs again', async () => {
    const { device, ex, a, b, async } = await freshDevice();
    async.mockRejectedValueOnce(new Error('bad shader'));
    await expect(warmUp(device, () => toArray(matmul(a, b), ex))).rejects.toMatchObject({
      name: 'RunntimeError',
      code: 'EXECUTION_FAILED',
      message: 'GPU error during pipeline compile: bad shader',
    });
    expectClose(await toArray(matmul(a, b), ex), PRODUCT);
  });
});
