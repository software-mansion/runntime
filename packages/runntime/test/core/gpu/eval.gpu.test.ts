/** GPU buffer leak checks for gpuExecutor: intermediates go back to its pool,
 *  a warm graph allocates nothing, and dispose() frees every buffer it made. */
import { describe, expect, it } from 'vitest';
import { evalValues, gpuExecutor, type RunntimeExecutor } from '../../../src/core/gpu/eval.ts';
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
import type { Value } from '../../../src/core/graph/value.ts';
import { buffersSince, gpu, liveBuffers, randomF32, repeated, upload } from '../../gpu.ts';

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
    // flash: qLen > 1, subgroups, headDim % 16 == 0
    build: ({ x, m }) => sdpa(x, m, m, FULL_ATTENTION),
  },
  {
    name: 'rows split attention',
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
