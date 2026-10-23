/** GPU buffer leak checks for gpuExecutor: intermediates go back to its pool,
 *  a warm graph allocates nothing, and dispose() frees every buffer it made. */
import { describe, expect, it } from 'vitest';
import { evalValues, gpuExecutor, type RunntimeExecutor } from '../../../src/core/gpu/eval.ts';
import {
  add,
  gatherRows,
  layerNorm,
  matmul,
  mul,
  softmax,
  tanh,
} from '../../../src/core/graph/ops.ts';
import type { Value } from '../../../src/core/graph/value.ts';
import { buffersSince, gpu, liveBuffers, randomF32, upload } from '../../gpu.ts';

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
  /** A known leak, not fixed yet, that fails the warm check. */
  runLeak?: string;
  /** A known leak, not fixed yet, that fails the dispose check. */
  disposeLeak?: string;
  /** A known leak, not fixed yet, that fails the captured-frame check. */
  captureLeak?: string;
}

/** Why every captured frame fails: the pool frees only buffers it holds, and
 *  a captured target is still out. */
const CAPTURED_TARGET = 'dispose() keeps the captured target, which never went back to the pool';

const GRAPHS: GraphCase[] = [
  {
    name: 'an elementwise chain',
    build: ({ x }) => tanh(add(mul(x, x), 1)),
    captureLeak: CAPTURED_TARGET,
  },
  {
    name: 'layerNorm without a bias',
    build: ({ x, w }) => layerNorm(add(x, 1), w),
    disposeLeak: 'dispose() keeps the dummy bias buffer and the layerNorm uniforms',
    captureLeak: CAPTURED_TARGET,
  },
  {
    name: 'softmax',
    build: ({ x }) => softmax(add(x, 1)),
    disposeLeak: 'dispose() keeps the dummy segments buffer',
    captureLeak: CAPTURED_TARGET,
  },
  {
    name: 'matmul',
    build: ({ x, m }) => tanh(matmul(x, m)),
    captureLeak: CAPTURED_TARGET,
  },
  {
    name: 'gatherRows',
    build: ({ x }) => gatherRows(add(x, 1), [3, 0, 3]),
    runLeak: 'gatherRows uploads its index list on every dispatch and never destroys it',
    disposeLeak: 'the gatherRows index list outlives dispose()',
    captureLeak: CAPTURED_TARGET,
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

describe.for(GRAPHS)('gpuExecutor with $name', ({ build, runLeak, disposeLeak, captureLeak }) => {
  (runLeak ? it.fails : it)('allocates nothing once warm', async () => {
    const values = inputs();
    const ex = gpuExecutor(root);
    try {
      for (let i = 0; i < WARM_RUNS; i++) await run(ex, build(values));
      const before = liveBuffers();
      for (let i = 0; i < STEADY_RUNS; i++) await run(ex, build(values));
      expect(buffersSince(before)).toEqual([]);
    } finally {
      ex.dispose();
    }
  });

  (disposeLeak ? it.fails : it)('frees every buffer it made on dispose', async () => {
    const values = inputs();
    // A throwaway executor first: pipelines and the readback staging buffer
    // belong to the root and outlive every executor.
    const first = gpuExecutor(root);
    await run(first, build(values));
    first.dispose();

    const before = liveBuffers();
    const ex = gpuExecutor(root);
    await run(ex, build(values));
    ex.dispose();
    expect(buffersSince(before)).toEqual([]);
  });

  (captureLeak ? it.fails : it)('frees a captured frame on dispose', async () => {
    const values = inputs();
    const first = gpuExecutor(root);
    await run(first, build(values));
    first.dispose();

    const before = liveBuffers();
    const ex = gpuExecutor(root);
    const { targets, replay } = ex.captureFrame(() => {
      const out = build(values);
      evalValues([out], ex);
      return out;
    });
    const read = ex.readbackOnSubmit(targets);
    replay();
    await read;
    ex.dispose();
    expect(buffersSince(before)).toEqual([]);
  });
});
