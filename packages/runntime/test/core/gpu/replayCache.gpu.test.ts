/** GPU buffer leak checks for createReplayCache: replays allocate nothing,
 *  and dispose() frees every bucket's buffers. */
import { describe, expect, it } from 'vitest';
import { createReplayCache } from '../../../src/core/gpu/replayCache.ts';
import { add, softmax } from '../../../src/core/graph/ops.ts';
import { buffersSince, gpu, liveBuffers, randomF32 } from '../../gpu.ts';

const { root } = await gpu();

/** Columns of the one input plane; rows are the bucket length. */
const COLS = 16;
const BUCKETS = [4, 8];
/** Replays per bucket after its recording that must not allocate a buffer. */
const STEADY_RUNS = 20;

function cache() {
  return createReplayCache({
    root,
    buckets: BUCKETS,
    inputs: (T) => ({ x: [T, COLS] as const }),
    captureFill: (T, scratch) => scratch.x.set(randomF32(T * COLS)),
    build: (_, { x }) => softmax(add(x, 1)),
  });
}

const fill = (T: number, scratch: { x: Float32Array }) => scratch.x.set(randomF32(T * COLS, T));

describe('createReplayCache', () => {
  it('allocates nothing once every bucket is recorded', async () => {
    const replays = cache();
    try {
      for (const T of BUCKETS) await replays.run(T, fill);
      const before = liveBuffers();
      for (let i = 0; i < STEADY_RUNS; i++) for (const T of BUCKETS) await replays.run(T, fill);
      expect(buffersSince(before)).toEqual([]);
    } finally {
      replays.dispose();
    }
  });

  it('frees every bucket on dispose', async () => {
    // A throwaway cache first: pipelines and the readback staging buffer
    // belong to the root and outlive every cache.
    const first = cache();
    await first.warmAll();
    first.dispose();

    const before = liveBuffers();
    const replays = cache();
    await replays.warmAll();
    replays.dispose();
    expect(buffersSince(before)).toEqual([]);
  });
});
