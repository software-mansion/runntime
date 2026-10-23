import { describe, expect, it } from 'vitest';
import { BufferPool } from '../../../src/core/gpu/bufferPool.ts';
import type { GpuBufferRef, ValueMeta } from '../../../src/core/graph/value.ts';

/** A pool over stub buffers that count their destroy() calls. */
function stubPool() {
  const made: { elems: number; destroyed: number }[] = [];
  const pool = new BufferPool((shape) => {
    const b = { elems: shape.elems, destroyed: 0 };
    made.push(b);
    return {
      destroy() {
        b.destroyed++;
      },
    } as unknown as GpuBufferRef;
  });
  return { pool, made };
}

const f32 = (elems: number): ValueMeta => ({ elems, dtype: 'f32' });

describe('BufferPool', () => {
  it('reuses a released buffer of the same size and dtype', () => {
    const { pool, made } = stubPool();
    const a = pool.acquire(f32(16));
    pool.release(a, f32(16));
    expect(pool.acquire(f32(16))).toBe(a);
    expect(pool.acquire({ elems: 16, dtype: 'f16' })).not.toBe(a);
    expect(made).toHaveLength(2);
  });

  it('trim destroys the free buffers of sizes not acquired since the last trim', () => {
    const { pool, made } = stubPool();
    // Call 1 uses sizes 16 and 32.
    const a = pool.acquire(f32(16));
    const b = pool.acquire(f32(32));
    pool.release(a, f32(16));
    pool.release(b, f32(32));
    expect(pool.trim()).toBe(false);
    // Call 2 uses 16 again and a new size 64; 32 goes.
    const a2 = pool.acquire(f32(16));
    const c = pool.acquire(f32(64));
    pool.release(a2, f32(16));
    pool.release(c, f32(64));
    expect(pool.trim()).toBe(true);
    expect(made.map((m) => m.destroyed)).toEqual([0, 1, 0]);
    // 16 and 64 stay free for the next call.
    expect(pool.acquire(f32(16))).toBe(a);
    expect(pool.acquire(f32(64))).toBe(c);
    expect(made).toHaveLength(3);
  });

  it('trim keeps a size whose buffers are all in use', () => {
    const { pool, made } = stubPool();
    const a = pool.acquire(f32(16));
    pool.release(a, f32(16));
    pool.trim();
    // Taken again, so 16 counts as used and nothing is free to destroy.
    pool.acquire(f32(16));
    expect(pool.trim()).toBe(false);
    pool.release(a, f32(16));
    expect(pool.acquire(f32(16))).toBe(a);
    expect(made).toHaveLength(1);
    expect(made[0]!.destroyed).toBe(0);
  });

  it('dispose destroys every free buffer', () => {
    const { pool, made } = stubPool();
    const a = pool.acquire(f32(16));
    const b = pool.acquire(f32(32));
    pool.release(a, f32(16));
    pool.release(b, f32(32));
    pool.dispose();
    expect(made.map((m) => m.destroyed)).toEqual([1, 1]);
  });
});
