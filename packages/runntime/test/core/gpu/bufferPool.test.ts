import { describe, expect, it, vi } from 'vitest';
import { BufferPool } from '../../../src/core/gpu/bufferPool.ts';
import type { GpuBufferRef, ValueMeta } from '../../../src/core/graph/value.ts';

const F32_16: ValueMeta = { elems: 16, dtype: 'f32' };
const F16_16: ValueMeta = { elems: 16, dtype: 'f16' };

/** A pool over stub buffers that record destroy() calls. */
function pool() {
  const made: { destroy: ReturnType<typeof vi.fn> }[] = [];
  const p = new BufferPool(() => {
    const buf = { destroy: vi.fn() };
    made.push(buf);
    return buf as unknown as GpuBufferRef;
  });
  return { p, made };
}

describe('BufferPool', () => {
  it('hands a released buffer to the next acquire of the same size and dtype', () => {
    const { p, made } = pool();
    const a = p.acquire(F32_16);
    p.release(a, F32_16);
    expect(p.acquire(F32_16)).toBe(a);
    expect(made).toHaveLength(1);
  });

  it('makes a new buffer for another dtype of the same size', () => {
    const { p, made } = pool();
    p.release(p.acquire(F32_16), F32_16);
    p.acquire(F16_16);
    expect(made).toHaveLength(2);
  });

  it('destroys its free buffers on dispose', () => {
    const { p, made } = pool();
    p.release(p.acquire(F32_16), F32_16);
    p.dispose();
    expect(made[0]!.destroy).toHaveBeenCalledOnce();
  });

  it('destroys the buffers still out on dispose', () => {
    const { p, made } = pool();
    p.acquire(F32_16);
    p.dispose();
    expect(made[0]!.destroy).toHaveBeenCalledOnce();
  });
});
