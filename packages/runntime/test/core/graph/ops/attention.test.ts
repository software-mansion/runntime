import { describe, expect, it } from 'vitest';
import { rope, sdpa, sdpaPacked } from '../../../../src/core/graph/ops.ts';
import { input, meta } from '../values.ts';

// 4 query heads sharing 2 kv heads (GQA), 8 values per head
const QH = 4;
const KVH = 2;
const HD = 8;
const heads = { qHeads: QH, kvHeads: KVH, headDim: HD };
const causal = { ...heads, windowLeft: Infinity, windowRight: 0 };
const full = { ...heads, windowLeft: Infinity, windowRight: Infinity };

/** q of `qLen` rows, k and v of `kvLen` rows. */
function qkv(qLen: number, kvLen: number) {
  return {
    q: input([qLen, QH * HD]),
    k: input([kvLen, KVH * HD]),
    v: input([kvLen, KVH * HD]),
  };
}

describe('sdpa', () => {
  it('gives one output row per query row', () => {
    const { q, k, v } = qkv(5, 7);
    const c = sdpa(q, k, v, causal);
    expect(c.shape).toEqual(meta([5, QH * HD]));
    expect(c.pending?.op).toBe('attn');
    expect(c.pending?.attrs?.slice(0, 3)).toEqual([QH, KVH, HD]);
  });

  it('carries an unbounded window as a large finite number', () => {
    const { q, k, v } = qkv(5, 7);
    const [, , , left, right] = sdpa(q, k, v, causal).pending!.attrs!;
    expect(Number.isFinite(left)).toBe(true);
    expect(left).toBeGreaterThan(1e9);
    expect(right).toBe(0);
  });

  it.each([
    { name: 'decode (one query row)', qLen: 1, kvLen: 7, opts: { ...causal, qPosOffset: 6 } },
    { name: 'cross-attention, more queries than keys', qLen: 5, kvLen: 3, opts: full },
    {
      name: 'a sliding window',
      qLen: 5,
      kvLen: 5,
      opts: { ...heads, windowLeft: 2, windowRight: 0 },
    },
    { name: 'every query head on one kv head', qLen: 5, kvLen: 5, opts: { ...causal, kvHeads: 1 } },
  ])('accepts $name', ({ qLen, kvLen, opts }) => {
    const q = input([qLen, QH * HD]);
    const kv = input([kvLen, opts.kvHeads * HD]);
    expect(sdpa(q, kv, kv, opts).shape).toEqual(meta([qLen, QH * HD]));
  });

  it('takes sinks and packed-sequence segments as extra inputs', () => {
    const { q, k, v } = qkv(5, 5);
    const sinks = input([QH]);
    const segments = input([5, 2]);
    const c = sdpa(q, k, v, { ...causal, sinks, segments, maxSegment: 3 });
    expect(c.pending?.inputs).toEqual([q, k, v, sinks, segments]);
  });

  it.each([
    {
      name: 'q cols that are not qHeads·headDim',
      q: input([5, 30]),
      error: /q cols 30 != qHeads\*headDim 32/,
    },
    {
      name: 'k cols that are not kvHeads·headDim',
      k: input([7, 8]),
      error: /k cols 8 != kvHeads\*headDim 16/,
    },
    { name: 'v rows that differ from k rows', v: input([6, 16]), error: /v rows 6 != k rows 7/ },
    {
      name: 'v cols that are not kvHeads·headDim',
      v: input([7, 8]),
      error: /v cols 8 != kvHeads\*headDim 16/,
    },
    {
      name: 'mixed dtypes',
      v: input([7, 16], 'f16'),
      error: /must share a dtype, got f32\/f32\/f16/,
    },
  ])('rejects $name', ({ q, k, v, error }) => {
    const base = qkv(5, 7);
    expect(() => sdpa(q ?? base.q, k ?? base.k, v ?? base.v, causal)).toThrow(error);
  });

  it.each([
    {
      name: 'qHeads not divisible by kvHeads',
      opts: { ...causal, kvHeads: 3 },
      error: /qHeads 4 not divisible by kvHeads 3/,
    },
    { name: 'zero kv heads', opts: { ...causal, kvHeads: 0 }, error: /not divisible by kvHeads 0/ },
    {
      name: 'a negative window',
      opts: { ...causal, windowLeft: -1 },
      error: /windowLeft -1 must be a non-negative integer/,
    },
    {
      name: 'a fractional window',
      opts: { ...causal, windowRight: 0.5 },
      error: /windowRight 0.5 must be/,
    },
    {
      name: 'a negative qPosOffset',
      opts: { ...causal, qPosOffset: -1 },
      error: /qPosOffset -1 must be a non-negative integer/,
    },
    {
      name: 'queries past the last key under a causal mask',
      opts: { ...causal, qPosOffset: 3 },
      error: /qPosOffset\+qLen 8 exceeds key rows kvLen 7/,
    },
    {
      name: 'sinks of the wrong length',
      opts: { ...causal, sinks: input([2]) },
      error: /sinks must be f32 with 4 elems/,
    },
    {
      name: 'segments of the wrong size',
      opts: { ...causal, segments: input([5]) },
      error: /segments must be f32 \[qLen, 2\] = 10 elems/,
    },
    {
      name: 'maxSegment without segments',
      opts: { ...causal, maxSegment: 3 },
      error: /maxSegment needs segments/,
    },
    {
      name: 'a fractional maxSegment',
      opts: { ...causal, segments: input([5, 2]), maxSegment: 1.5 },
      error: /maxSegment 1.5 must be a positive integer/,
    },
  ])('rejects $name', ({ opts, error }) => {
    const { q, k, v } = qkv(5, 7);
    expect(() => sdpa(q, k, v, opts)).toThrow(error);
  });

  it('rejects segments on decode', () => {
    const { q, k, v } = qkv(1, 7);
    expect(() => sdpa(q, k, v, { ...causal, segments: input([1, 2]) })).toThrow(
      /segments need qLen > 1/,
    );
  });

  it('rejects a head size over the kernel limit of 128', () => {
    const x = input([2, 160]);
    expect(() => sdpa(x, x, x, { ...full, qHeads: 1, kvHeads: 1, headDim: 160 })).toThrow(
      /headDim 160 exceeds the kernel's max \(128\)/,
    );
  });
});

describe('sdpaPacked', () => {
  // q, k and v side by side: (4 + 2·2) heads of 8
  const packed = (t: number) => input([t, (QH + 2 * KVH) * HD]);

  it('reads q, k and v from one tensor and gives the q-shaped output', () => {
    const c = sdpaPacked(packed(5), causal);
    expect(c.shape).toEqual(meta([5, QH * HD]));
    expect(c.pending?.attrs?.at(-1)).toBe(1); // the packed flag
  });

  it.each([
    {
      name: 'cols that are not (qHeads + 2·kvHeads)·headDim',
      x: input([5, 60]),
      opts: causal,
      error: /qkv cols 60 != \(qHeads \+ 2\*kvHeads\)\*headDim = 64/,
    },
    { name: 'a single row', x: packed(1), opts: causal, error: /qLen must be > 1/ },
    {
      name: 'an offset under a causal mask',
      x: packed(5),
      opts: { ...causal, qPosOffset: 1 },
      error: /exceeds key rows/,
    },
  ])('rejects $name', ({ x, opts, error }) => {
    expect(() => sdpaPacked(x, opts)).toThrow(error);
  });

  it('allows an offset when the right side is unbounded', () => {
    expect(sdpaPacked(packed(5), { ...full, qPosOffset: 1 }).shape).toEqual(meta([5, QH * HD]));
  });
});

describe('rope', () => {
  const tables = (t: number, headDim = HD) => ({
    cos: input([t, headDim]),
    sin: input([t, headDim]),
  });

  it('rotates every head of x', () => {
    const { cos, sin } = tables(5);
    const c = rope(input([5, 32], 'f16'), cos, sin, { headDim: HD });
    expect(c.shape).toEqual(meta([5, 32], 'f16'));
    expect(c.pending?.attrs).toEqual([HD, 0, 32]); // headDim, srcStart, width
  });

  it('rotates one section of a wider slab, like the k part of a fused qkv', () => {
    const { cos, sin } = tables(5);
    const c = rope(input([5, 64]), cos, sin, { headDim: HD, srcStart: 32, width: 16 });
    expect(c.shape).toEqual(meta([5, 16]));
    expect(c.pending?.attrs).toEqual([HD, 32, 16]);
  });

  it.each([
    {
      name: 'an odd headDim',
      opts: { headDim: 7 },
      error: /headDim 7 must be a positive even integer/,
    },
    {
      name: 'an odd srcStart',
      opts: { headDim: HD, srcStart: 3, width: 8 },
      error: /srcStart 3 must be a non-negative even integer/,
    },
    {
      name: 'a section past the end',
      opts: { headDim: HD, srcStart: 24, width: 16 },
      error: /window \[24, 40\) exceeds 32 input cols/,
    },
    {
      name: 'a width that is not whole heads',
      opts: { headDim: HD, width: 12 },
      error: /width 12 must be a multiple of headDim 8/,
    },
  ])('rejects $name', ({ opts, error }) => {
    const { cos, sin } = tables(5);
    expect(() => rope(input([5, 32]), cos, sin, opts)).toThrow(error);
  });

  it.each([
    {
      name: 'tables for the wrong length',
      cos: input([4, HD]),
      error: /cos must be f32 \[T,headDim\] = 40 elems, got 32/,
    },
    { name: 'f16 tables', cos: input([5, HD], 'f16'), error: /cos must be f32/ },
  ])('rejects $name', ({ cos, error }) => {
    expect(() => rope(input([5, 32]), cos, input([5, HD]), { headDim: HD })).toThrow(error);
  });
});
