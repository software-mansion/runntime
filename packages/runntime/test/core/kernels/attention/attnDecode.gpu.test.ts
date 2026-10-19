import { describe, expect, it } from 'vitest';
import { toArray } from '../../../../src/core/gpu/eval.ts';
import { sdpa } from '../../../../src/core/graph/ops.ts';
import {
  ATTN_DECODE_CHUNK,
  ATTN_DECODE_MAX_CHUNKS,
} from '../../../../src/core/kernels/attention/attnDecode.ts';
import {
  expectClose,
  gpu,
  gpuWithoutSubgroups,
  randomF32,
  requireFeature,
  upload,
} from '../../../gpu.ts';
import { checkSdpa, sdpaRef, type SdpaCase } from './checkSdpa.ts';

const main = await gpu();
const withoutSubgroups = await gpuWithoutSubgroups();
const { root, ex } = main;

/** Decode as a model runs it: one query at the end of the cache. */
const decode = (kvLen: number, opts: Partial<SdpaCase> = {}): SdpaCase => ({
  qLen: 1,
  kvLen,
  qHeads: 4,
  kvHeads: 2,
  headDim: 16,
  windowRight: 0,
  qPosOffset: kvLen - 1,
  ...opts,
});

describe('attnDecode kernels', () => {
  // Routes are picked in gpu/dispatch/attention.ts: qLen = 1 takes decode up
  // to ATTN_DECODE_CHUNK keys and decodeSplit past it. Neither uses
  // subgroups, so both devices run the same kernels.
  const CASES: { name: string; c: SdpaCase }[] = [
    // decode: kvLen <= ATTN_DECODE_CHUNK
    { name: 'one key (decode)', c: decode(1) },
    { name: 'odd key count (decode)', c: decode(7) },
    { name: 'one key per thread (decode)', c: decode(64) },
    { name: 'one key past a thread each (decode)', c: decode(65) },
    { name: 'at the chunk (decode)', c: decode(ATTN_DECODE_CHUNK) },
    { name: 'headDim 1 (decode)', c: decode(40, { headDim: 1 }) },
    // decode: 37 has no tile of 8 or more that divides it, so the last tile is partial
    { name: 'headDim 37 (decode)', c: decode(40, { headDim: 37 }) },
    { name: 'headDim 128, the widest it takes (decode)', c: decode(40, { headDim: 128 }) },
    { name: 'a sliding window (decode)', c: decode(200, { windowLeft: 10 }) },
    { name: 'sinks (decode)', c: decode(200, { sinks: true }) },
    { name: 'every query head on one kv head (decode)', c: decode(50, { kvHeads: 1 }) },
    { name: '72 query heads (decode)', c: decode(50, { qHeads: 72, kvHeads: 8 }) },
    {
      name: 'full attention, not at the end (decode)',
      c: decode(50, { windowRight: Infinity, qPosOffset: 10 }),
    },
    // decode: windowLeft 0 past the last key leaves the query no key, so it comes out 0
    {
      name: 'a query that sees no key, with sinks (decode)',
      c: decode(5, { windowLeft: 0, windowRight: Infinity, qPosOffset: 9, sinks: true }),
    },
    // decodeSplit: kvLen > ATTN_DECODE_CHUNK
    { name: 'one key past the chunk (decodeSplit)', c: decode(ATTN_DECODE_CHUNK + 1) },
    {
      name: 'every chunk full (decodeSplit)',
      c: decode(ATTN_DECODE_CHUNK * ATTN_DECODE_MAX_CHUNKS),
    },
    // decodeSplit: past ATTN_DECODE_MAX_CHUNKS chunks, so each chunk grows
    { name: 'past the chunk cap (decodeSplit)', c: decode(5000) },
    { name: 'headDim 37 (decodeSplit)', c: decode(600, { headDim: 37 }) },
    { name: 'headDim 128 (decodeSplit)', c: decode(600, { headDim: 128 }) },
    // decodeSplit: a window shorter than the chunks leaves the later ones empty
    { name: 'a sliding window (decodeSplit)', c: decode(1000, { windowLeft: 10 }) },
    { name: 'sinks (decodeSplit)', c: decode(1000, { sinks: true }) },
    { name: 'every query head on one kv head (decodeSplit)', c: decode(600, { kvHeads: 1 }) },
    // decodeSplit: the scratch holds partials for 64 heads
    { name: '64 query heads (decodeSplit)', c: decode(600, { qHeads: 64, kvHeads: 8 }) },
    {
      name: 'a query that sees no key, with sinks (decodeSplit)',
      c: decode(300, { windowLeft: 0, windowRight: Infinity, qPosOffset: 400, sinks: true }),
    },
  ];
  it.for(
    (['main', 'noSubgroups'] as const).flatMap((device) =>
      (['f32', 'f16'] as const).flatMap((dtype) => CASES.map((c) => ({ device, dtype, ...c }))),
    ),
  )('$dtype $name on $device', async ({ device, dtype, c }, ctx) => {
    const dev = device === 'main' ? main : withoutSubgroups;
    if (dtype === 'f16') requireFeature(ctx, dev.root, 'shader-f16');
    await checkSdpa(dev, c, dtype);
  });

  // Known bug: the decodeSplit scratch holds partials for 64 heads and its
  // merge runs one workgroup of 64 threads, one per head, so heads past 64
  // are never written. Turn into `it` with the fix.
  // decodeSplit: kvLen > ATTN_DECODE_CHUNK
  it.fails('72 query heads (decodeSplit)', () =>
    checkSdpa(main, decode(600, { qHeads: 72, kvHeads: 8 })),
  );

  // decodeSplit: two calls in one submit share the split scratch, so the
  // second must not overwrite partials the first merge has not read yet.
  it('two chained calls in one submit (decodeSplit)', async () => {
    const shape = { qLen: 1, kvLen: 600, qHeads: 4, kvHeads: 4, headDim: 16 };
    const window = { windowLeft: Infinity, windowRight: 0, qPosOffset: 599 };
    const q = randomF32(64, 1);
    const k = randomF32(600 * 64, 2);
    const v = randomF32(600 * 64, 3);
    const kv = (data: Float32Array) => upload(root, data, [600, 64]);
    const opts = { ...shape, ...window };
    const first = sdpa(upload(root, q, [1, 64]), kv(k), kv(v), opts);
    const got = await toArray(sdpa(first, kv(v), kv(k), opts), ex);
    const firstWant = Float32Array.from(sdpaRef(q, k, v, opts));
    expectClose(got, sdpaRef(firstWant, v, k, opts), { atol: 2e-5, rtol: 1e-4 });
  });

  it('refuses a head wider than 128', () => {
    const q = upload(root, new Float32Array(130), [1, 130]);
    const kv = upload(root, new Float32Array(4 * 130), [4, 130]);
    const opts = {
      qHeads: 1,
      kvHeads: 1,
      headDim: 130,
      windowLeft: Infinity,
      windowRight: 0,
      qPosOffset: 3,
    };
    expect(() => sdpa(q, kv, kv, opts)).toThrow(/headDim 130 exceeds the kernel's max \(128\)/);
  });

  it('refuses segments', () => {
    const q = upload(root, new Float32Array(8), [1, 8]);
    const kv = upload(root, new Float32Array(32), [4, 8]);
    const segments = upload(root, new Float32Array(2), [1, 2]);
    const opts = {
      qHeads: 1,
      kvHeads: 1,
      headDim: 8,
      windowLeft: Infinity,
      windowRight: 0,
      qPosOffset: 3,
    };
    expect(() => sdpa(q, kv, kv, { ...opts, segments })).toThrow(/segments need qLen > 1/);
  });
});
