/** Every route gpu/dispatch/attention.ts can pick, plus rope. Windows are
 *  (left, right): causal is (Infinity, 0), full is (Infinity, Infinity). */
import { abSuite, type AbCase } from '../ab.ts';
import { mat, randMat } from './inputs.ts';

type Float = 'f32' | 'f16';

interface Attn {
  qLen: number;
  kvLen: number;
  qHeads: number;
  kvHeads: number;
  headDim: number;
  dtype?: Float;
  causal?: boolean;
  window?: number;
  sinks?: boolean;
  noSubgroups?: boolean;
}

function sdpa(name: string, o: Attn): AbCase {
  const { qLen, kvLen, qHeads, kvHeads, headDim, dtype = 'f32' } = o;
  return {
    name,
    needs: dtype === 'f16' ? ['shader-f16'] : [],
    noSubgroups: o.noSubgroups,
    build: (version) => {
      const q = randMat(version, qLen, qHeads * headDim, dtype);
      const k = randMat(version, kvLen, kvHeads * headDim, dtype);
      const v = randMat(version, kvLen, kvHeads * headDim, dtype);
      const sinks = o.sinks ? randMat(version, 1, qHeads) : undefined;
      const windowLeft = o.window ?? Infinity;
      const windowRight = o.causal ? 0 : (o.window ?? Infinity);
      // A causal query block sits at the end of the keys, like a decoder step.
      const qPosOffset = o.causal ? kvLen - qLen : undefined;
      return () =>
        version.lib.sdpa(q, k, v, {
          qHeads,
          kvHeads,
          headDim,
          windowLeft,
          windowRight,
          qPosOffset,
          sinks,
        });
    },
  };
}

/** MiniLM-style packed qkv [T, 3·H·hd], each row attending to its own segment. */
function packed(name: string, t: number, heads: number, headDim: number, dtype: Float): AbCase {
  return {
    name,
    needs: dtype === 'f16' ? ['shader-f16'] : [],
    build: (version) => {
      const qkv = randMat(version, t, 3 * heads * headDim, dtype);
      // Two sentences in one batch: rows attend within their own half.
      const half = t / 2;
      const seg = Float32Array.from({ length: t * 2 }, (_, i) =>
        i >> 1 < half ? (i & 1 ? half : 0) : i & 1 ? t : half,
      );
      const segments = version.upload(seg, mat(t, 2));
      return () =>
        version.lib.sdpaPacked(qkv, {
          qHeads: heads,
          kvHeads: heads,
          headDim,
          windowLeft: Infinity,
          windowRight: Infinity,
          segments,
          maxSegment: half,
        });
    },
  };
}

function rope(name: string, t: number, width: number, headDim: number, dtype: Float): AbCase {
  return {
    name,
    needs: dtype === 'f16' ? ['shader-f16'] : [],
    build: (version) => {
      const x = randMat(version, t, width, dtype);
      const cos = randMat(version, t, headDim);
      const sin = randMat(version, t, headDim);
      return () => version.lib.rope(x, cos, sin, { headDim });
    },
  };
}

abSuite('attention', [
  // decode: one query, kvLen ≤ 256.
  sdpa('decode, Moonshine-tiny self-attn, 8 heads hd 36, kv 120', {
    qLen: 1,
    kvLen: 120,
    qHeads: 8,
    kvHeads: 8,
    headDim: 36,
    causal: true,
  }),
  sdpa('decode f16, hd 64, kv 256', {
    qLen: 1,
    kvLen: 256,
    qHeads: 8,
    kvHeads: 8,
    headDim: 64,
    dtype: 'f16',
    causal: true,
  }),
  sdpa('decode, hd with no divisor tile, hd 37', {
    qLen: 1,
    kvLen: 64,
    qHeads: 4,
    kvHeads: 4,
    headDim: 37,
  }),

  // decodeSplit: one query, kvLen > 256.
  sdpa('decodeSplit, Moonshine-tiny cross-attn, kv 1500', {
    qLen: 1,
    kvLen: 1500,
    qHeads: 8,
    kvHeads: 8,
    headDim: 36,
  }),
  sdpa('decodeSplit f16, GQA 14/2 heads hd 64, kv 2048', {
    qLen: 1,
    kvLen: 2048,
    qHeads: 14,
    kvHeads: 2,
    headDim: 64,
    dtype: 'f16',
  }),

  // flash: qLen > 1, subgroups, hd % 16 = 0.
  sdpa('flash, 8 lanes, privacy-filter 14/2 heads hd 64, window 128, sinks', {
    qLen: 512,
    kvLen: 512,
    qHeads: 14,
    kvHeads: 2,
    headDim: 64,
    window: 128,
    sinks: true,
  }),
  sdpa('flash f16, 8 lanes, streaming encoder hd 64, causal', {
    qLen: 256,
    kvLen: 256,
    qHeads: 8,
    kvHeads: 8,
    headDim: 64,
    dtype: 'f16',
    causal: true,
  }),
  sdpa('flash, 4 lanes, hd 48', {
    qLen: 128,
    kvLen: 128,
    qHeads: 6,
    kvHeads: 6,
    headDim: 48,
  }),
  packed('flash packed + segments, MiniLM bucket 128, hd 32', 128, 12, 32, 'f32'),
  packed('flash packed f16 + segments, MiniLM bucket 512, hd 32', 512, 12, 32, 'f16'),

  // rowsSubgroupSplit: f32, subgroups, hd % 16 ≠ 0.
  sdpa('rowsSubgroupSplit, Moonshine-tiny encoder hd 36', {
    qLen: 200,
    kvLen: 200,
    qHeads: 8,
    kvHeads: 8,
    headDim: 36,
  }),
  sdpa('rowsSubgroupSplit, Moonshine-base encoder hd 52', {
    qLen: 200,
    kvLen: 200,
    qHeads: 8,
    kvHeads: 8,
    headDim: 52,
  }),

  // rows: f16 with hd % 16 ≠ 0, or no subgroups with short keys.
  sdpa('rows f16, Moonshine-tiny encoder hd 36', {
    qLen: 200,
    kvLen: 200,
    qHeads: 8,
    kvHeads: 8,
    headDim: 36,
    dtype: 'f16',
  }),
  sdpa('rows, MiniLM bucket 16 hd 32', {
    qLen: 16,
    kvLen: 16,
    qHeads: 12,
    kvHeads: 12,
    headDim: 32,
    noSubgroups: true,
  }),

  // rowsSplit: f32, no subgroups, more than 96 keys.
  sdpa('rowsSplit, MiniLM bucket 128 hd 32', {
    qLen: 128,
    kvLen: 128,
    qHeads: 12,
    kvHeads: 12,
    headDim: 32,
    noSubgroups: true,
  }),

  // batch: more than 4,194,240 (row, head) pairs and not flash.
  sdpa('batch f16, 65536 × 64 heads hd 8, kv 16', {
    qLen: 65536,
    kvLen: 16,
    qHeads: 64,
    kvHeads: 1,
    headDim: 8,
    dtype: 'f16',
  }),

  rope('rope, privacy-filter q [512,896] hd 64', 512, 896, 64, 'f32'),
  rope('rope f16, Moonshine-tiny [200,288] hd 36', 200, 288, 36, 'f16'),
]);
