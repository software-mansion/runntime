/** Shared by the quantW kernel suites: random int weights packed into the
 *  quantW layout, and their dequantized f64 values for the reference. */
import type { TgpuRoot } from 'typegpu';
import { onTestFinished } from 'vitest';
import { uploadU32 } from '../../../../src/core/gpu/buffers.ts';
import { materialized, type Value } from '../../../../src/core/graph/value.ts';
import { randomF32 } from '../../../gpu.ts';

/** A quantized [E·K, N] weight: the GPU value, its scales and what it
 *  dequantizes to. */
export interface QuantWeight {
  w: Value;
  /** f32 [E·K / groupSize, N]: one scale per group of rows in each column. */
  scales: Float32Array;
  /** q · scale for row r and column c at [r·N + c], in f64. */
  dequant: Float64Array;
}

/** Packs signed ints q [E·K, N] the way kernels/quantCommon.ts lays them out,
 *  written from that description: 16-value tiles per (expert, 4-column unit,
 *  4-row chunk), column-major in the tile at `bits` per value, tiles k4-major
 *  per (expert, unit). */
function packQuantW(q: Int8Array, experts: number, k: number, n: number, bits: 4 | 8) {
  const words = new Uint32Array((experts * k * n * bits) / 32);
  const mask = (1 << bits) - 1;
  for (let e = 0; e < experts; e++) {
    for (let u = 0; u < n / 4; u++) {
      for (let k4 = 0; k4 < k / 4; k4++) {
        const tileWord = ((e * (n / 4) + u) * (k / 4) + k4) * (bits / 2);
        for (let c = 0; c < 4; c++) {
          for (let r = 0; r < 4; r++) {
            const v = q[(e * k + k4 * 4 + r) * n + u * 4 + c]!;
            const bit = (c * 4 + r) * bits;
            const at = tileWord + Math.floor(bit / 32);
            words[at] = (words[at]! | ((v & mask) << (bit % 32))) >>> 0;
          }
        }
      }
    }
  }
  return words;
}

/** Random ints over the full signed range of `bits` and positive scales,
 *  packed and uploaded. The buffer is destroyed when the test ends. */
export function randomQuantWeight(
  root: TgpuRoot,
  {
    experts,
    k,
    n,
    bits,
    groupSize,
  }: { experts: number; k: number; n: number; bits: 4 | 8; groupSize: number },
  seed: number,
): QuantWeight {
  const rows = experts * k;
  const half = 2 ** (bits - 1);
  const q = Int8Array.from(randomF32(rows * n, seed), (v) =>
    Math.min(half - 1, Math.floor(v * half)),
  );
  const scales = randomF32((rows / groupSize) * n, seed + 1).map((v) => (Math.abs(v) + 0.1) / half);
  const dequant = new Float64Array(rows * n);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < n; c++) {
      dequant[r * n + c] = q[r * n + c]! * scales[Math.floor(r / groupSize) * n + c]!;
    }
  }
  const words = packQuantW(q, experts, k, n, bits);
  const buffer = uploadU32(root, words);
  onTestFinished(() => buffer.destroy());
  const w = materialized({ elems: words.length, dtype: 'quantW', dims: [rows, n] }, buffer);
  return { w, scales, dequant };
}
