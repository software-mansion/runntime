/** Moonshine rope table builder. Model math rather than core, which owns only
 *  the op consuming these tables.
 *
 *  HF's Moonshine rotates the interleaved pairs (2p, 2p + 1), so entries 2p
 *  and 2p + 1 of a table row hold pair p's angle.
 *
 *  Rotary is partial: dims past rotaryDim get cos=1 and sin=0, where the
 *  rotation is an exact identity.
 *
 *  Row t depends only on t, so slicing a longer table is the same as rebuilding
 *  it. */

import type { MoonshineConfig } from './config.ts';

/** The columns of a head that HF's Moonshine rotates together, for the rope
 *  op. */
export const MOONSHINE_ROPE_LAYOUT = 'interleaved';

export interface MoonshineRopeTables {
  cos: Float32Array;
  sin: Float32Array;
}

export function buildRopeTables(positions: number, cfg: MoonshineConfig): MoonshineRopeTables {
  const hd = cfg.headDim;
  const half = cfg.rotaryDim / 2;
  const cos = new Float32Array(positions * hd);
  const sin = new Float32Array(positions * hd);
  for (let t = 0; t < positions; t++) {
    const base = t * hd;
    for (let p = 0; p < half; p++) {
      // inv_freq_p = theta^(−2p/rotaryDim), HF's arange(0, dim, 2)/dim
      // exponent.
      const angle = t * cfg.ropeTheta ** (-(2 * p) / cfg.rotaryDim);
      const c = Math.cos(angle);
      const sn = Math.sin(angle);
      cos[base + 2 * p] = c;
      cos[base + 2 * p + 1] = c;
      sin[base + 2 * p] = sn;
      sin[base + 2 * p + 1] = sn;
    }
    for (let c = cfg.rotaryDim; c < hd; c++) {
      cos[base + c] = 1; // identity pass-through
      sin[base + c] = 0;
    }
  }
  return { cos, sin };
}
