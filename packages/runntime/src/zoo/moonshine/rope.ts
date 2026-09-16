/** Moonshine rope table builder. Model math rather than core, which owns only
 *  the op consuming these tables.
 *
 *  HF's convention is already the kernel's interleaved-pair form, so the
 *  projections need no permutation; a GPU equivalence test guards that.
 *
 *  Rotary is partial: dims past rotaryDim get cos=1 and sin=0, where the
 *  rotation is an exact identity. Every entry is scaled by headDim^(−1/4), so
 *  q and k together carry 1/√headDim and the kernel needs no score scale.
 *
 *  Row t depends only on t, so slicing a longer table is the same as rebuilding
 *  it. */

import type { MoonshineConfig } from './config.ts';

export interface MoonshineRopeTables {
  cos: Float32Array;
  sin: Float32Array;
}

export function ropeQkScale(headDim: number): number {
  return headDim ** -0.25;
}

export function buildRopeTables(positions: number, cfg: MoonshineConfig): MoonshineRopeTables {
  const hd = cfg.headDim;
  const half = cfg.rotaryDim / 2;
  const s = ropeQkScale(hd);
  const cos = new Float32Array(positions * hd);
  const sin = new Float32Array(positions * hd);
  for (let t = 0; t < positions; t++) {
    const base = t * hd;
    for (let p = 0; p < half; p++) {
      // inv_freq_p = theta^(−2p/rotaryDim), HF's arange(0, dim, 2)/dim
      // exponent.
      const angle = t * cfg.ropeTheta ** (-(2 * p) / cfg.rotaryDim);
      const c = Math.cos(angle) * s;
      const sn = Math.sin(angle) * s;
      cos[base + 2 * p] = c;
      cos[base + 2 * p + 1] = c;
      sin[base + 2 * p] = sn;
      sin[base + 2 * p + 1] = sn;
    }
    for (let c = cfg.rotaryDim; c < hd; c++) {
      cos[base + c] = s; // identity pass-through, scale still applied
      sin[base + c] = 0;
    }
  }
  return { cos, sin };
}
