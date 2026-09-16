/** YaRN rope table math, plus the privacy filter's rope wiring.
 *
 *  Table construction is model math — each architecture has its own rope
 *  recipe, and HF and MLX likewise keep it in per-model rotary classes — so it
 *  lives here. core owns only the `rope` op and kernel that consume the
 *  tables. */

export interface RopeParams {
  positions: number;
  headDim: number;
  theta: number;
  scalingFactor: number;
  ntkAlpha: number;
  ntkBeta: number;
  initialContextLength: number;
}

export interface RopeTables {
  cos: Float32Array;
  sin: Float32Array;
}

export function computeRopeTables(p: RopeParams): RopeTables {
  const half = p.headDim / 2;
  const freq = new Float64Array(half);
  for (let i = 0; i < half; i++) freq[i] = p.theta ** ((2 * i) / p.headDim);

  let concentration = 1.0;
  const invFreq = new Float64Array(half);
  if (p.scalingFactor > 1.0) {
    concentration = 0.1 * Math.log(p.scalingFactor) + 1.0;
    const low =
      (half * Math.log(p.initialContextLength / (p.ntkBeta * 2 * Math.PI))) / Math.log(p.theta);
    const high =
      (half * Math.log(p.initialContextLength / (p.ntkAlpha * 2 * Math.PI))) / Math.log(p.theta);
    for (let i = 0; i < half; i++) {
      const ramp = Math.min(1, Math.max(0, (i - low) / (high - low)));
      const mask = 1 - ramp;
      invFreq[i] = (1 / (p.scalingFactor * freq[i]!)) * (1 - mask) + (1 / freq[i]!) * mask;
    }
  } else {
    for (let i = 0; i < half; i++) invFreq[i] = 1 / freq[i]!;
  }

  const cos = new Float32Array(p.positions * half);
  const sin = new Float32Array(p.positions * half);
  for (let t = 0; t < p.positions; t++) {
    for (let i = 0; i < half; i++) {
      const angle = t * invFreq[i]!;
      cos[t * half + i] = Math.cos(angle) * concentration;
      sin[t * half + i] = Math.sin(angle) * concentration;
    }
  }
  return { cos, sin };
}

import { rope, type Value } from '../../core/index.ts';

export const QK_SCALE = 0.3535533905932738; // 64^(-1/4), applied to q and k

const HEAD_DIM = 64;
const HALF = HEAD_DIM / 2;

export interface RopeConstants {
  cosE: Float32Array;
  sinE: Float32Array;
  rawCos: Float32Array;
  rawSin: Float32Array;
}

export function ropeConstants(tokens: number): RopeConstants {
  const { cos, sin } = computeRopeTables({
    positions: tokens,
    headDim: HEAD_DIM,
    theta: 150000,
    scalingFactor: 32,
    ntkAlpha: 1,
    ntkBeta: 32,
    initialContextLength: 4096,
  });
  const cosE = new Float32Array(tokens * HEAD_DIM);
  const sinE = new Float32Array(tokens * HEAD_DIM);
  for (let t = 0; t < tokens; t++) {
    for (let p = 0; p < HALF; p++) {
      const c = cos[t * HALF + p]! * QK_SCALE;
      const s = sin[t * HALF + p]! * QK_SCALE;
      cosE[t * HEAD_DIM + 2 * p] = c;
      cosE[t * HEAD_DIM + 2 * p + 1] = c;
      sinE[t * HEAD_DIM + 2 * p] = s;
      sinE[t * HEAD_DIM + 2 * p + 1] = s;
    }
  }
  return { cosE, sinE, rawCos: cos, rawSin: sin };
}

export function applyRope(x: Value, cosE: Value, sinE: Value): Value {
  return rope(x, cosE, sinE, { headDim: HEAD_DIM });
}
