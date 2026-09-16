/** Moonshine conv stem: raw 16 kHz mono audio [L, 1] → encoder frames
 *  [T, d] (384× downsample, ~41.7 frames/s):
 *      conv1 k=127 s=64 (no bias) → tanh → nn.GroupNorm(1, d)
 *      → conv2 k=7 s=3 (+bias) → gelu → conv3 k=3 s=2 (+bias) → gelu
 *  Valid padding throughout; conv weights arrive tap-major [k·C_in, C_out]
 *  from the loader. Each stage's bias and activation ride the conv
 *  epilogue — one dispatch per conv. */

import { nn, type Value } from '../../core/index.ts';
import type { MoonshineConfig } from './config.ts';

const K1 = 127;
const S1 = 64;
const K2 = 7;
const S2 = 3;
const K3 = 3;
const S3 = 2;

const convOut = (t: number, k: number, s: number) => Math.floor((t - k) / s) + 1;

export const MIN_SAMPLES = 895;

export function stemFrames(samples: number): { conv1: number; conv2: number; conv3: number } {
  if (!Number.isInteger(samples) || samples < MIN_SAMPLES) {
    throw new Error(
      `stemFrames: need ≥ ${MIN_SAMPLES} samples (~56 ms at 16 kHz) for one encoder frame, got ${samples}`,
    );
  }
  const t1 = convOut(samples, K1, S1);
  const t2 = convOut(t1, K2, S2);
  return { conv1: t1, conv2: t2, conv3: convOut(t2, K3, S3) };
}

export class ConvStem extends nn.Module {
  readonly conv1: nn.Conv1d;
  readonly groupnorm: nn.GroupNorm;
  readonly conv2: nn.Conv1d;
  readonly conv3: nn.Conv1d;
  constructor(cfg: MoonshineConfig) {
    super();
    const d = cfg.dModel;
    this.conv1 = new nn.Conv1d(1, d, K1, { stride: S1, bias: false, activation: 'tanh' });
    this.groupnorm = new nn.GroupNorm(1, d);
    this.conv2 = new nn.Conv1d(d, 2 * d, K2, { stride: S2, activation: 'gelu' });
    this.conv3 = new nn.Conv1d(2 * d, d, K3, { stride: S3, activation: 'gelu' });
  }
  forward(x: Value): Value {
    return this.conv3.forward(this.conv2.forward(this.groupnorm.forward(this.conv1.forward(x))));
  }
}
