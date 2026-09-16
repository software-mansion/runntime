/** Moonshine v2 frontend: raw 16 kHz mono audio, reshaped on the CPU into
 *  non-overlapping 80-sample frames [T, 80] (5 ms, 200 Hz), → encoder frames
 *  [T/4, enc.hidden] (~50 Hz):
 *      per-frame CMVN → asinh(k·x) → linear [80 → d] (no bias) → silu
 *      → causal conv1 k=5 s=2 (d → 2d) → silu → causal conv2 k=5 s=2 (2d → d)
 *  CMVN (mean-center, divide by RMS, eps 1e-6) is exactly a bias-free
 *  LayerNorm whose weight the loader supplies as ones. k is the checkpoint's
 *  learnable asinh compression scale — a [1] Parameter holding exp(log_k),
 *  broadcast-multiplied before the asinh. Both convs left-pad kernel−1 zeros
 *  (causal), so each keeps ceil(T/2) frames; no activation after conv2. */

import { asinh, mul, nn, silu, type Value } from '../../core/index.ts';
import type { MoonshineStreamingConfig } from './config.ts';

const K = 5;
const S = 2;
const PAD = K - 1;

export function frontendFrames(
  samples: number,
  frameLen: number,
): { input: number; conv1: number; conv2: number } {
  if (!Number.isInteger(samples) || samples < frameLen || samples % frameLen !== 0) {
    throw new Error(
      `frontendFrames: samples ${samples} must be a positive multiple of frameLen ${frameLen}`,
    );
  }
  const input = samples / frameLen;
  const conv1 = Math.floor((input - 1) / S) + 1; // causal stride-2: ceil(T/2)
  return { input, conv1, conv2: Math.floor((conv1 - 1) / S) + 1 };
}

export class StreamingFrontend extends nn.Module {
  readonly cmvn: nn.LayerNorm;
  readonly k: nn.Parameter;
  readonly linear: nn.Linear;
  readonly conv1: nn.Conv1d;
  readonly conv2: nn.Conv1d;
  constructor(cfg: MoonshineStreamingConfig) {
    super();
    const d = cfg.enc.hidden;
    this.cmvn = new nn.LayerNorm(cfg.frameLen, 1e-6);
    this.k = new nn.Parameter({ elems: 1, dtype: 'f32', dims: [1] });
    this.linear = new nn.Linear(cfg.frameLen, d, { bias: false });
    this.conv1 = new nn.Conv1d(d, 2 * d, K, { stride: S, padLeft: PAD });
    this.conv2 = new nn.Conv1d(2 * d, d, K, { stride: S, padLeft: PAD });
  }
  forward(x: Value): Value {
    const compressed = asinh(mul(this.cmvn.forward(x), this.k.value));
    const h = silu(this.conv1.forward(silu(this.linear.forward(compressed))));
    return this.conv2.forward(h);
  }
}
