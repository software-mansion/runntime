/** torchaudio's resample with its defaults: a Hann-windowed sinc, 6 zero
 *  crossings wide, cut off at 0.99 of the lower Nyquist. The codec's encoder
 *  runs it to give HuBERT 16 kHz audio. On the CPU, in f64. */

import { RunntimeError } from '../../core/index.ts';

const ZERO_CROSSINGS = 6;
const ROLLOFF = 0.99;

function gcd(a: number, b: number): number {
  while (b) [a, b] = [b, a % b];
  return a;
}

/** `samples` at `from` Hz to `to` Hz, ceil(length · to / from) samples. Both
 *  rates are positive integers. */
export function resample(samples: Float32Array, from: number, to: number): Float32Array {
  if (!Number.isInteger(from) || !Number.isInteger(to) || from <= 0 || to <= 0) {
    throw new RunntimeError(
      'INVALID_ARGUMENT',
      `resample: rates ${from} and ${to} must be positive integers`,
    );
  }
  if (from === to) return samples;
  const g = gcd(from, to);
  const orig = from / g;
  const next = to / g;
  const base = Math.min(orig, next) * ROLLOFF;
  const width = Math.ceil((ZERO_CROSSINGS * orig) / base);
  // One filter per output phase over 2·width + orig input taps, starting
  // `width` before the phase's first input sample.
  const taps = 2 * width + orig;
  const kernels = new Float64Array(next * taps);
  for (let p = 0; p < next; p++) {
    for (let k = 0; k < taps; k++) {
      const t = Math.min(
        Math.max((-p / next + (k - width) / orig) * base, -ZERO_CROSSINGS),
        ZERO_CROSSINGS,
      );
      const window = Math.cos((t * Math.PI) / ZERO_CROSSINGS / 2) ** 2;
      const x = t * Math.PI;
      kernels[p * taps + k] = (x === 0 ? 1 : Math.sin(x) / x) * window * (base / orig);
    }
  }
  const n = samples.length;
  const out = new Float32Array(Math.ceil((next * n) / orig));
  for (let i = 0; i < out.length; i++) {
    const p = i % next;
    const start = Math.floor(i / next) * orig - width;
    let sum = 0;
    for (let k = 0; k < taps; k++) {
      const j = start + k;
      if (j >= 0 && j < n) sum += samples[j]! * kernels[p * taps + k]!;
    }
    out[i] = sum;
  }
  return out;
}
