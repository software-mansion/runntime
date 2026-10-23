/** EmbeddingGemma 2 audio encoding: 16 kHz samples in, their token rows for
 *  the text model out, one eager forward of the audio tower per clip. */

import { RunntimeError, tensor, toArray, type RunntimeExecutor } from '../../core/index.ts';
import type { EmbeddingGemma2AudioInputs, EmbeddingGemma2AudioModel } from './audioModel.ts';
import { SPEECH_SAMPLE_RATE } from '../tasks/audio/audioInput.ts';

/** 20 ms frames every 10 ms, each zero-padded to the FFT size. */
const FRAME = 320;
const HOP = 160;
const FFT = 512;
const MEL_FLOOR = 1e-3;
const MAX_FREQUENCY = 8000;
/** Only the first 30 seconds of a clip count. */
const MAX_SAMPLES = 30 * SPEECH_SAMPLE_RATE;
/** Fewest samples that make one frame, about 10 ms. */
const MIN_SAMPLES = FRAME / 2 + 1;

/** Mel frames of a clip of at least MIN_SAMPLES samples. The first frame is
 *  centered on sample 0. */
const frameCount = (samples: number) =>
  Math.floor((Math.min(samples, MAX_SAMPLES) - MIN_SAMPLES) / HOP) + 1;

/** Tokens a clip of `samples` samples takes: each conv halves the frames,
 *  rounding up. Throws INVALID_ARGUMENT for a clip too short for one frame. */
export function audioTokens(samples: number): number {
  if (samples < MIN_SAMPLES) {
    throw new RunntimeError(
      'INVALID_ARGUMENT',
      `audio: a clip needs at least ${MIN_SAMPLES} samples, got ${samples}`,
    );
  }
  return Math.ceil(Math.ceil(frameCount(samples) / 2) / 2);
}

interface MelBank {
  window: Float32Array;
  /** Per mel bin: its first FFT bin and weights. Each triangle is a few bins
   *  wide. */
  filters: { start: number; weights: Float64Array }[];
  /** Where each input point lands in the FFT's bit-reversed order. */
  order: Uint16Array;
  cos: Float64Array;
  sin: Float64Array;
}

let bank: MelBank | undefined;

/** Periodic Hann window, HTK mel triangles from 0 Hz to 8 kHz with no area
 *  normalization, and the FFT's tables. */
function melBank(mels: number): MelBank {
  if (bank?.filters.length === mels) return bank;
  const window = new Float32Array(FRAME);
  for (let n = 0; n < FRAME; n++) window[n] = 0.5 - 0.5 * Math.cos((2 * Math.PI * n) / FRAME);
  const bins = FFT / 2 + 1;
  const mel = (f: number) => 2595 * Math.log10(1 + f / 700);
  const hz = (m: number) => 700 * (10 ** (m / 2595) - 1);
  const top = mel(MAX_FREQUENCY);
  const edge = (i: number) => hz((top * i) / (mels + 1));
  const filters = Array.from({ length: mels }, (_, m) => {
    const [lo, mid, hi] = [edge(m), edge(m + 1), edge(m + 2)];
    const w = new Float64Array(bins);
    let start = bins;
    let end = 0;
    for (let k = 0; k < bins; k++) {
      const f = (k * SPEECH_SAMPLE_RATE) / 2 / (bins - 1);
      const v = Math.max(0, Math.min((f - lo) / (mid - lo), (hi - f) / (hi - mid)));
      if (v > 0) {
        w[k] = v;
        start = Math.min(start, k);
        end = k + 1;
      }
    }
    return { start, weights: w.slice(start, Math.max(start, end)) };
  });
  const order = new Uint16Array(FFT);
  for (let i = 1, j = 0; i < FFT; i++) {
    let bit = FFT >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    order[i] = j;
  }
  const cos = new Float64Array(FFT / 2);
  const sin = new Float64Array(FFT / 2);
  for (let k = 0; k < FFT / 2; k++) {
    cos[k] = Math.cos((2 * Math.PI * k) / FFT);
    sin[k] = -Math.sin((2 * Math.PI * k) / FFT);
  }
  bank = { window, filters, order, cos, sin };
  return bank;
}

/** In-place radix-2 FFT of FFT points already in bit-reversed order. */
function fft(re: Float64Array, im: Float64Array, cos: Float64Array, sin: Float64Array): void {
  for (let size = 2; size <= FFT; size <<= 1) {
    const half = size >> 1;
    const step = FFT / size;
    for (let s = 0; s < FFT; s += size) {
      for (let k = 0; k < half; k++) {
        const c = cos[k * step]!;
        const n = sin[k * step]!;
        const a = s + k;
        const b = a + half;
        const tr = re[b]! * c - im[b]! * n;
        const ti = re[b]! * n + im[b]! * c;
        re[b] = re[a]! - tr;
        im[b] = im[a]! - ti;
        re[a] = re[a]! + tr;
        im[a] = im[a]! + ti;
      }
    }
  }
}

/** The first `frames` log-mel frames of `samples`, [frames, mels]. */
export function logMel(samples: Float32Array, frames: number, mels: number): Float32Array {
  const { window, filters, order, cos, sin } = melBank(mels);
  const out = new Float32Array(frames * mels);
  const re = new Float64Array(FFT);
  const im = new Float64Array(FFT);
  const mag = new Float64Array(FFT / 2 + 1);
  for (let f = 0; f < frames; f++) {
    re.fill(0);
    im.fill(0);
    // Frame f starts FRAME / 2 before sample f · HOP; earlier samples are 0.
    const first = f * HOP - FRAME / 2;
    // Windowed samples are f32, the FFT runs in f64.
    for (let i = Math.max(0, -first); i < FRAME; i++) {
      re[order[i]!] = Math.fround(samples[first + i]! * window[i]!);
    }
    fft(re, im, cos, sin);
    for (let k = 0; k < mag.length; k++) mag[k] = Math.sqrt(re[k]! * re[k]! + im[k]! * im[k]!);
    filters.forEach(({ start, weights }, m) => {
      let s = 0;
      for (let k = 0; k < weights.length; k++) s += mag[start + k]! * weights[k]!;
      out[f * mels + m] = Math.log(s + MEL_FLOOR);
    });
  }
  return out;
}

/** The first conv's taps [P + 1, 9]: a 3×3 window at stride 2, padded by 1,
 *  over the [frames, mels] image, then a row of zeros. */
export function convPatches(mel: Float32Array, frames: number, mels: number): Float32Array {
  const rows = Math.ceil(frames / 2);
  const cols = mels / 2;
  const out = new Float32Array((rows * cols + 1) * 9);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const o = (r * cols + c) * 9;
      for (let kh = 0; kh < 3; kh++) {
        const y = 2 * r - 1 + kh;
        if (y < 0 || y >= frames) continue;
        for (let kw = 0; kw < 3; kw++) {
          const x = 2 * c - 1 + kw;
          if (x >= 0 && x < mels) out[o + kh * 3 + kw] = mel[y * mels + x]!;
        }
      }
    }
  }
  return out;
}

/** The second conv's taps: for each of its output pixels, the 3×3 first-conv
 *  pixels it reads at stride 2, the zero row where it reads padding. */
export function convTaps(rows: number, cols: number): Float32Array {
  const outRows = Math.ceil(rows / 2);
  const outCols = cols / 2;
  const zero = rows * cols;
  const out = new Float32Array(outRows * outCols * 9);
  let o = 0;
  for (let r = 0; r < outRows; r++) {
    for (let c = 0; c < outCols; c++) {
      for (let kh = 0; kh < 3; kh++) {
        for (let kw = 0; kw < 3; kw++) {
          const y = 2 * r - 1 + kh;
          const x = 2 * c - 1 + kw;
          out[o++] = y >= 0 && y < rows && x >= 0 && x < cols ? y * cols + x : zero;
        }
      }
    }
  }
  return out;
}

/** A clip, mono samples at 16 kHz in [-1, 1], to its token rows
 *  [tokens · text hidden]. Only the first 30 s count. Runs on `ex` and trims
 *  its pool after. Throws INVALID_ARGUMENT for a clip shorter than one
 *  frame. */
export async function encodeAudio(
  model: EmbeddingGemma2AudioModel,
  samples: Float32Array,
  ex: RunntimeExecutor,
): Promise<Float32Array> {
  const { mels } = model.cfg;
  const tokens = audioTokens(samples.length);
  const frames = frameCount(samples.length);
  const act = model.input_proj.weight.shape.dtype === 'f16' ? 'f16' : 'f32';
  const patches = convPatches(logMel(samples, frames, mels), frames, mels);
  const taps = convTaps(Math.ceil(frames / 2), mels / 2);
  const inputs: EmbeddingGemma2AudioInputs = {
    patches: tensor(patches, { elems: patches.length, dtype: act, dims: [patches.length / 9, 9] }),
    taps: tensor(taps, { elems: taps.length, dtype: 'f32', dims: [taps.length] }),
    tokens,
  };
  try {
    return await toArray(model.forward(inputs), ex);
  } finally {
    inputs.patches.buffer.destroy();
    inputs.taps.buffer.destroy();
    ex.trim();
  }
}
