import { describe, expect, it } from 'vitest';
import {
  audioTokens,
  convPatches,
  convTaps,
  logMel,
} from '../../../src/zoo/embeddinggemma2/audioEncoder.ts';
import { expectClose } from '../../gpu.ts';

const MELS = 128;

/** A fixed clip in [-1, 1]. The Hugging Face values below come from the same
 *  formula. */
function clip(n: number): Float32Array {
  return Float32Array.from({ length: n }, (_, i) => (((i * 7919) % 2001) - 1000) / 1000);
}

describe('audioTokens', () => {
  // From Hugging Face transformers 5.20.2, Gemma4Processor._compute_audio_num_tokens.
  // Past 30 s (480000 samples) the count stays at 750.
  it.each([
    [161, 1],
    [320, 1],
    [321, 1],
    [480, 1],
    [481, 1],
    [640, 1],
    [641, 1],
    [1600, 3],
    [16000, 25],
    [160000, 250],
    [480000, 750],
    [480001, 750],
    [800000, 750],
  ])('%i samples take %i tokens like Hugging Face', (samples, want) => {
    expect(audioTokens(samples)).toBe(want);
  });

  // Hugging Face gives 0 tokens for a clip shorter than one frame. Here that
  // is an error, since a clip with no tokens has nothing to embed.
  it.each([0, 1, 160])('throws INVALID_ARGUMENT for %i samples', (samples) => {
    expect(() => audioTokens(samples)).toThrow(
      expect.objectContaining({ code: 'INVALID_ARGUMENT' }),
    );
  });
});

describe('logMel', () => {
  // Hugging Face transformers 5.20.2, Gemma4AudioFeatureExtractor defaults
  // (128 HTK mels to 8 kHz, 20 ms frames every 10 ms, FFT 512, log(x + 1e-3)),
  // `_extract_spectrogram` on clip(161): one frame centered on sample 0.
  const ONE_FRAME = [
    -6.90776, 0.443268, -0.741177, 0.416395, -0.0913078, 0.453116, 0.17818, 0.556772, 0.186426,
    0.73417, -0.0566456, 0.976754, -0.325494, 1.09566, 0.197807, 0.731283, 0.755873, 0.101103,
    0.726927, 0.155474, 0.0665132, 1.06764, 0.842721, 0.883131, 1.39741, 1.41949, 1.33345, 1.43752,
    1.81473, 2.02144, 2.29696, 2.59565, 2.84625, 3.05395, 3.20296, 2.93176, 2.69603, 2.32192,
    1.82945, 1.3919, 1.14761, 0.910669, 0.695603, 0.753413, 0.877547, 0.831726, 0.569979, 0.153825,
    0.0953239, 0.399273, 0.699653, 1.22831, 1.9003, 2.56732, 2.76336, 2.69065, 2.46757, 1.90993,
    1.28306, 1.20696, 0.9323, 1.25082, 1.29012, 1.0057, 0.188754, -0.282205, 0.358574, 1.56364,
    2.40726, 2.63917, 2.4782, 1.89898, 1.34967, 1.19974, 1.46838, 1.53582, 0.959643, -0.429999,
    0.124457, 1.77916, 2.52719, 2.50953, 1.89516, 1.41173, 1.50877, 1.77727, 1.22057, -0.351928,
    0.81063, 2.26545, 2.5698, 1.93392, 1.5174, 1.85222, 1.74913, 0.48956, 0.674441, 2.31178,
    2.46365, 1.79349, 1.81703, 2.02021, 0.953244, 0.845066, 2.41734, 2.32233, 1.80275, 2.15668,
    1.55959, 0.801085, 2.37657, 2.30914, 1.98386, 2.23055, 1.08984, 1.88944, 2.50913, 2.06066,
    2.34509, 1.299, 2.08273, 2.45152, 2.27213, 2.23454, 1.39744, 2.54077, 2.28973, 2.4814,
  ];

  // The same on clip(800), 4 frames, every 8th mel bin of each.
  const FOUR_FRAMES_EVERY_8TH = [
    [
      -6.90776, 0.186426, 0.755873, 1.39741, 2.84625, 1.14761, 0.0953239, 2.46757, 0.188754,
      1.34967, 2.52719, 0.81063, 0.674441, 2.41734, 1.98386, 2.08273,
    ],
    [
      -6.90776, -3.30138, -0.0881657, -3.10561, 3.00654, -1.34624, -1.751, 2.23995, -0.46269,
      -1.32062, 3.07019, 0.254623, 0.262551, 2.80339, 0.716054, 2.27172,
    ],
    [
      -6.90776, -0.857713, 0.419209, 0.18839, 3.06665, 0.108037, 0.487826, 2.07317, 0.885977,
      0.582375, 3.10113, 1.37728, 1.45602, 2.84521, 1.57587, 2.54015,
    ],
    [
      -6.90776, -2.55444, -0.00801087, -2.14573, 3.00899, -1.42247, -1.0249, 2.24252, -0.162103,
      -1.05715, 3.07198, 0.368205, 0.453794, 2.80679, 0.85154, 2.29349,
    ],
  ];

  // The golden values carry 6 significant digits.
  const TOL = { atol: 1e-4, rtol: 1e-5 };

  it('matches Hugging Face on one frame, every mel bin', () => {
    expectClose(logMel(clip(161), 1, MELS), ONE_FRAME, TOL);
  });

  it('matches Hugging Face on four frames', () => {
    const got = logMel(clip(800), 4, MELS);
    const every8th = Array.from({ length: 4 }, (_, f) =>
      Array.from({ length: MELS / 8 }, (_, i) => got[f * MELS + 8 * i]!),
    );
    expectClose(every8th.flat(), FOUR_FRAMES_EVERY_8TH.flat(), TOL);
  });

  it('is log(1e-3) in every bin of silence', () => {
    expectClose(logMel(new Float32Array(1600), 9, MELS), new Array(9 * MELS).fill(Math.log(1e-3)));
  });

  it('reads only the samples of the frames it is asked for', () => {
    // Frame 0 ends at sample 159, so samples from 160 on do not change it.
    const a = clip(800);
    const b = a.slice();
    b.fill(0.5, 160);
    expect(logMel(b, 1, MELS)).toEqual(logMel(a, 1, MELS));
  });
});

describe('convPatches', () => {
  // A 3×3 window at stride 2, padded by 1, worked out by hand on the
  // [3 frames, 4 mels] image
  //   1  2  3  4
  //   5  6  7  8
  //   9 10 11 12
  // which gives 2 × 2 output pixels, then one row of zeros.
  it('takes each output pixel its 3×3 window, zeros in the padding', () => {
    const mel = Float32Array.from({ length: 12 }, (_, i) => i + 1);
    expect(Array.from(convPatches(mel, 3, 4))).toEqual([
      ...[0, 0, 0, 0, 1, 2, 0, 5, 6],
      ...[0, 0, 0, 2, 3, 4, 6, 7, 8],
      ...[0, 5, 6, 0, 9, 10, 0, 0, 0],
      ...[6, 7, 8, 10, 11, 12, 0, 0, 0],
      ...[0, 0, 0, 0, 0, 0, 0, 0, 0],
    ]);
  });

  it('gives ceil(frames / 2) rows for an odd frame count', () => {
    // 5 frames of 4 mels: 3 × 2 pixels plus the zero row.
    expect(convPatches(new Float32Array(20), 5, 4)).toHaveLength((3 * 2 + 1) * 9);
  });
});

describe('convTaps', () => {
  // The second conv over a [3, 4] first-conv map, by hand: indices y·4 + x of
  // the pixels each 3×3 window reads at stride 2, 12 (the zero row) where it
  // reads padding.
  it('points each tap at its first-conv pixel or the zero row', () => {
    expect(Array.from(convTaps(3, 4))).toEqual([
      ...[12, 12, 12, 12, 0, 1, 12, 4, 5],
      ...[12, 12, 12, 1, 2, 3, 5, 6, 7],
      ...[12, 4, 5, 12, 8, 9, 12, 12, 12],
      ...[5, 6, 7, 9, 10, 11, 12, 12, 12],
    ]);
  });

  it('gives ceil(rows / 2) output rows for an odd row count', () => {
    // 5 rows of 4: 3 × 2 output pixels of 9 taps.
    expect(convTaps(5, 4)).toHaveLength(3 * 2 * 9);
  });
});
