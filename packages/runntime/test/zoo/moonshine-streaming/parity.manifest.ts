/** Parity manifest of Moonshine Streaming tiny: the transformers model in
 *  f32 as the reference, speech from LibriSpeech as inputs. */
import { models } from '../../../src/zoo/models.ts';
import {
  COSINE_WITHIN_F16_NOISE,
  COSINE_WITHIN_NOISE,
  WITHIN_NOISE,
  type ParityManifest,
} from '../../parity/manifest.ts';

const DEFAULTS = 'methodology default';

export const MOONSHINE_STREAMING_TINY_PARITY = {
  family: 'moonshine-streaming-tiny',
  reference: {
    repo: 'UsefulSensors/moonshine-streaming-tiny',
    revision: 'f8e9dfd8c562c257c151a907b7b7f2fe8ff8511a',
    dtype: 'f32',
  },
  inputs: {
    source:
      'LibriSpeech test-clean (Panayotov et al., 2015), openslr/librispeech_asr at 71cacbfb7e2354c4226d01e70d77d5fca3d04ba1: for each of the first 12 speakers, the first clip of at most 30 s',
    license: 'CC BY 4.0',
    count: 12,
  },
  // runntime's chunk length and token cap.
  settings: { maxSeconds: 30, tokensPerSecond: 6.5 },
  variants: [
    {
      name: 'F32',
      config: models.speechToText.MOONSHINE_STREAMING.TINY.F32,
      thresholds: {
        // Level A: within three times the reference noise in error.
        encoderCosineDistance: COSINE_WITHIN_NOISE,
        encoderError: WITHIN_NOISE,
        logitsCosineDistance: COSINE_WITHIN_NOISE,
        logitsError: WITHIN_NOISE,
        // Level B: identical tokens, teacher forced, and identical text.
        tokenMismatches: { value: 0, source: DEFAULTS },
        textMismatches: { value: 0, source: DEFAULTS },
      },
      // A step where the reference scores two tokens closer than this is
      // a near-tie.
      margins: {
        logitGap: {
          value: 3,
          timesNoise: true,
          source: 'three times the logit gap noise in meta.json',
        },
      },
    },
    {
      // DEFAULT loads the same file.
      name: 'F16',
      config: models.speechToText.MOONSHINE_STREAMING.TINY.F16,
      thresholds: {
        // Level A: within three times the f16 noise in error.
        encoderCosineDistance: COSINE_WITHIN_F16_NOISE,
        logitsCosineDistance: COSINE_WITHIN_F16_NOISE,
        // Level B: WER delta <= 0.5 pp.
        werDelta: { value: 0.005, source: DEFAULTS },
      },
    },
  ],
} as const satisfies ParityManifest;
