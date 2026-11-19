/** Parity manifests of Moonshine tiny and base: the transformers model in
 *  f32 as the reference, speech from LibriSpeech as inputs. Each size has its
 *  own references and noise. */
import { models } from '../../../src/zoo/models.ts';
import { COSINE_WITHIN_NOISE, WITHIN_NOISE, type ParityManifest } from '../../parity/manifest.ts';

const DEFAULTS = 'methodology default';

const manifest = <S extends 'TINY' | 'BASE'>(size: S, revision: string) =>
  ({
    family: `moonshine-${size.toLowerCase() as Lowercase<S>}`,
    reference: {
      repo: `UsefulSensors/moonshine-${size.toLowerCase() as Lowercase<S>}`,
      revision,
      dtype: 'f32',
    },
    inputs: {
      source:
        'LibriSpeech test-clean (Panayotov et al., 2015), openslr/librispeech_asr at 71cacbfb7e2354c4226d01e70d77d5fca3d04ba1: for each of the first 12 speakers, the first clip of at most 26 s',
      license: 'CC BY 4.0',
      count: 12,
    },
    // runntime's chunk length, and the token cap of the model card.
    settings: { maxSeconds: 26, tokensPerSecond: 6.5 },
    variants: [
      {
        name: 'F32',
        config: models.speechToText.MOONSHINE[size].F32,
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
        config: models.speechToText.MOONSHINE[size].F16,
        thresholds: {
          // Level A: cos >= 0.999.
          encoderCosineDistance: { value: 1e-3, source: DEFAULTS },
          logitsCosineDistance: { value: 1e-3, source: DEFAULTS },
          // Level B: WER delta <= 0.5 pp.
          werDelta: { value: 0.005, source: DEFAULTS },
        },
      },
    ],
  }) as const satisfies ParityManifest;

export const MOONSHINE_TINY_PARITY = manifest('TINY', '390624ed33d594443aa4aa221f5b9f283b545b5a');
export const MOONSHINE_BASE_PARITY = manifest('BASE', '7a73d8d55ac0ba2ef3ae761593f6784b51f96dcf');
