/** Parity manifests of Reverb: one per weight file, each against reverb-tts
 *  in f32 on that file's own weights. The int4 and int8 files are the
 *  release's quantized weights, repacked for runntime. Inputs are sentences
 *  written for the test, spoken in the release's preset voices. */
import { models } from '../../../src/zoo/models.ts';
import type { TextToSpeechModel } from '../../../src/zoo/tasks/audio/textToSpeech.ts';
import {
  type ModelConfig,
  type ParityManifest,
  type ParityVariant,
} from '../../parity/manifest.ts';

type Config = ModelConfig & TextToSpeechModel;

const DEFAULTS = 'methodology default';
const TOP_1 = 'methodology default for top-1 agreement: the text to speech row gives no number';

/** The most a log-probability may be off in f32 math: three times the
 *  largest noise of the four references. Their noise is one model's f32
 *  rounding, and each file's is the error of its worst slot, so every file
 *  takes the largest. */
const withinNoise = (noise: number, reference: string) =>
  ({
    value: 3 * noise,
    source: `three times the ${reference} reference noise, shared by the four files: same math`,
  }) as const;
// For the best code of a slot, and for the code a slot holds.
const CONFIDENCE_ERROR = withinNoise(5.989e-4, 'fp32');
const HELD_CONFIDENCE_ERROR = withinNoise(4.577e-4, 'int8 and int4');

/** A variant in f32 math, held to the reference like any f32 model. */
const exact = (name: string, config: Config) =>
  ({
    name,
    config,
    thresholds: {
      // Level A: each pass on the reference's grid, within the noise, and
      // the same best code in every slot.
      confidenceError: CONFIDENCE_ERROR,
      heldConfidenceError: HELD_CONFIDENCE_ERROR,
      codeMismatches: { value: 0, source: DEFAULTS },
      // Level B: identical codes under the reference's noise. A take of
      // another length than the reference's differs in every code.
      tokenMismatches: { value: 0, source: DEFAULTS },
    },
    // A slot where the reference scores its two best codes closer than this
    // is a near-tie.
    margins: {
      logProbGap: {
        value: 3,
        timesNoise: true,
        source: 'three times the log-probability gap noise in meta.json',
      },
    },
  }) as const satisfies ParityVariant<Config>;

/** A variant in f16 math, with at most `mismatches` of the best codes
 *  differing, one share over every pass of every take. Sampling carries a
 *  difference from pass to pass, so the codes it ends with are reported and
 *  not gated. So is the cosine of a pass: its vector is the best codes'
 *  log-probabilities over the slots, mostly near zero, so one slot sets it,
 *  and the logits of every slot are too large to save as a reference. */
const half = (name: string, config: Config, mismatches: number) =>
  ({
    name,
    config,
    thresholds: { codeMismatches: { value: mismatches, source: TOP_1 } },
  }) as const satisfies ParityVariant<Config>;

const manifest = <V extends string>(variant: V, variants: readonly ParityVariant<Config>[]) =>
  ({
    family: `reverb-${variant}`,
    reference: {
      repo: 'software-mansion/Reverb-TTS',
      // The release tag v1.0.0.
      revision: '72be799c8fb6d56a206d260b425c48bf3bd5943d',
      dtype: 'f32',
    },
    inputs: {
      source:
        "14 sentences written for the test, a short and a long one in each of Reverb's 7 languages, each in a preset voice of the release: voices/ of software-mansion/Reverb-TTS at the same revision",
      license: 'CC BY-NC 4.0',
      count: 14,
    },
    // The release variant, the seed of every take and the revision of the
    // reference's codec, eustlb/higgs-audio-v2-tokenizer.
    settings: { variant, seed: 0, codec: '0e3a9177c2991016b8e5e17d23aa474ee665bef2' },
    variants,
  }) as const satisfies ParityManifest<Config>;

const { F32, F16, INT8, INT4 } = models.textToSpeech.REVERB;

export const REVERB_FP32_PARITY = manifest('fp32', [exact('F32', F32)]);
export const REVERB_FP16_PARITY = manifest('fp16', [half('F16', F16, 0.01)]);
// An int file runs in f16 where the device has it; in f32 math it is held to
// the reference of its own weights.
export const REVERB_INT8_PARITY = manifest('int8', [
  exact('INT8 in f32', { ...INT8, dtype: 'f32' }),
  half('INT8', INT8, 0.03),
]);
// DEFAULT loads the same file.
export const REVERB_INT4_PARITY = manifest('int4', [
  exact('INT4 in f32', { ...INT4, dtype: 'f32' }),
  half('INT4', INT4, 0.03),
]);
