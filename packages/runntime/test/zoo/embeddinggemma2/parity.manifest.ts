/** Parity manifest of EmbeddingGemma 2: the sentence-transformers model in
 *  f32 as the reference, text from SQuAD, photos and captions from COCO and
 *  speech from LibriSpeech as inputs. */
import { models } from '../../../src/zoo/models.ts';
import {
  COSINE_WITHIN_NOISE,
  WITHIN_NOISE,
  type ParityManifest,
  type Threshold,
} from '../../parity/manifest.ts';

const { F32, F16 } = models.embedding.EMBEDDINGGEMMA_2;

/** Items ranked per asking text in the task check. */
export const TOP_K = 5;

/** Each metric, threshold and margin exists once per modality. */
export const MODALITIES = ['text', 'image', 'audio'] as const;

const DEFAULTS = 'methodology default';
const RESIZE =
  "#122: runntime's bicubic resize uses float weights, the reference torchvision's int16 fixed-point weights";

/** `thresholds` once per modality: `error` as `textError`, `imageError` and
 *  `audioError`. */
const perModality = (thresholds: Readonly<Record<string, Threshold>>) =>
  Object.fromEntries(
    MODALITIES.flatMap((modality) =>
      Object.entries(thresholds).map(([name, threshold]) => [
        `${modality}${name[0]!.toUpperCase()}${name.slice(1)}`,
        threshold,
      ]),
    ),
  );

export const EMBEDDINGGEMMA2_PARITY = {
  family: 'embeddinggemma2',
  reference: {
    repo: 'google/embeddinggemma-2',
    revision: 'f9972ac22e696a105c482202646979e33e4b5a1e',
    dtype: 'f32',
  },
  inputs: {
    source:
      'eight pairs per modality. SQuAD v1.1 validation (Rajpurkar et al., 2016), rajpurkar/squad at 7b6d24c440a36b6815f21b70d25016731768db1f: the first question of 8 articles and its paragraph, for two articles the first five paragraphs joined. COCO val2017 (Lin et al., 2014) from images.cocodataset.org: the first 8 images under CC BY 2.0, by image id, and the first caption of each. LibriSpeech test-clean (Panayotov et al., 2015), openslr/librispeech_asr at 71cacbfb7e2354c4226d01e70d77d5fca3d04ba1: the first clip of each of the first 8 speakers and its transcript',
    license: 'CC BY-SA 4.0, CC BY 2.0, CC BY 4.0',
    count: 48,
  },
  settings: { topK: TOP_K },
  variants: [
    {
      name: 'F32',
      config: F32,
      thresholds: {
        ...perModality({
          // Level A: within three times the reference noise in error.
          cosineDistance: COSINE_WITHIN_NOISE,
          error: WITHIN_NOISE,
          // Level B: the same top-k ranking.
          rankMismatches: { value: 0, source: DEFAULTS },
        }),
        imageCosineDistance: { ...COSINE_WITHIN_NOISE, knownIssue: RESIZE },
        imageError: { ...WITHIN_NOISE, knownIssue: RESIZE },
      },
      // Two items the reference scores closer than this are a near-tie.
      margins: perModality({
        score: { value: 3, timesNoise: true, source: 'three times the score noise in meta.json' },
      }),
    },
    {
      // DEFAULT loads the same files.
      name: 'F16',
      config: F16,
      thresholds: perModality({
        // Level A: cos >= 0.999.
        cosineDistance: { value: 1e-3, source: DEFAULTS },
        // Level B: top-k agrees on >= 99%. With 40 slots per modality that
        // allows no miss.
        topKMissed: { value: 0.01, source: DEFAULTS },
      }),
      // Two items the reference scores closer than f16 resolves are a
      // near-tie.
      margins: perModality({
        score: {
          value: 2 ** -11,
          source: 'f16 unit roundoff: scores are dot products of unit vectors',
        },
      }),
    },
  ],
} as const satisfies ParityManifest;
