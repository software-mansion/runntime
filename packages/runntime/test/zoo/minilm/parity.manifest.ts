/** Parity manifest of all-MiniLM-L6-v2: the sentence-transformers model in
 *  f32 as the reference, questions and paragraphs from SQuAD as inputs. */
import { models } from '../../../src/zoo/models.ts';
import type { ParityManifest } from '../../parity/manifest.ts';

const { F32, F16 } = models.embedding.ALL_MINILM_L6_V2;

/** Passages ranked per question in the task check. */
export const TOP_K = 5;

const DEFAULTS = 'methodology default';
const NOISE = 'three times the noise in meta.json';

export const MINILM_PARITY = {
  family: 'minilm',
  reference: {
    repo: 'sentence-transformers/all-MiniLM-L6-v2',
    revision: '1110a243fdf4706b3f48f1d95db1a4f5529b4d41',
    dtype: 'f32',
  },
  inputs: {
    source:
      'SQuAD v1.1 validation (Rajpurkar et al., 2016), rajpurkar/squad at 7b6d24c440a36b6815f21b70d25016731768db1f: the first question of 25 articles and its paragraph, for three articles the first five paragraphs joined',
    license: 'CC BY-SA 4.0',
    count: 50,
  },
  variants: [
    {
      name: 'F32',
      config: F32,
      thresholds: {
        // Level A: within three times the reference noise.
        cosineDistance: { value: 3, timesNoise: true, source: NOISE },
        error: { value: 3, timesNoise: true, source: NOISE },
        // Level B: the same top-k ranking.
        rankMismatches: { value: 0, source: DEFAULTS },
      },
    },
    {
      name: 'F16',
      config: F16,
      thresholds: {
        // Level A: cos >= 0.999.
        cosineDistance: { value: 1e-3, source: DEFAULTS },
        // Level B: top-k agrees on >= 99%.
        topKMissed: { value: 0.01, source: DEFAULTS },
      },
    },
  ],
} as const satisfies ParityManifest;
