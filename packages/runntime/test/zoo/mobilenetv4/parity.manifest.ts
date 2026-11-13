/** Parity manifest of MobileNetV4-Conv-S: the timm model in f32 as the
 *  reference, photos from COCO as inputs. */
import { models } from '../../../src/zoo/models.ts';
import type { ParityManifest } from '../../parity/manifest.ts';

const DEFAULTS = 'methodology default';

export const MOBILENETV4_PARITY = {
  family: 'mobilenetv4',
  reference: {
    repo: 'timm/mobilenetv4_conv_small.e2400_r224_in1k',
    revision: '331fb803779522b685cf942e15f914fb6741c1eb',
    dtype: 'f32',
  },
  inputs: {
    source:
      'COCO val2017 (Lin et al., 2014) from images.cocodataset.org: the first 25 images under CC BY 2.0, by image id',
    license: 'CC BY 2.0',
    count: 25,
  },
  variants: [
    {
      name: 'SMALL',
      config: models.imageClassification.MOBILENETV4.SMALL,
      thresholds: {
        // Level A: cos >= 0.999, on logits with their mean subtracted.
        cosineDistance: { value: 1e-3, source: DEFAULTS },
        // Level B: top-1 agrees on >= 99%.
        top1Missed: {
          value: 0.01,
          source: DEFAULTS,
          knownIssue: 'runntime resizes linearly, the reference bicubic with antialias',
        },
      },
      // Two classes the reference scores closer than f16 resolves are a
      // near-tie.
      margins: {
        score: {
          value: 2 ** -11,
          source: 'f16 unit roundoff: the weights are f16, scores are probabilities',
        },
      },
    },
  ],
} as const satisfies ParityManifest;
