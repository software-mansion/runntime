/** Parity manifests of DepthART S and B: the upstream model in f32 as the
 *  reference, photos from COCO as inputs. Each size has its own references
 *  and noise. The reference checkpoints are CC BY-NC 4.0: their training
 *  labels come from Depth Anything V2-L. */
import { models } from '../../../src/zoo/models.ts';
import type { ParityManifest } from '../../parity/manifest.ts';

// cos >= 0.999.
const COSINE = {
  value: 1e-3,
  source: 'methodology default, f16 weights',
} as const;

const SIZES = { S: 's', B: 'b' } as const;

const manifest = <S extends keyof typeof SIZES>(size: S) =>
  ({
    family: `depthart-${SIZES[size]}`,
    reference: {
      repo: 'Fengxue93/DepthART',
      revision: '9e154201cd8fb3453c9def581997aaf1b4da3c22',
      dtype: 'f32',
    },
    inputs: {
      source:
        'COCO val2017 (Lin et al., 2014) from images.cocodataset.org: the first 25 images under CC BY 2.0, by image id',
      license: 'CC BY 2.0',
      count: 25,
    },
    // The checkpoint's size, and the commit of xuefeng-cvr/DepthART on
    // GitHub the reference's code comes from.
    settings: {
      size: SIZES[size],
      code: 'b307c123f19051e2d969960bba93ee3684874b90',
    },
    variants: [
      {
        name: size,
        config: models.depthEstimation.DEPTHART[size],
        thresholds: {
          // Level A: on maps with their mean subtracted, from the
          // reference's own model input.
          cosineDistance: COSINE,
          // Level B: the same, from the photo.
          imageCosineDistance: {
            ...COSINE,
            knownIssue: 'runntime resizes linearly, the reference bicubic',
          },
        },
      },
    ],
  }) as const satisfies ParityManifest;

// DEFAULT loads the S file.
export const DEPTHART_S_PARITY = manifest('S');
export const DEPTHART_B_PARITY = manifest('B');
