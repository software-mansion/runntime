/** Parity manifest of the Privacy Filter. The hosted file holds the
 *  experts' weights in int8, so there are two references, both opf, the
 *  upstream code, in f32: on the hosted file's own weights for the logits,
 *  and on the upstream weights for the spans. Inputs are texts written for
 *  the test. */
import { models } from '../../../src/zoo/models.ts';
import { COSINE_WITHIN_NOISE, WITHIN_NOISE, type ParityManifest } from '../../parity/manifest.ts';

export const PRIVACY_FILTER_PARITY = {
  family: 'privacy-filter',
  reference: {
    repo: 'openai/privacy-filter',
    revision: '7ffa9a043d54d1be65afb281eddf0ffbe629385b',
    dtype: 'f32',
  },
  inputs: {
    source:
      '28 texts written for the test, with made-up people, addresses, numbers and secrets: every label several times, 7 languages, one text with nothing to find and one longer than the attention band',
    license: 'MIT',
    count: 28,
  },
  // The commit of opf, openai/privacy-filter on GitHub, and the revision of
  // the hosted file, software-mansion/runntime-privacy-filter.
  settings: {
    opf: 'f7f00ca7fb869683eb732c010299d901457f19c3',
    hosted: '74a8a921d1841f3e8890d640cb5e5ee73775185f',
  },
  variants: [
    {
      name: 'INT8',
      config: models.privacyFilter.PRIVACY_FILTER.DEFAULT,
      thresholds: {
        // Level A, against opf on the hosted file's weights: the logits of
        // every token within the noise, and the same best label on >= 99%
        // of the tokens.
        error: WITHIN_NOISE,
        cosineDistance: COSINE_WITHIN_NOISE,
        labelMismatches: { value: 0.01, source: 'methodology default for top-1 agreement' },
        // Level B, against opf on the upstream weights: span F1 >= 0.98,
        // as 1 − F1.
        spanF1Distance: { value: 0.02, source: 'methodology default for int8' },
      },
    },
  ],
} as const satisfies ParityManifest;
