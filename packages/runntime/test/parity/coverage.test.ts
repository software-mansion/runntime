/** Every entry in `models` has a parity manifest, with inputs and
 *  thresholds. Runs without a GPU, so CI holds a new model to it. */
import { describe, expect, it } from 'vitest';
import { MINILM_PARITY } from '../zoo/minilm/parity.manifest.ts';
import { covers, modelEntries, type ParityManifest } from './manifest.ts';

const MANIFESTS: readonly ParityManifest[] = [MINILM_PARITY];

/** Families in `models` without a manifest, by the start of their entries'
 *  paths. A family leaves the list with its manifest. A new model is never
 *  added here: it lands with a manifest. */
const PENDING = [
  'embedding.EMBEDDINGGEMMA_2',
  'privacyFilter.PRIVACY_FILTER',
  'speechToText.MOONSHINE',
  'speechToText.MOONSHINE_STREAMING',
  'objectDetection.YOLO26',
  'instanceSegmentation.YOLO26_SEG',
  'keypointDetection.YOLO26_POSE',
  'depthEstimation.DEPTHART',
  'imageClassification.MOBILENETV4',
];

const variants = MANIFESTS.flatMap((manifest) => manifest.variants);
const pending = (name: string) => PENDING.some((family) => name.startsWith(`${family}.`));

describe('parity coverage', () => {
  it.each(modelEntries())('%s has a parity variant', (name, entry) => {
    const covered = variants.some((variant) => covers(variant.config, entry));
    // A pending family with a manifest has to leave PENDING.
    expect(covered).toBe(!pending(name));
  });

  it('lists only pending families that are in models', () => {
    const names = modelEntries().map(([name]) => name);
    for (const family of PENDING) {
      expect(names.some((name) => name.startsWith(`${family}.`))).toBe(true);
    }
  });

  it.each(MANIFESTS)(
    '$family has a pinned reference, 10 to 50 inputs and thresholds',
    (manifest) => {
      expect(manifest.reference.revision).toMatch(/^[0-9a-f]{40}$/);
      expect(manifest.inputs.count).toBeGreaterThanOrEqual(10);
      expect(manifest.inputs.count).toBeLessThanOrEqual(50);
      expect(manifest.inputs.license).not.toBe('');
      for (const { thresholds, margins = {} } of manifest.variants) {
        expect(Object.keys(thresholds)).not.toHaveLength(0);
        for (const { source } of Object.values(thresholds)) expect(source).not.toBe('');
        for (const { source } of Object.values(margins)) expect(source).not.toBe('');
      }
    },
  );
});
