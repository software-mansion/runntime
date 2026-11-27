/** Every entry in `models` has a parity manifest, with inputs and
 *  thresholds. Runs without a GPU, so CI holds a new model to it. */
import { describe, expect, it } from 'vitest';
import { EMBEDDINGGEMMA2_PARITY } from '../zoo/embeddinggemma2/parity.manifest.ts';
import { MINILM_PARITY } from '../zoo/minilm/parity.manifest.ts';
import { MOBILENETV4_PARITY } from '../zoo/mobilenetv4/parity.manifest.ts';
import { MOONSHINE_STREAMING_TINY_PARITY } from '../zoo/moonshine-streaming/parity.manifest.ts';
import { MOONSHINE_BASE_PARITY, MOONSHINE_TINY_PARITY } from '../zoo/moonshine/parity.manifest.ts';
import {
  REVERB_FP16_PARITY,
  REVERB_FP32_PARITY,
  REVERB_INT4_PARITY,
  REVERB_INT8_PARITY,
} from '../zoo/reverb/parity.manifest.ts';
import { covers, modelEntries, type ParityManifest } from './manifest.ts';

const MANIFESTS: readonly ParityManifest[] = [
  EMBEDDINGGEMMA2_PARITY,
  MINILM_PARITY,
  MOBILENETV4_PARITY,
  MOONSHINE_TINY_PARITY,
  MOONSHINE_BASE_PARITY,
  MOONSHINE_STREAMING_TINY_PARITY,
  REVERB_FP32_PARITY,
  REVERB_FP16_PARITY,
  REVERB_INT8_PARITY,
  REVERB_INT4_PARITY,
];

/** Families in `models` without a manifest, by the start of their entries'
 *  paths. A family leaves the list with its manifest. A new model is never
 *  added here: it lands with a manifest. */
const PENDING = [
  'privacyFilter.PRIVACY_FILTER',
  'objectDetection.YOLO26',
  'instanceSegmentation.YOLO26_SEG',
  'keypointDetection.YOLO26_POSE',
  'depthEstimation.DEPTHART',
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
