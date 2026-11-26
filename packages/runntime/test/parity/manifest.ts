/** Parity manifests: what each model family is compared with, on which
 *  inputs, and how close every variant in `models` has to be. */
import { models } from '../../src/zoo/models.ts';

/** The fields of a `models` entry that pick the weights and the math. */
export interface ModelConfig {
  readonly modelPath: string;
  readonly visionPath?: string;
  readonly audioPath?: string;
  readonly dtype?: 'f32' | 'f16';
}

export interface Threshold {
  /** The most the metric may be. */
  readonly value: number;
  /** Set when `value` counts in multiples of the metric's reference noise:
   *  `noise` in meta.json, or `noiseF16` with 'f16'. */
  readonly timesNoise?: true | 'f16';
  /** The most a limit in multiples of the noise may be. */
  readonly max?: number;
  /** Where the value comes from. */
  readonly source: string;
  /** Set while the variant is known to exceed this threshold, with the
   *  reason. Its test then passes only as long as it does. */
  readonly knownIssue?: string;
}

/** An f32 output within three times the reference noise. */
export const WITHIN_NOISE = {
  value: 3,
  timesNoise: true,
  source: 'three times the noise in meta.json',
} as const;

/** The same limit for a cosine distance. */
export const COSINE_WITHIN_NOISE = {
  value: 9,
  timesNoise: true,
  source:
    '1 − cos grows with the square of the error: nine times the noise is three times the noise in error',
} as const;

/** The same limit on the f16 noise, for an f16 variant, at most the
 *  methodology default. */
export const COSINE_WITHIN_F16_NOISE = {
  value: 9,
  timesNoise: 'f16',
  max: 1e-3,
  source:
    "nine times the f16 noise in meta.json, three times in error, at most the methodology's 1e-3",
} as const;

export interface ParityVariant<M extends ModelConfig> {
  readonly name: string;
  /** The entry in `models`. */
  readonly config: M;
  /** The gates, by metric name. */
  readonly thresholds: Readonly<Record<string, Threshold>>;
  /** The reference score gaps up to which a ranking difference is a
   *  near-tie and not a miss, by the name of their noise in `meta.json`. */
  readonly margins?: Readonly<Record<string, Threshold>>;
}

export interface ParityManifest<M extends ModelConfig = ModelConfig> {
  /** The family's folder in the references directory. */
  readonly family: string;
  /** The PyTorch model the references come from. */
  readonly reference: {
    readonly repo: string;
    readonly revision: string;
    readonly dtype: 'f32' | 'f16' | 'bf16';
  };
  readonly inputs: { readonly source: string; readonly license: string; readonly count: number };
  /** Values the family's tools/parity script sets too, by their name in
   *  `meta.json`. */
  readonly settings?: Readonly<Record<string, string | number>>;
  readonly variants: readonly ParityVariant<M>[];
}

const isModelConfig = (node: object): node is ModelConfig =>
  'modelPath' in node && typeof node.modelPath === 'string';

/** Every entry in `models` by its path, like `embedding.ALL_MINILM_L6_V2.F32`. */
export function modelEntries(): [name: string, config: ModelConfig][] {
  const entries: [string, ModelConfig][] = [];
  const walk = (node: object, path: string[]) => {
    if (isModelConfig(node)) {
      entries.push([path.join('.'), node]);
      return;
    }
    for (const [key, child] of Object.entries(node)) {
      if (typeof child === 'object' && child !== null) walk(child, [...path, key]);
    }
  };
  walk(models, []);
  return entries;
}

/** Whether a parity run of `variant` covers `entry`: the same files with
 *  the same dtype. An entry with no dtype runs as one of the file's listed
 *  dtypes, picked by the device. */
export function covers(variant: ModelConfig, entry: ModelConfig): boolean {
  return (
    variant.modelPath === entry.modelPath &&
    variant.visionPath === entry.visionPath &&
    variant.audioPath === entry.audioPath &&
    (entry.dtype === undefined || variant.dtype === entry.dtype)
  );
}

/** What a family's `meta.json` holds that its manifest states too, and the
 *  reference noise. */
export interface ParityMeta {
  readonly repo: string;
  readonly revision: string;
  readonly dtype: string;
  readonly inputs?: { readonly license?: string; readonly count?: number };
  readonly settings?: Readonly<Record<string, string | number>>;
  readonly noise: Readonly<Record<string, number>>;
  /** The same for the model run in f16. */
  readonly noiseF16?: Readonly<Record<string, number>>;
}

/** Throws when `meta` and `manifest` disagree on a value both hold, or when
 *  a threshold counts in the f16 noise and `meta` has none: the references
 *  come from other settings than the test expects. */
export function checkMeta(manifest: ParityManifest, meta: ParityMeta): void {
  const { family, reference, inputs, settings = {} } = manifest;
  const fields: [field: string, want: unknown, got: unknown][] = [
    ['repo', reference.repo, meta.repo],
    ['revision', reference.revision, meta.revision],
    ['dtype', reference.dtype, meta.dtype],
    ['inputs.license', inputs.license, meta.inputs?.license],
    ['inputs.count', inputs.count, meta.inputs?.count],
    ...Object.entries(settings).map(
      ([name, want]) =>
        [`settings.${name}`, want, meta.settings?.[name]] as [string, unknown, unknown],
    ),
  ];
  for (const [field, want, got] of fields) {
    if (got !== want) {
      throw new Error(
        `stale ${family} reference: ${field} is ${JSON.stringify(got)} in meta.json, ${JSON.stringify(want)} in the manifest. Run its tools/parity script again.`,
      );
    }
  }
  const usesF16 = manifest.variants.some(({ thresholds, margins = {} }) =>
    [...Object.values(thresholds), ...Object.values(margins)].some((t) => t.timesNoise === 'f16'),
  );
  if (usesF16 && meta.noiseF16 === undefined) {
    throw new Error(
      `stale ${family} reference: no noiseF16 in meta.json, which the manifest uses. Run its tools/parity script again.`,
    );
  }
}

/** The limit of each threshold in `thresholds`, by name: its value, or with
 *  `timesNoise` that many times the metric's noise in the family's
 *  `meta.json` (`noise`, or `noiseF16` with 'f16'), at most `max`. A missing
 *  noise gives NaN. */
export function limits(
  thresholds: Readonly<Record<string, Threshold>>,
  meta: ParityMeta,
): Record<string, number> {
  return Object.fromEntries(
    Object.entries(thresholds).map(([name, { value, timesNoise, max = Infinity }]) => {
      if (timesNoise === undefined) return [name, value];
      const noise = (timesNoise === 'f16' ? meta.noiseF16 : meta.noise)?.[name];
      return [name, Math.min(value * (noise ?? NaN), max)];
    }),
  );
}

/** The names of the thresholds in `thresholds` that `metrics` exceed, with
 *  the reference noise in the family's `meta.json`. */
export function exceeded(
  metrics: Readonly<Record<string, number>>,
  thresholds: Readonly<Record<string, Threshold>>,
  meta: ParityMeta,
): string[] {
  // Written so a NaN or missing metric or noise counts as exceeded.
  return Object.entries(limits(thresholds, meta))
    .filter(([name, limit]) => !(metrics[name]! <= limit))
    .map(([name]) => name);
}
