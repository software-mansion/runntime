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
  /** Set when `value` counts in multiples of the metric's reference noise. */
  readonly timesNoise?: true;
  /** Where the value comes from. */
  readonly source: string;
}

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
}

/** Throws when `meta` and `manifest` disagree on a value both hold: the
 *  references come from other settings than the test expects. */
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
}

/** The limit of each threshold in `thresholds`, by name. `noise` is the
 *  reference noise in the family's `meta.json`, by the same names. A missing
 *  noise gives NaN. */
export function limits(
  thresholds: Readonly<Record<string, Threshold>>,
  noise: Readonly<Record<string, number>>,
): Record<string, number> {
  return Object.fromEntries(
    Object.entries(thresholds).map(([name, { value, timesNoise }]) => [
      name,
      value * (timesNoise ? noise[name]! : 1),
    ]),
  );
}

/** The names of the thresholds in `thresholds` that `metrics` exceed.
 *  `noise` is the reference noise in the family's `meta.json`, by metric. */
export function exceeded(
  metrics: Readonly<Record<string, number>>,
  thresholds: Readonly<Record<string, Threshold>>,
  noise: Readonly<Record<string, number>>,
): string[] {
  // Written so a NaN or missing metric or noise counts as exceeded.
  return Object.entries(limits(thresholds, noise))
    .filter(([name, limit]) => !(metrics[name]! <= limit))
    .map(([name]) => name);
}
