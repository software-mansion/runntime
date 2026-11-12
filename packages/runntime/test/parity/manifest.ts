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

/** The names of the thresholds in `thresholds` that `metrics` exceed.
 *  `noise` is the reference noise in the family's `meta.json`, by metric. */
export function exceeded(
  metrics: Readonly<Record<string, number>>,
  thresholds: Readonly<Record<string, Threshold>>,
  noise: Readonly<Record<string, number>>,
): string[] {
  // Written so a NaN or missing metric or noise counts as exceeded.
  return Object.entries(thresholds)
    .filter(
      ([name, { value, timesNoise }]) =>
        !(metrics[name]! <= value * (timesNoise ? noise[name]! : 1)),
    )
    .map(([name]) => name);
}
