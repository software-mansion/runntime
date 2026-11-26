/** What every `*.parity.ts` shares in the page: reading the saved
 *  references, breaking a weights file on purpose, and the run of every
 *  variant against its thresholds, with the report. */
import { afterAll, beforeAll, describe, expect, inject, it } from 'vitest';
import { commands } from 'vitest/browser';
import {
  bufferSource,
  parseSafetensorsHeader,
  type RangeSource,
} from '../../src/core/weights/safetensors.ts';
import { imageBufferFromImageData, type ImageBuffer } from '../../src/zoo/index.ts';
import type { ModelPath } from '../../src/zoo/load.ts';
import {
  checkMeta,
  exceeded,
  limits,
  type ParityManifest,
  type ParityMeta,
  type Threshold,
} from './manifest.ts';
import type { Summary } from './metrics.ts';

async function fetchBytes(url: string): Promise<Uint8Array<ArrayBuffer>> {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  return new Uint8Array(await res.arrayBuffer());
}

/** A file in the references directory, by its path under it. */
async function referenceBytes(path: string): Promise<Uint8Array<ArrayBuffer>> {
  try {
    return await fetchBytes(`/@fs${inject('parityDir')}/${path}`);
  } catch (err) {
    throw new Error(
      `no reference ${path} in ${inject('parityDir')}: run its tools/parity script, or set RUNNTIME_PARITY_DIR`,
      { cause: err },
    );
  }
}

/** A JSON reference file. `T` is the shape its tools/parity script writes. */
export async function readJson<T>(path: string): Promise<T> {
  return JSON.parse(new TextDecoder().decode(await referenceBytes(path))) as T;
}

/** The F32 and I32 tensors of a safetensors reference file, by name. */
export async function readTensors(path: string): Promise<Map<string, Float32Array | Int32Array>> {
  const bytes = await referenceBytes(path);
  const { tensors } = await parseSafetensorsHeader(bufferSource(bytes));
  const out = new Map<string, Float32Array | Int32Array>();
  for (const [name, { dtype, begin, end }] of tensors) {
    // A copy, so the view starts at an aligned offset.
    const data = bytes.slice(begin, end).buffer;
    if (dtype === 'F32') out.set(name, new Float32Array(data));
    else if (dtype === 'I32') out.set(name, new Int32Array(data));
    else throw new Error(`${path}: tensor ${name} is ${dtype}, expected F32 or I32`);
  }
  return out;
}

/** The `count` rows of a `[count, ...]` reference tensor. */
export function rows(tensor: Float32Array | Int32Array, count: number): Float32Array[] {
  const size = tensor.length / count;
  return Array.from({ length: count }, (_, i) =>
    Float32Array.from(tensor.subarray(i * size, (i + 1) * size)),
  );
}

/** A PNG in the references directory as RGBA bytes. */
export async function readImage(path: string): Promise<ImageBuffer> {
  const bitmap = await createImageBitmap(new Blob([await referenceBytes(path)]), {
    colorSpaceConversion: 'none',
  });
  const { width, height } = bitmap;
  const context = new OffscreenCanvas(width, height).getContext('2d')!;
  context.drawImage(bitmap, 0, 0);
  bitmap.close();
  return imageBufferFromImageData(context.getImageData(0, 0, width, height));
}

/** The safetensors file at `url` with the F32 tensor `name` replaced by
 *  `change` of it: a model that loads and runs, and computes the wrong thing. */
async function brokenWeights(
  url: string,
  name: string,
  change: (weight: Float32Array, shape: readonly number[]) => Float32Array,
): Promise<RangeSource> {
  const bytes = await fetchBytes(url);
  const info = (await parseSafetensorsHeader(bufferSource(bytes))).tensors.get(name);
  if (info?.dtype !== 'F32') throw new Error(`${url}: ${name} is not an F32 tensor`);
  const weight = new Float32Array(bytes.slice(info.begin, info.end).buffer);
  bytes.set(new Uint8Array(change(weight, info.shape).buffer), info.begin);
  return bufferSource(bytes);
}

/** The file at `url` with the square weight `name` transposed. */
export function transposedWeights(url: string, name: string): Promise<RangeSource> {
  return brokenWeights(url, name, (weight, [rows, cols]) => {
    if (rows === undefined || rows !== cols || weight.length !== rows * cols) {
      throw new Error(`${url}: ${name} is not a square weight`);
    }
    return Float32Array.from(weight, (_, i) => weight[(i % rows) * rows + Math.floor(i / rows)]!);
  });
}

/** The file at `url` with the vector `name` back to front: a norm that
 *  gives every channel another channel's value. */
export function reversedWeights(url: string, name: string): Promise<RangeSource> {
  return brokenWeights(url, name, (weight) => weight.slice().reverse());
}

/** A report cell of a summary: mean / p99 / max. */
export const summaryCell = ({ mean, p99, max }: Summary) =>
  [mean, p99, max].map((v) => v.toExponential(1)).join(' / ');

/** Writes `rows` as a markdown table to `.parity/results/<family>.md`, the
 *  table a pull request carries. Columns follow the first row's keys. */
async function writeReport(
  family: string,
  rows: readonly Readonly<Record<string, string>>[],
): Promise<void> {
  const columns = Object.keys(rows[0] ?? {});
  const line = (cells: readonly string[]) => `| ${cells.join(' | ')} |`;
  const table = [
    line(columns),
    line(columns.map(() => '---')),
    ...rows.map((row) => line(columns.map((column) => row[column]!))),
  ];
  await commands.writeFile(`${inject('parityResultsDir')}/${family}.md`, `${table.join('\n')}\n`);
}

/** The `models` entries a manifest's variants run. */
type ConfigOf<P extends ParityManifest> = P['variants'][number]['config'];

/** What one model gives on the family's inputs. */
export interface Measured {
  /** The report cells of the run, by column. */
  readonly cells: Readonly<Record<string, string>>;
  /** What the thresholds gate, by metric name. */
  readonly metrics: Readonly<Record<string, number>>;
}

export interface ParityRun<P extends ParityManifest, Model extends { dispose(): void }> {
  readonly manifest: P;
  /** Loads the model of `config` from `modelPath`: the variant's own
   *  weights, or the broken ones. */
  load(config: ConfigOf<P>, modelPath: ModelPath): Promise<Model>;
  /** Runs the model on the family's inputs and compares with the reference.
   *  `margins` are the variant's, as numbers. */
  measure(model: Model, margins: Readonly<Record<string, number>>): Promise<Measured>;
  /** How the first variant's weights are broken: a model that loads and
   *  runs, and has to fail every threshold of every variant. */
  readonly broken: {
    /** What is wrong with it, for the report row. */
    readonly label: string;
    weights(modelPath: string): Promise<RangeSource>;
  };
}

/** The tests of one family: every variant stays within its thresholds, the
 *  broken model fails all of them, and each run is a row of the report.
 *  Throws before any test when the references are stale. */
export async function runParity<P extends ParityManifest, Model extends { dispose(): void }>({
  manifest,
  load,
  measure,
  broken,
}: ParityRun<P, Model>): Promise<void> {
  const { family, variants } = manifest;
  const meta = await readJson<ParityMeta>(`${family}/meta.json`);
  checkMeta(manifest, meta);
  const { noise } = meta;
  const report: Record<string, string>[] = [];
  afterAll(() => writeReport(family, report));

  /** Loads and measures one model before the tests around it, and disposes
   *  it after them. Returns its metrics, for the tests to read. */
  const measured = (
    variant: string,
    config: ConfigOf<P>,
    margins: ParityManifest['variants'][number]['margins'],
    weights: () => Promise<ModelPath>,
    result: (metrics: Measured['metrics']) => string,
  ) => {
    let model: Model | undefined;
    let metrics: Measured['metrics'];
    beforeAll(async () => {
      model = await load(config, await weights());
      const { cells, metrics: got } = await measure(model, limits(margins ?? {}, noise));
      metrics = got;
      report.push({ variant, ...cells, result: result(got) });
    });
    afterAll(() => model?.dispose());
    return () => metrics;
  };

  describe.each(variants)(`${family} $name`, ({ name, config, thresholds, margins }) => {
    const metrics = measured(
      name,
      config,
      margins,
      async () => config.modelPath,
      (got) => {
        const over = exceeded(got, thresholds, noise);
        if (over.length === 0) return 'pass';
        const known = over.every((metric) => thresholds[metric]!.knownIssue !== undefined);
        return known ? `known issue: ${over.join(', ')}` : 'fail';
      },
    );

    for (const [metric, threshold] of Object.entries<Threshold>(thresholds)) {
      const { knownIssue } = threshold;
      const title = `stays within its ${metric} threshold`;
      const test = () => expect(exceeded(metrics(), { [metric]: threshold }, noise)).toEqual([]);
      if (knownIssue === undefined) it(title, test);
      else it.fails(`${title} (known issue: ${knownIssue})`, test);
    }
  });

  describe(`${family} with ${broken.label}`, () => {
    const { name, config, margins } = variants[0]!;
    const metrics = measured(
      `${name}, ${broken.label}`,
      config,
      margins,
      () => broken.weights(config.modelPath),
      () => 'fails, as it has to',
    );

    it.each(variants)('fails every $name threshold', ({ thresholds }) => {
      expect(exceeded(metrics(), thresholds, noise)).toEqual(Object.keys(thresholds));
    });
  });
}

declare module 'vitest' {
  export interface ProvidedContext {
    parityDir: string;
    parityResultsDir: string;
  }
}
