/** all-MiniLM-L6-v2 through createEmbedder against sentence-transformers,
 *  on the same questions and passages. Level A compares each vector, level B
 *  the passages ranked for each question. */
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { createEmbedder, type Embedder } from '../../../src/zoo/index.ts';
import { gpu } from '../../gpu.ts';
import { readJson, readTensors, transposedWeights, writeReport } from '../../parity/harness.ts';
import { exceeded } from '../../parity/manifest.ts';
import {
  cosine,
  maxAbsError,
  missing,
  rankMismatches,
  summarize,
  topK,
  type Summary,
} from '../../parity/metrics.ts';
import { MINILM_PARITY, TOP_K } from './parity.manifest.ts';

await gpu();

const { family, variants } = MINILM_PARITY;
const inputs = await readJson<{ queries: string[]; passages: string[] }>(`${family}/inputs.json`);
const reference = await readTensors(`${family}/reference.safetensors`);
const { noise } = await readJson<{ noise: Record<string, number> }>(`${family}/meta.json`);
const texts = [...inputs.queries, ...inputs.passages];
const wantVectors = [
  ...rows(reference.get('queries')!, inputs.queries.length),
  ...rows(reference.get('passages')!, inputs.passages.length),
];
const wantRanking = reference.get('ranking')!;

/** The `count` rows of a `[count, dim]` reference tensor. */
function rows(tensor: Float32Array | Int32Array, count: number): Float32Array[] {
  const dim = tensor.length / count;
  return Array.from({ length: count }, (_, i) =>
    Float32Array.from(tensor.subarray(i * dim, (i + 1) * dim)),
  );
}

interface Measured {
  cosineDistance: Summary;
  error: Summary;
  /** What the thresholds gate: the worst cosine distance and error, the
   *  share of ranking positions that differ, and the share of top-k passages
   *  missed. */
  metrics: { cosineDistance: number; error: number; rankMismatches: number; topKMissed: number };
}

/** Embeds every text one at a time and compares with the reference. */
async function measure(embedder: Embedder): Promise<Measured> {
  const got: Float32Array[] = [];
  for (const text of texts) got.push(await embedder.embed(text));
  const cosineDistance = summarize(got.map((v, i) => 1 - cosine(v, wantVectors[i]!)));
  const error = summarize(got.map((v, i) => maxAbsError(v, wantVectors[i]!)));

  const queries = got.slice(0, inputs.queries.length);
  const passages = got.slice(inputs.queries.length);
  let mismatches = 0;
  let missed = 0;
  queries.forEach((query, q) => {
    const scores = passages.map((passage) => query.reduce((dot, v, i) => dot + v * passage[i]!, 0));
    const want = wantRanking.subarray(q * TOP_K, (q + 1) * TOP_K);
    mismatches += rankMismatches(topK(scores, TOP_K), want);
    missed += missing(topK(scores, TOP_K), want);
  });
  const ranked = queries.length * TOP_K;
  return {
    cosineDistance,
    error,
    metrics: {
      cosineDistance: cosineDistance.max,
      error: error.max,
      rankMismatches: mismatches / ranked,
      topKMissed: missed / ranked,
    },
  };
}

const report: Record<string, string>[] = [];
afterAll(() => writeReport(family, report));

const cell = ({ mean, p99, max }: Summary) =>
  [mean, p99, max].map((v) => v.toExponential(1)).join(' / ');

/** Adds the report row of one run. */
function record(variant: string, { cosineDistance, error, metrics }: Measured, result: string) {
  report.push({
    variant,
    '1 − cos (mean / p99 / max)': cell(cosineDistance),
    'max error (mean / p99 / max)': cell(error),
    'ranking positions differing': `${(100 * metrics.rankMismatches).toFixed(1)}%`,
    'top-k missed': `${(100 * metrics.topKMissed).toFixed(1)}%`,
    result,
  });
}

describe.each(variants)('all-MiniLM-L6-v2 $name', ({ name, config, thresholds }) => {
  let embedder: Embedder | undefined;
  let measured: Measured;
  beforeAll(async () => {
    embedder = await createEmbedder(config);
    measured = await measure(embedder);
    record(
      name,
      measured,
      exceeded(measured.metrics, thresholds, noise).length === 0 ? 'pass' : 'fail',
    );
  });
  afterAll(() => embedder?.dispose());

  it('stays within its thresholds', () => {
    expect(exceeded(measured.metrics, thresholds, noise)).toEqual([]);
  });
});

describe('all-MiniLM-L6-v2 with a transposed weight', () => {
  /** The weight the broken model gets transposed. */
  const BROKEN = 'encoder.layer.0.attention.output.dense.weight';
  let embedder: Embedder | undefined;
  let measured: Measured;
  beforeAll(async () => {
    const { config } = variants[0];
    const modelPath = await transposedWeights(config.modelPath, BROKEN);
    embedder = await createEmbedder({ ...config, modelPath });
    measured = await measure(embedder);
    record('F32, one weight transposed', measured, 'fails, as it has to');
  });
  afterAll(() => embedder?.dispose());

  it.each(variants)('fails every $name threshold', ({ thresholds }) => {
    expect(exceeded(measured.metrics, thresholds, noise)).toEqual(Object.keys(thresholds));
  });
});
