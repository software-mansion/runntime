/** all-MiniLM-L6-v2 through createEmbedder against sentence-transformers,
 *  on the same questions and passages. Level A compares each vector, level B
 *  the passages ranked for each question. */
import { createEmbedder, type Embedder } from '../../../src/zoo/index.ts';
import { gpu } from '../../gpu.ts';
import {
  readJson,
  readTensors,
  runParity,
  transposedWeights,
  type Measured,
} from '../../parity/harness.ts';
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

const { family } = MINILM_PARITY;
const inputs = await readJson<{ queries: string[]; passages: string[] }>(`${family}/inputs.json`);
const reference = await readTensors(`${family}/reference.safetensors`);
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

const cell = ({ mean, p99, max }: Summary) =>
  [mean, p99, max].map((v) => v.toExponential(1)).join(' / ');

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
  const metrics = {
    cosineDistance: cosineDistance.max,
    error: error.max,
    rankMismatches: mismatches / ranked,
    topKMissed: missed / ranked,
  };
  return {
    cells: {
      '1 − cos (mean / p99 / max)': cell(cosineDistance),
      'max error (mean / p99 / max)': cell(error),
      'ranking positions differing': `${(100 * metrics.rankMismatches).toFixed(1)}%`,
      'top-k missed': `${(100 * metrics.topKMissed).toFixed(1)}%`,
    },
    metrics,
  };
}

await runParity({
  manifest: MINILM_PARITY,
  load: (config, modelPath) => createEmbedder({ ...config, modelPath }),
  measure,
  broken: {
    label: 'one weight transposed',
    weights: (modelPath) =>
      transposedWeights(modelPath, 'encoder.layer.0.attention.output.dense.weight'),
  },
});
