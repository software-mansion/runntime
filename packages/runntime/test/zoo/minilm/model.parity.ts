/** all-MiniLM-L6-v2 through createEmbedder against sentence-transformers,
 *  on the same questions and passages. Level A compares each vector, level B
 *  the passages ranked for each question. */
import { createEmbedder, type Embedder } from '../../../src/zoo/index.ts';
import { gpu } from '../../gpu.ts';
import {
  readJson,
  readTensors,
  rows,
  runParity,
  summaryCell,
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

const dot = (a: Float32Array, b: Float32Array) => a.reduce((sum, v, i) => sum + v * b[i]!, 0);
/** The reference score of every passage, for each question. */
const wantScores = wantVectors
  .slice(0, inputs.queries.length)
  .map((query) => wantVectors.slice(inputs.queries.length).map((passage) => dot(query, passage)));

/** Embeds every text one at a time and compares with the reference. */
async function measure(
  embedder: Embedder,
  margins: Readonly<Record<string, number>>,
): Promise<Measured> {
  const got: Float32Array[] = [];
  for (const text of texts) got.push(await embedder.embed(text));
  const cosineDistance = summarize(got.map((v, i) => 1 - cosine(v, wantVectors[i]!)));
  const error = summarize(got.map((v, i) => maxAbsError(v, wantVectors[i]!)));

  const queries = got.slice(0, inputs.queries.length);
  const passages = got.slice(inputs.queries.length);
  let mismatches = 0;
  let missed = 0;
  let nearPositions = 0;
  let nearItems = 0;
  queries.forEach((query, q) => {
    const ranking = topK(
      passages.map((passage) => dot(query, passage)),
      TOP_K,
    );
    const want = wantRanking.subarray(q * TOP_K, (q + 1) * TOP_K);
    const positions = rankMismatches(ranking, want, wantScores[q]!, margins.score!);
    const items = missing(ranking, want, wantScores[q]!, margins.score!);
    mismatches += positions.count;
    missed += items.count;
    nearPositions += positions.nearTies;
    nearItems += items.nearTies;
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
      '1 − cos (mean / p99 / max)': summaryCell(cosineDistance),
      'max error (mean / p99 / max)': summaryCell(error),
      'ranking positions differing': `${(100 * metrics.rankMismatches).toFixed(1)}%`,
      'top-k missed': `${(100 * metrics.topKMissed).toFixed(1)}%`,
      'near-ties, not gated (ranking / top-k)': `${nearPositions} / ${nearItems}`,
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
