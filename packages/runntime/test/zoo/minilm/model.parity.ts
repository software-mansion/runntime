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
import { cosine, maxAbsError, rankingDifferences, summarize } from '../../parity/metrics.ts';
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
const split = (vectors: readonly Float32Array[]) => ({
  asking: vectors.slice(0, inputs.queries.length),
  ranked: vectors.slice(inputs.queries.length),
});

/** Embeds every text one at a time and compares with the reference. */
async function measure(
  embedder: Embedder,
  margins: Readonly<Record<string, number>>,
): Promise<Measured> {
  const got: Float32Array[] = [];
  for (const text of texts) got.push(await embedder.embed(text));
  const cosineDistance = summarize(got.map((v, i) => 1 - cosine(v, wantVectors[i]!)));
  const error = summarize(got.map((v, i) => maxAbsError(v, wantVectors[i]!)));

  const { positions, items } = rankingDifferences(
    split(got),
    split(wantVectors),
    wantRanking,
    TOP_K,
    margins.score!,
  );
  const ranked = inputs.queries.length * TOP_K;
  const metrics = {
    cosineDistance: cosineDistance.max,
    error: error.max,
    rankMismatches: positions.count / ranked,
    topKMissed: items.count / ranked,
  };
  return {
    cells: {
      '1 − cos (mean / p99 / max)': summaryCell(cosineDistance),
      'max error (mean / p99 / max)': summaryCell(error),
      'ranking positions differing': `${(100 * metrics.rankMismatches).toFixed(1)}%`,
      'top-k missed': `${(100 * metrics.topKMissed).toFixed(1)}%`,
      'near-ties, not gated (ranking / top-k)': `${positions.nearTies} / ${items.nearTies}`,
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
