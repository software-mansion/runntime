/** EmbeddingGemma 2 through createEmbedder against sentence-transformers,
 *  on the same text, photos and speech. Level A compares each vector, grouped
 *  by the tower that reads the input. Level B compares what each asking text
 *  ranks: passages for a question, photos for a caption, clips for a
 *  transcript. */
import {
  createEmbedder,
  type Embedder,
  type EmbedInput,
  type EmbeddingTask,
} from '../../../src/zoo/index.ts';
import { gpu } from '../../gpu.ts';
import {
  readImage,
  readJson,
  readTensors,
  rows,
  runParity,
  summaryCell,
  transposedWeights,
  type Measured,
} from '../../parity/harness.ts';
import { cosine, maxAbsError, rankingDifferences, summarize } from '../../parity/metrics.ts';
import { EMBEDDINGGEMMA2_PARITY, MODALITIES, TOP_K } from './parity.manifest.ts';

await gpu();

const { family } = EMBEDDINGGEMMA2_PARITY;
const inputs = await readJson<{
  queries: string[];
  passages: string[];
  captions: string[];
  images: string[];
  transcripts: string[];
  clips: string[];
}>(`${family}/inputs.json`);
const reference = await readTensors(`${family}/reference.safetensors`);
const images = await Promise.all(inputs.images.map(readImage));

type Name = keyof typeof inputs;
type Modality = (typeof MODALITIES)[number];

/** Every input by its list in `inputs.json`, with the task it is embedded
 *  with. Images and audio take no task. The clips are the samples the
 *  reference decoded. */
const INPUTS: Record<Name, { items: readonly EmbedInput[]; task?: EmbeddingTask }> = {
  queries: { items: inputs.queries, task: 'query' },
  passages: { items: inputs.passages, task: 'document' },
  captions: { items: inputs.captions, task: 'query' },
  images: { items: images.map((image) => ({ image })) },
  transcripts: { items: inputs.transcripts, task: 'query' },
  clips: {
    items: inputs.clips.map((_, i) => ({
      audio: Float32Array.from(reference.get(`samples.${i}`)!),
    })),
  },
};
const NAMES = Object.keys(INPUTS) as Name[];
const want = Object.fromEntries(
  NAMES.map((name) => [name, rows(reference.get(name)!, INPUTS[name].items.length)]),
) as Record<Name, Float32Array[]>;

/** The inputs each tower embeds: level A compares them together. */
const TOWERS: Record<Modality, readonly Name[]> = {
  text: ['queries', 'passages', 'captions', 'transcripts'],
  image: ['images'],
  audio: ['clips'],
};
/** The asking texts of each modality and what they rank, for level B. */
const PAIRS: Record<Modality, { asking: Name; ranked: Name }> = {
  text: { asking: 'queries', ranked: 'passages' },
  image: { asking: 'captions', ranked: 'images' },
  audio: { asking: 'transcripts', ranked: 'clips' },
};

/** Embeds every input one at a time and compares with the reference, each
 *  modality on its own. */
async function measure(
  embedder: Embedder,
  margins: Readonly<Record<string, number>>,
): Promise<Measured> {
  const got = {} as Record<Name, Float32Array[]>;
  for (const name of NAMES) {
    const { items, task } = INPUTS[name];
    got[name] = [];
    for (const input of items) got[name].push(await embedder.embed(input, { task }));
  }
  const cells: Record<string, string> = {};
  const metrics: Record<string, number> = {};
  for (const modality of MODALITIES) {
    const gotVectors = TOWERS[modality].flatMap((name) => got[name]);
    const wantVectors = TOWERS[modality].flatMap((name) => want[name]);
    const cosineDistance = summarize(gotVectors.map((v, i) => 1 - cosine(v, wantVectors[i]!)));
    const error = summarize(gotVectors.map((v, i) => maxAbsError(v, wantVectors[i]!)));

    const { asking, ranked } = PAIRS[modality];
    const { positions, items } = rankingDifferences(
      { asking: got[asking], ranked: got[ranked] },
      { asking: want[asking], ranked: want[ranked] },
      reference.get(`${modality}Ranking`)!,
      TOP_K,
      margins[`${modality}Score`]!,
    );
    const total = got[asking].length * TOP_K;
    metrics[`${modality}CosineDistance`] = cosineDistance.max;
    metrics[`${modality}Error`] = error.max;
    metrics[`${modality}RankMismatches`] = positions.count / total;
    metrics[`${modality}TopKMissed`] = items.count / total;
    const percent = (count: number) => `${((100 * count) / total).toFixed(1)}%`;
    cells[`${modality}: 1 − cos (mean / p99 / max)`] = summaryCell(cosineDistance);
    cells[`${modality}: max error (mean / p99 / max)`] = summaryCell(error);
    cells[`${modality}: ranking positions differing / top-k missed`] =
      `${percent(positions.count)} / ${percent(items.count)}`;
    cells[`${modality}: near-ties, not gated (ranking / top-k)`] =
      `${positions.nearTies} / ${items.nearTies}`;
  }
  return { cells, metrics };
}

await runParity({
  manifest: EMBEDDINGGEMMA2_PARITY,
  load: (config, modelPath) => createEmbedder({ ...config, modelPath }),
  measure,
  broken: {
    // Images and audio reach the text model as input rows, so its weights
    // shape every modality.
    label: 'one text weight transposed',
    weights: (modelPath) =>
      transposedWeights(modelPath, 'language_model.layers.0.self_attn.k_proj.weight'),
  },
});
