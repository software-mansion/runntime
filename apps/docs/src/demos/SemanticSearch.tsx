import {
  createTextEmbedder,
  models,
  similarity,
  type LoadOptions,
  type TextEmbedder,
} from 'runntime/zoo';
import { useEffect, useState, type FormEvent } from 'react';
import { CORPUS } from './corpus.ts';
import { useModel } from './useModel.ts';
import { WeightsGate } from './WeightsGate.tsx';

const DEFAULT_QUERY = 'I cannot get into my account';
const TOP_K = 4;

/** Module scope on purpose: a stable identity, so the loader hook does not
 *  see a new factory on every render. */
const loadEmbedder = (opts: LoadOptions) =>
  createTextEmbedder(models.textEmbedding.ALL_MINILM_L6_V2.F16, opts);

interface Hit {
  text: string;
  score: number;
}

function Search({ embedder }: { embedder: TextEmbedder }) {
  const [index, setIndex] = useState<Float32Array[]>();
  const [query, setQuery] = useState(DEFAULT_QUERY);
  const [hits, setHits] = useState<Hit[]>([]);
  const [queryMs, setQueryMs] = useState<number>();

  const rank = async (text: string, vectors: Float32Array[]) => {
    const start = performance.now();
    const queryVector = await embedder.embed(text);
    setQueryMs(performance.now() - start);
    setHits(
      vectors
        .map((vector, i) => ({ text: CORPUS[i]!, score: similarity(queryVector, vector) }))
        .sort((a, b) => b.score - a.score)
        .slice(0, TOP_K),
    );
  };

  // One batched pass over the corpus, then the first search, so the panel is
  // not empty.
  useEffect(() => {
    void (async () => {
      const vectors = await embedder.embedBatch(CORPUS);
      setIndex(vectors);
      await rank(DEFAULT_QUERY, vectors);
    })();
  }, [embedder]);

  const onSubmit = (event: FormEvent) => {
    event.preventDefault();
    if (index && query.trim()) void rank(query.trim(), index);
  };

  return (
    <div className="demo not-content">
      <form className="demo-row" onSubmit={onSubmit}>
        <input
          className="demo-input"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Query"
          aria-label="Search query"
        />
        <button className="demo-button" type="submit" disabled={!index}>
          Search
        </button>
      </form>

      <ul className="hits">
        {hits.map((hit) => (
          <li className="hit" key={hit.text}>
            <span className="hit-text">{hit.text}</span>
            <span className="hit-track">
              <span className="hit-bar" style={{ width: `${Math.max(0, hit.score) * 100}%` }} />
            </span>
            <span className="hit-score">{hit.score.toFixed(3)}</span>
          </li>
        ))}
      </ul>

      <div className="demo-stats">
        <span>{CORPUS.length} vectors indexed</span>
        <span>dim {embedder.dim}</span>
        <span>query embed {queryMs === undefined ? '—' : `${queryMs.toFixed(1)} ms`}</span>
      </div>
    </div>
  );
}

export function SemanticSearch() {
  const { state, start } = useModel(loadEmbedder);

  // The embedder owns GPU buffers; release them when the page unmounts.
  useEffect(() => {
    if (state.status !== 'ready') return;
    const embedder = state.model;
    return () => embedder.dispose();
  }, [state]);

  return (
    <WeightsGate
      state={state}
      onStart={start}
      note="all-MiniLM-L6-v2, 45 MB of safetensors from the Hugging Face Hub, run in f16 on the GPU. Cached in OPFS after the first load."
    >
      {(embedder) => <Search embedder={embedder} />}
    </WeightsGate>
  );
}
