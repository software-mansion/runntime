import { createEmbedder, models, similarity, type Embedder, type LoadOptions } from 'runntime/zoo';
import { useEffect, useState, type FormEvent, type ReactNode } from 'react';
import { CORPUS } from './corpus.ts';
import { useModel } from './useModel.ts';
import { WeightsGate } from './WeightsGate.tsx';

const DEFAULT_QUERY = 'I cannot get into my account';
const TOP_K = 4;

/** The hosted embedding models, in the order the picker lists them. Module
 *  scope on purpose: each `load` keeps a stable identity, so the loader hook
 *  does not see a new factory on every render. */
const EMBEDDING_MODELS = {
  ALL_MINILM_L6_V2: {
    label: 'all-MiniLM-L6-v2',
    note: 'all-MiniLM-L6-v2, 45 MB of safetensors from the Hugging Face Hub, run in f16 on the GPU. 384-dim vectors, English text. Cached in OPFS after the first load.',
    load: (opts: LoadOptions) => createEmbedder(models.embedding.ALL_MINILM_L6_V2.F16, opts),
  },
  EMBEDDINGGEMMA_2: {
    label: 'EmbeddingGemma 2',
    note: 'EmbeddingGemma 2, the 542 MB text model and a 32 MB tokenizer from the Hugging Face Hub, run in f16 on the GPU. 768-dim vectors, 100+ languages. Cached in OPFS after the first load.',
    // Text only: the vision and audio towers are never used here.
    load: (opts: LoadOptions) =>
      createEmbedder(
        { ...models.embedding.EMBEDDINGGEMMA_2.F16, visionPath: null, audioPath: null },
        opts,
      ),
  },
};

type ModelKey = keyof typeof EMBEDDING_MODELS;

const isModelKey = (key: string): key is ModelKey => Object.hasOwn(EMBEDDING_MODELS, key);

function ModelSelect({
  value,
  onChange,
  disabled,
}: {
  value: ModelKey;
  onChange: (key: ModelKey) => void;
  disabled: boolean;
}) {
  return (
    <select
      className="demo-select"
      value={value}
      onChange={(e) => isModelKey(e.target.value) && onChange(e.target.value)}
      disabled={disabled}
      aria-label="Embedding model"
    >
      {Object.entries(EMBEDDING_MODELS).map(([key, { label }]) => (
        <option key={key} value={key}>
          {label}
        </option>
      ))}
    </select>
  );
}

interface Hit {
  text: string;
  score: number;
}

function Search({ embedder, modelSelect }: { embedder: Embedder; modelSelect: ReactNode }) {
  const [index, setIndex] = useState<Float32Array[]>();
  const [query, setQuery] = useState(DEFAULT_QUERY);
  const [hits, setHits] = useState<Hit[]>([]);
  const [queryMs, setQueryMs] = useState<number>();

  const rank = async (text: string, vectors: Float32Array[]) => {
    const start = performance.now();
    const queryVector = await embedder.embed(text, { task: 'query' });
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
      const vectors = await embedder.embedBatch(CORPUS, { task: 'document' });
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
      {modelSelect}
      <form className="demo-row demo-row-actions" onSubmit={onSubmit}>
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

function Demo({
  modelKey,
  onModelChange,
}: {
  modelKey: ModelKey;
  onModelChange: (key: ModelKey) => void;
}) {
  const { note, load } = EMBEDDING_MODELS[modelKey];
  const { state, start } = useModel(load);

  // The embedder owns GPU buffers; release them when the page unmounts or the
  // model changes.
  useEffect(() => {
    if (state.status !== 'ready') return;
    const embedder = state.model;
    return () => embedder.dispose();
  }, [state]);

  const modelSelect = (
    <ModelSelect value={modelKey} onChange={onModelChange} disabled={state.status === 'loading'} />
  );

  return (
    <WeightsGate state={state} onStart={start} note={note} controls={modelSelect}>
      {(embedder) => <Search embedder={embedder} modelSelect={modelSelect} />}
    </WeightsGate>
  );
}

export function SemanticSearch() {
  const [modelKey, setModelKey] = useState<ModelKey>('ALL_MINILM_L6_V2');

  // Keyed by the model: a switch unmounts the old demo, which disposes its
  // embedder, and mounts the new one idle. Nothing downloads until the user
  // clicks.
  return <Demo key={modelKey} modelKey={modelKey} onModelChange={setModelKey} />;
}
