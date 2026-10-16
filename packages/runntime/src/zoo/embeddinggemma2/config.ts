/** EmbeddingGemma 2 text model dimensions (google/embeddinggemma-2
 *  config.json, text_config). One fixed size: the vision and audio towers
 *  are separate models. */

/** The two layer kinds: sliding layers see ±slidingWindow tokens, full
 *  layers the whole input. */
export type LayerType = 'sliding' | 'full';

export interface EmbeddingGemma2Config {
  vocab: number;
  hidden: number;
  ffn: number;
  layers: number;
  heads: number;
  /** Width of each layer's per-layer input (PLE). */
  pleDim: number;
  embeddingDim: number;
  eps: number;
  /** Keys a sliding layer sees on each side of a query. */
  slidingWindow: number;
  layerTypes: readonly LayerType[];
  /** Head size, kv heads and rope theta per layer kind. */
  attention: Record<LayerType, { headDim: number; kvHeads: number; theta: number }>;
}

export const EMBEDDINGGEMMA2_TEXT: EmbeddingGemma2Config = {
  vocab: 262144,
  hidden: 512,
  ffn: 2048,
  layers: 24,
  heads: 4,
  pleDim: 512,
  embeddingDim: 768,
  eps: 1e-6,
  slidingWindow: 512,
  // Every sixth layer is full.
  layerTypes: Array.from({ length: 24 }, (_, i) => (i % 6 === 5 ? 'full' : 'sliding')),
  attention: {
    sliding: { headDim: 256, kvHeads: 2, theta: 10000 },
    full: { headDim: 512, kvHeads: 1, theta: 1000000 },
  },
};

export const EMBEDDING_DIM = EMBEDDINGGEMMA2_TEXT.embeddingDim;

/** Output sizes the model was trained to be cut to (Matryoshka). */
export const EMBEDDING_DIMS = [768, 512, 256, 128] as const;

/** Longest input in tokens, `<bos>` and `<eos>` included. The model takes
 *  8192, this runner 2048. */
export const EMBEDDINGGEMMA2_MAX_TOKENS = 2048;

/** The instruction each task puts before the text (the checkpoint's
 *  sentence-transformers prompts). `query` and `document` are the two sides
 *  of search: queries get one, the texts searched get the other. */
export const EMBEDDINGGEMMA2_PROMPTS = {
  query: 'task: search result | query: ',
  document: 'title: none | text: ',
  questionAnswering: 'task: question answering | query: ',
  factChecking: 'task: fact checking | query: ',
  classification: 'task: classification | query: ',
  clustering: 'task: clustering | query: ',
  similarity: 'task: sentence similarity | query: ',
  codeRetrieval: 'task: code retrieval | query: ',
} as const;
