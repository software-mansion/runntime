/** EmbeddingGemma 2 dimensions (google/embeddinggemma-2 config.json): the
 *  text model (text_config) and the vision tower (vision_config). One fixed
 *  size of each. */

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

/** Longest input in tokens, `<bos>` and `<eos>` included, image and video
 *  tokens counted. The model takes 8192, this runner 2048. */
export const EMBEDDINGGEMMA2_MAX_TOKENS = 2048;

export interface EmbeddingGemma2VisionConfig {
  hidden: number;
  ffn: number;
  layers: number;
  heads: number;
  headDim: number;
  eps: number;
  /** Side of a square patch in pixels. */
  patch: number;
  /** Side of the square of patches averaged into one image token. */
  pool: number;
  /** Rows of each of the x and y position embedding tables. */
  positions: number;
  ropeTheta: number;
}

export const EMBEDDINGGEMMA2_VISION: EmbeddingGemma2VisionConfig = {
  hidden: 768,
  ffn: 3072,
  layers: 16,
  heads: 12,
  headDim: 64,
  eps: 1e-6,
  patch: 16,
  pool: 3,
  positions: 10240,
  ropeTheta: 100,
};

/** Token budgets an image may take, most tokens for the most detail. */
export const IMAGE_TOKEN_BUDGETS = [70, 140, 280, 560, 1120] as const;

/** Tokens an image takes at most by default. */
export const IMAGE_TOKENS = 280;

/** Tokens each video frame takes at most. */
export const VIDEO_FRAME_TOKENS = 140;

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
