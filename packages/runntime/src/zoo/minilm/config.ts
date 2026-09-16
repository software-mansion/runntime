/** all-MiniLM-L6-v2 dimensions (sentence-transformers checkpoint config.json).
 *  One fixed size — the checkpoint family has no variants. */

export interface MinilmConfig {
  hidden: number;
  layers: number;
  heads: number;
  headDim: number;
  ffn: number;
  vocab: number;
  maxPositions: number;
  eps: number;
}

export const MINILM_L6: MinilmConfig = {
  hidden: 384,
  layers: 6,
  heads: 12,
  headDim: 32,
  ffn: 1536,
  vocab: 30522,
  maxPositions: 512,
  eps: 1e-12,
};

export const EMBEDDING_DIM = MINILM_L6.hidden;
