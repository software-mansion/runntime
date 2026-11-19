/** Reverb's fixed architecture: a bidirectional Qwen3 backbone over text and
 *  Higgs Audio v2 codec tokens. */

export interface ReverbConfig {
  readonly layers: number;
  readonly hidden: number;
  readonly ffn: number;
  readonly heads: number;
  readonly kvHeads: number;
  readonly headDim: number;
  readonly eps: number;
  readonly ropeTheta: number;
  /** Text vocabulary and the rank of its factorized embedding. */
  readonly vocab: number;
  readonly embeddingRank: number;
  /** Codec codebooks, and codes per codebook including the mask. */
  readonly codebooks: number;
  readonly codes: number;
}

export const REVERB: ReverbConfig = Object.freeze({
  layers: 24,
  hidden: 1024,
  ffn: 2048,
  heads: 12,
  kvHeads: 6,
  headDim: 128,
  eps: 1e-6,
  ropeTheta: 1e6,
  vocab: 62035,
  embeddingRank: 256,
  codebooks: 8,
  codes: 1025,
});

/** Sample rate of the codec's audio: 25 frames of 960 samples a second. */
export const SAMPLE_RATE = 24_000;

/** Codec frames per second of audio. */
export const FRAME_RATE = 25;

/** The shortest reference clip Reverb takes, in seconds. */
export const MIN_VOICE_SECONDS = 3;

/** The languages Reverb speaks, as ISO 639-1 codes. */
export const LANGUAGES = ['en', 'de', 'fr', 'es', 'it', 'pl', 'pt'] as const;

/** One of the languages Reverb speaks. */
export type ReverbLanguage = (typeof LANGUAGES)[number];

/** The mask code, in every codebook: the one after the codec's 1024. */
export const MASK = 1024;

/** Codebooks 0 and 1, which carry content and timing, come out of the main
 *  passes; the polish passes fill the rest. */
export const COARSE_CODEBOOKS = 2;
