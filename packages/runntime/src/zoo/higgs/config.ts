/** Higgs Audio v2 codec decoder geometry. */
export interface HiggsDecoderConfig {
  /** RVQ codebooks per frame. */
  readonly codebooks: number;
  readonly codebookSize: number;
  /** Width of a frame's input row, from the codebook table. */
  readonly latent: number;
  /** Channels after the first conv; each block halves them. */
  readonly channels: number;
  /** Upsampling of each block; their product is the samples per frame. */
  readonly strides: readonly number[];
  /** Kernel of every conv but the 1×1 ones. */
  readonly kernel: number;
}

export const HIGGS_DECODER: HiggsDecoderConfig = Object.freeze({
  codebooks: 8,
  codebookSize: 1024,
  latent: 256,
  channels: 1024,
  strides: [8, 5, 4, 2, 3],
  kernel: 7,
});
