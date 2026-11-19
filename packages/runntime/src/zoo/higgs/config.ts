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

/** Higgs Audio v2 codec encoder geometry. */
export interface HiggsEncoderConfig {
  /** Rate of the audio in. */
  readonly sampleRate: number;
  /** Rate HuBERT reads. */
  readonly semanticRate: number;
  /** DAC's encoder: channels after the first conv, which each block doubles,
   *  the downsampling of each block (their product is the samples per
   *  frame), the kernel of the first conv and the residual units, and the
   *  last conv's kernel and channels. */
  readonly acoustic: {
    readonly channels: number;
    readonly strides: readonly number[];
    readonly kernel: number;
    readonly outKernel: number;
    readonly latent: number;
  };
  /** HuBERT: the feature convs, the transformer and its conv position
   *  embedding, the zeros padded to each side of its input, and the stride
   *  over its frames that gives one per codec frame. */
  readonly hubert: {
    readonly convDim: number;
    readonly convKernels: readonly number[];
    readonly convStrides: readonly number[];
    readonly hidden: number;
    readonly heads: number;
    readonly layers: number;
    readonly ffn: number;
    readonly posKernel: number;
    readonly posGroups: number;
    readonly pad: number;
    readonly frameStride: number;
  };
  /** The semantic conv stack after HuBERT: its blocks, the dilation of each
   *  block's residual units, and the kernel of every conv but the 1×1 ones. */
  readonly semanticBlocks: number;
  readonly semanticDilations: readonly number[];
  readonly semanticKernel: number;
  readonly codebooks: number;
  readonly codebookSize: number;
  /** Width of a codebook row. */
  readonly codebookDim: number;
}

export const HIGGS_ENCODER: HiggsEncoderConfig = Object.freeze({
  sampleRate: 24_000,
  semanticRate: 16_000,
  acoustic: { channels: 64, strides: [8, 5, 4, 2, 3], kernel: 7, outKernel: 3, latent: 256 },
  hubert: {
    convDim: 512,
    convKernels: [10, 3, 3, 3, 3, 2, 2],
    convStrides: [5, 2, 2, 2, 2, 2, 2],
    hidden: 768,
    heads: 12,
    layers: 12,
    ffn: 3072,
    posKernel: 128,
    posGroups: 16,
    pad: 160,
    frameStride: 2,
  },
  semanticBlocks: 2,
  semanticDilations: [1, 1],
  semanticKernel: 3,
  codebooks: 8,
  codebookSize: 1024,
  codebookDim: 64,
});
