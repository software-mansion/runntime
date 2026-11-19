import { afterEach, describe, expect, it, vi } from 'vitest';
import { MOONSHINE_STREAMING_TINY } from '../../../src/zoo/moonshine-streaming/config.ts';
import { MoonshineStreamingModel } from '../../../src/zoo/moonshine-streaming/model.ts';
import { chunksFetched, onceInFileOrder } from '../chunkedLoad.ts';
import { moonshineFile } from '../moonshine/hostedFile.ts';

const tiny = MOONSHINE_STREAMING_TINY;
const small = { hidden: 16, ffn: 32, layers: 2, heads: 2, headDim: 8 };
const size = { hidden: small.hidden, vocab: 64 };
const model = () =>
  new MoonshineStreamingModel({
    ...tiny,
    enc: { ...tiny.enc, ...small, windows: tiny.enc.windows.slice(0, small.layers) },
    dec: { ...tiny.dec, ...small, rotaryDim: 6 },
    vocab: size.vocab,
    maxPositions: 32,
  });

/** The encoder parameters the hook computes, by the tensor each reads. */
const RENAMED = {
  'model.encoder.embedder.comp.k': 'model.encoder.embedder.comp.log_k',
  'model.encoder.final_norm.weight': 'model.encoder.final_norm.gamma',
  ...Object.fromEntries(
    Array.from({ length: small.layers }, (_, i) => `model.encoder.layers.${i}`).flatMap((layer) =>
      ['input_layernorm', 'post_attention_layernorm'].map((norm) => [
        `${layer}.${norm}.weight`,
        `${layer}.${norm}.gamma`,
      ]),
    ),
  ),
};

describe('MoonshineStreamingModel', () => {
  afterEach(() => vi.restoreAllMocks());

  it('loads the hosted file without a warning, each chunk once: its hook drops proj_out.weight', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const chunks = await chunksFetched(model(), moonshineFile(model(), size, {}, RENAMED), 2048);
    expect(warn).not.toHaveBeenCalled();
    expect(chunks.length).toBeGreaterThan(12);
    expect(chunks).toEqual(onceInFileOrder(chunks));
  });

  it('still warns about a tensor it does not know', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const file = moonshineFile(model(), size, { 'proj_out.bias': [size.vocab] }, RENAMED);
    await chunksFetched(model(), file, 2048);
    expect(warn).toHaveBeenCalledExactlyOnceWith(
      'loadStateDict: state dict has tensors the model does not read: proj_out.bias.',
    );
  });
});
