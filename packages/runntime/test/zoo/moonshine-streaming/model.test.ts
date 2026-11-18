import { afterEach, describe, expect, it, vi } from 'vitest';
import { MOONSHINE_STREAMING_TINY } from '../../../src/zoo/moonshine-streaming/config.ts';
import { MoonshineStreamingModel } from '../../../src/zoo/moonshine-streaming/model.ts';
import { chunksFetched, onceInFileOrder } from '../chunkedLoad.ts';
import { stackedParts } from '../stackedParts.ts';
import { moonshineFile } from '../moonshine/hostedFile.ts';
import { frontendFrames } from '../../../src/zoo/moonshine-streaming/frontend.ts';
import { input } from '../../core/graph/values.ts';
import { attentionScales, standInWeights } from '../attentionScales.ts';
import { ropeLayouts } from '../ropeLayouts.ts';

const tiny = MOONSHINE_STREAMING_TINY;
const small = { hidden: 16, ffn: 32, layers: 2, heads: 2, headDim: 8 };
const size = { hidden: small.hidden, vocab: 64 };
const QKV = ['q_proj', 'k_proj', 'v_proj'];
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

  it('stacks q, k and v, and the cross-attention k and v, in that order', async () => {
    const layers = (side: string) =>
      Array.from({ length: small.layers }, (_, i) => `model.${side}.layers.${i}`);
    expect(await stackedParts(model(), moonshineFile(model(), size, {}, RENAMED))).toEqual(
      Object.fromEntries([
        ...layers('encoder').map((l) => [`${l}.self_attn.qkv`, QKV]),
        ...layers('decoder').flatMap((l) => [
          [`${l}.self_attn.qkv`, QKV],
          [`${l}.encoder_attn.kv`, ['k_proj', 'v_proj']],
        ]),
      ]),
    );
  });
});

describe('MoonshineStreamingModel attention', () => {
  it('scales the encoder, decoder and cross-attention scores by 1/√headDim', () => {
    const m = standInWeights(model());
    const { encoder, decoder } = m.model;
    const frames = frontendFrames(1600, m.cfg.frameLen).input;
    const encOut = encoder.forward(input([frames, m.cfg.frameLen]));
    const cross = decoder.precomputeCrossKV(decoder.adapt(encOut));
    const table = input([3, small.headDim]);
    const out = decoder.forward(input([3, small.hidden]), table, table, cross);
    const attentions = 3 * small.layers;
    expect(attentionScales(out)).toEqual(Array(attentions).fill(1 / Math.sqrt(small.headDim)));
  });

  // HF's Moonshine Streaming rotate_half takes x[..., 0::2] and x[..., 1::2]:
  // pairs (2p, 2p + 1). Only the decoder self-attention has a rope.
  it('ropes the decoder q and k in the interleaved layout', () => {
    const m = standInWeights(model());
    const { encoder, decoder } = m.model;
    const frames = frontendFrames(1600, m.cfg.frameLen).input;
    const encOut = encoder.forward(input([frames, m.cfg.frameLen]));
    const cross = decoder.precomputeCrossKV(decoder.adapt(encOut));
    const table = input([3, small.headDim]);
    const out = decoder.forward(input([3, small.hidden]), table, table, cross);
    expect(ropeLayouts(out)).toEqual(
      Array(2 * small.layers).fill({ layout: 'interleaved', ropesPerHead: 1 }),
    );
  });
});
