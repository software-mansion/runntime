import { afterEach, describe, expect, it, vi } from 'vitest';
import { MOONSHINE_TINY } from '../../../src/zoo/moonshine/config.ts';
import { MoonshineModel } from '../../../src/zoo/moonshine/model.ts';
import { chunksFetched, onceInFileOrder } from '../chunkedLoad.ts';
import { stackedParts } from '../stackedParts.ts';
import { moonshineFile } from './hostedFile.ts';

const cfg = {
  ...MOONSHINE_TINY,
  dModel: 16,
  ffn: 32,
  encLayers: 2,
  decLayers: 2,
  heads: 2,
  headDim: 8,
  rotaryDim: 6,
  vocab: 64,
};
const size = { hidden: cfg.dModel, vocab: cfg.vocab };
const model = () => new MoonshineModel(cfg);
const QKV = ['q_proj', 'k_proj', 'v_proj'];

describe('MoonshineModel', () => {
  afterEach(() => vi.restoreAllMocks());

  it('loads a file with the tied proj_out.weight without a warning, each chunk once', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const chunks = await chunksFetched(model(), moonshineFile(model(), size), 2048);
    expect(warn).not.toHaveBeenCalled();
    expect(chunks.length).toBeGreaterThan(12);
    expect(chunks).toEqual(onceInFileOrder(chunks));
  });

  it('still warns about a tensor it does not know', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const file = moonshineFile(model(), size, { 'proj_out.bias': [size.vocab] });
    await chunksFetched(model(), file, 2048);
    expect(warn).toHaveBeenCalledExactlyOnceWith(
      'loadStateDict: state dict has tensors the model does not read: proj_out.bias.',
    );
  });

  it('stacks q, k and v, and the cross-attention k and v, in that order', async () => {
    const layers = (side: string, n: number) =>
      Array.from({ length: n }, (_, i) => `model.${side}.layers.${i}`);
    expect(await stackedParts(model(), moonshineFile(model(), size))).toEqual(
      Object.fromEntries([
        ...layers('encoder', cfg.encLayers).map((l) => [`${l}.self_attn.qkv`, QKV]),
        ...layers('decoder', cfg.decLayers).flatMap((l) => [
          [`${l}.self_attn.qkv`, QKV],
          [`${l}.encoder_attn.kv`, ['k_proj', 'v_proj']],
        ]),
      ]),
    );
  });
});
