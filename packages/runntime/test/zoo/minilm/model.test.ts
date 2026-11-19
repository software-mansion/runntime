import { afterEach, describe, expect, it, vi } from 'vitest';
import { MINILM_L6 } from '../../../src/zoo/minilm/config.ts';
import { MinilmModel } from '../../../src/zoo/minilm/model.ts';
import { chunksFetched, onceInFileOrder, zeroCheckpoint } from '../chunkedLoad.ts';

const cfg = { ...MINILM_L6, hidden: 32, heads: 4, headDim: 8, ffn: 64, layers: 4, vocab: 40 };
const layers = Array.from({ length: cfg.layers }, (_, i) => `encoder.layer.${i}.attention.self`);

/** A zero checkpoint with the tensor names of the hosted BertModel file,
 *  the pooler and `position_ids` included, and `extra` on top. */
const hostedFile = (extra: Record<string, number[]> = {}) =>
  zeroCheckpoint(
    new MinilmModel(cfg),
    layers.flatMap((self) => [`${self}.qkv.weight`, `${self}.qkv.bias`]),
    {
      'embeddings.token_type_embeddings.weight': [2, cfg.hidden],
      'embeddings.position_ids': [1, cfg.maxPositions],
      'pooler.dense.weight': [cfg.hidden, cfg.hidden],
      'pooler.dense.bias': [cfg.hidden],
      ...Object.fromEntries(
        layers.flatMap((self) =>
          ['query', 'key', 'value'].flatMap((part) => [
            [`${self}.${part}.weight`, [cfg.hidden, cfg.hidden]],
            [`${self}.${part}.bias`, [cfg.hidden]],
          ]),
        ),
      ),
      ...extra,
    },
  );

describe('MinilmModel', () => {
  afterEach(() => vi.restoreAllMocks());

  // The hook rebuilds the position table from two tensors at the front of
  // the file and fuses query, key and value in every layer. Each must bind
  // where the file holds its sources, or the loader fetches an evicted chunk
  // again.
  it('fetches each chunk of its checkpoint once, in file order', async () => {
    // A projection is 4 KB: two to a chunk, and the file spans far more
    // chunks than the 4 the source keeps.
    const chunks = await chunksFetched(new MinilmModel(cfg), hostedFile(), 8192);
    expect(chunks.length).toBeGreaterThan(12);
    expect(chunks).toEqual(onceInFileOrder(chunks));
  });

  it('loads the hosted file without a warning: its hook drops the tensors it never reads', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await chunksFetched(new MinilmModel(cfg), hostedFile(), 8192);
    expect(warn).not.toHaveBeenCalled();
  });

  it('still warns about a tensor it does not know', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await chunksFetched(new MinilmModel(cfg), hostedFile({ 'pooler.extra.weight': [2] }), 8192);
    expect(warn).toHaveBeenCalledExactlyOnceWith(
      'loadStateDict: state dict has tensors the model does not read: pooler.extra.weight.',
    );
  });
});
