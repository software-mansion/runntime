import { describe, expect, it, vi } from 'vitest';
import { MINILM_L6 } from '../../../src/zoo/minilm/config.ts';
import { MinilmModel } from '../../../src/zoo/minilm/model.ts';
import { chunksFetched, onceInFileOrder, zeroCheckpoint } from '../chunkedLoad.ts';

describe('MinilmModel', () => {
  // The hook rebuilds the position table from two tensors at the front of
  // the file and fuses query, key and value in every layer. Each must bind
  // where the file holds its sources, or the loader fetches an evicted chunk
  // again.
  it('fetches each chunk of its checkpoint once, in file order', async () => {
    const cfg = { ...MINILM_L6, hidden: 32, heads: 4, headDim: 8, ffn: 64, layers: 4, vocab: 40 };
    const layers = Array.from(
      { length: cfg.layers },
      (_, i) => `encoder.layer.${i}.attention.self`,
    );
    const file = zeroCheckpoint(
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
      },
    );
    // The unread pooler and position_ids are warned about.
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    // A projection is 4 KB: two to a chunk, and the file spans far more
    // chunks than the 4 the source keeps.
    const chunks = await chunksFetched(new MinilmModel(cfg), file, 8192);
    warn.mockRestore();
    expect(chunks.length).toBeGreaterThan(12);
    expect(chunks).toEqual(onceInFileOrder(chunks));
  });
});
