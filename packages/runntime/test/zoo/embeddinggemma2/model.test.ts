import { describe, expect, it } from 'vitest';
import {
  EMBEDDINGGEMMA2_TEXT,
  type EmbeddingGemma2Config,
} from '../../../src/zoo/embeddinggemma2/config.ts';
import { EmbeddingGemma2Model } from '../../../src/zoo/embeddinggemma2/model.ts';
import { chunksFetched, onceInFileOrder, zeroCheckpoint } from '../chunkedLoad.ts';

describe('EmbeddingGemma2Model', () => {
  // The hook builds each layer's qkv from q_proj, k_proj and v_proj and its
  // qkv_norm from q_norm and k_norm, rewrites layer_scalar and scales the PLE
  // projection. Each must bind where the file holds its sources, or the
  // loader fetches an evicted chunk again.
  it('fetches each chunk of its checkpoint once, in file order', async () => {
    const cfg: EmbeddingGemma2Config = {
      ...EMBEDDINGGEMMA2_TEXT,
      vocab: 64,
      hidden: 32,
      ffn: 64,
      layers: 6,
      heads: 2,
      pleDim: 16,
      layerTypes: ['sliding', 'sliding', 'sliding', 'sliding', 'sliding', 'full'],
      attention: {
        sliding: { headDim: 8, kvHeads: 1, theta: 10000 },
        full: { headDim: 16, kvHeads: 1, theta: 1000000 },
      },
    };
    const layer = (i: number) => `language_model.layers.${i}.self_attn`;
    const file = zeroCheckpoint(
      new EmbeddingGemma2Model(cfg),
      cfg.layerTypes.flatMap((_, i) => [`${layer(i)}.qkv.weight`, `${layer(i)}.qkv_norm`]),
      Object.fromEntries(
        cfg.layerTypes.flatMap((type, i) => {
          const { headDim, kvHeads } = cfg.attention[type];
          return [
            [`${layer(i)}.q_proj.weight`, [cfg.heads * headDim, cfg.hidden]],
            [`${layer(i)}.k_proj.weight`, [kvHeads * headDim, cfg.hidden]],
            [`${layer(i)}.v_proj.weight`, [kvHeads * headDim, cfg.hidden]],
            [`${layer(i)}.q_norm.weight`, [headDim]],
            [`${layer(i)}.k_norm.weight`, [headDim]],
          ];
        }),
      ),
    );
    // A layer is about 30 KB: a few chunks each, and the file spans far more
    // chunks than the 4 the source keeps.
    const chunks = await chunksFetched(new EmbeddingGemma2Model(cfg), file, 8192);
    expect(chunks.length).toBeGreaterThan(12);
    expect(chunks).toEqual(onceInFileOrder(chunks));
  });
});
