import { describe, expect, it } from 'vitest';
import {
  EMBEDDINGGEMMA2_TEXT,
  type EmbeddingGemma2Config,
} from '../../../src/zoo/embeddinggemma2/config.ts';
import { EmbeddingGemma2Model } from '../../../src/zoo/embeddinggemma2/model.ts';
import { chunksFetched, onceInFileOrder, zeroCheckpoint } from '../chunkedLoad.ts';
import { input } from '../../core/graph/values.ts';
import { attentionScales, standInWeights } from '../attentionScales.ts';
import { ropeLayouts } from '../ropeLayouts.ts';
import { stackedParts } from '../stackedParts.ts';

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
/** A zero checkpoint with the file's q_norm and k_norm in place of the
 *  qkv_norm the hook builds from them. */
const file = () =>
  zeroCheckpoint(
    new EmbeddingGemma2Model(cfg),
    cfg.layerTypes.map((_, i) => `${layer(i)}.qkv_norm`),
    Object.fromEntries(
      cfg.layerTypes.flatMap((type, i) => {
        const { headDim } = cfg.attention[type];
        return [
          [`${layer(i)}.q_norm.weight`, [headDim]],
          [`${layer(i)}.k_norm.weight`, [headDim]],
        ];
      }),
    ),
  );

describe('EmbeddingGemma2Model', () => {
  // Each layer stacks q_proj, k_proj and v_proj, and the hook builds its
  // qkv_norm from q_norm and k_norm, rewrites layer_scalar and scales the PLE
  // projection. Each must bind where the file holds its sources, or the
  // loader fetches an evicted chunk again.
  it('fetches each chunk of its checkpoint once, in file order', async () => {
    // A layer is about 30 KB: a few chunks each, and the file spans far more
    // chunks than the 4 the source keeps.
    const chunks = await chunksFetched(new EmbeddingGemma2Model(cfg), file(), 8192);
    expect(chunks.length).toBeGreaterThan(12);
    expect(chunks).toEqual(onceInFileOrder(chunks));
  });

  it('stacks q, k and v in that order', async () => {
    expect(await stackedParts(new EmbeddingGemma2Model(cfg), file())).toEqual(
      Object.fromEntries(
        cfg.layerTypes.map((_, i) => [`${layer(i)}.qkv`, ['q_proj', 'k_proj', 'v_proj']]),
      ),
    );
  });
});

describe('EmbeddingGemma2Model attention', () => {
  /** The output of a sliding layer and a full one for 6 tokens. */
  const forward = () => {
    const cfg: EmbeddingGemma2Config = {
      ...EMBEDDINGGEMMA2_TEXT,
      vocab: 64,
      hidden: 32,
      ffn: 64,
      layers: 2,
      heads: 2,
      pleDim: 16,
      layerTypes: ['sliding', 'full'],
      attention: {
        sliding: { headDim: 8, kvHeads: 1, theta: 10000 },
        full: { headDim: 16, kvHeads: 1, theta: 1000000 },
      },
    };
    const model = standInWeights(new EmbeddingGemma2Model(cfg));
    const t = 6;
    const rope = (headDim: number) => ({ cos: input([t, headDim]), sin: input([t, headDim]) });
    return model.forward({
      x0: input([t, cfg.hidden]),
      rope: { sliding: rope(8), full: rope(16) },
      selector: input([1, t]),
      segments: input([t, 2]),
      maxSegment: t,
    });
  };

  // HF's Gemma attention scales by 1: q and k are RMS-normed per head.
  it('runs every layer at scale 1', () => {
    expect(attentionScales(forward())).toEqual([1, 1]);
  });

  // HF's rotate_half splits a head in half: pairs (p, p + headDim/2). q and
  // k of each layer.
  it('ropes q and k in the half layout', () => {
    expect(ropeLayouts(forward())).toEqual(Array(4).fill({ layout: 'half', ropesPerHead: 1 }));
  });
});
