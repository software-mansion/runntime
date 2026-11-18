import { describe, expect, it } from 'vitest';
import { EMBEDDINGGEMMA2_VISION } from '../../../src/zoo/embeddinggemma2/config.ts';
import { EmbeddingGemma2VisionModel } from '../../../src/zoo/embeddinggemma2/visionModel.ts';
import { input } from '../../core/graph/values.ts';
import { attentionScales, standInWeights } from '../attentionScales.ts';
import { ropeLayouts } from '../ropeLayouts.ts';

describe('EmbeddingGemma2VisionModel attention', () => {
  const cfg = { ...EMBEDDINGGEMMA2_VISION, hidden: 32, ffn: 64, layers: 2, heads: 2, headDim: 8 };
  /** The output for 9 patches, which pool into one image token. */
  const forward = () => {
    const p = 9;
    return standInWeights(new EmbeddingGemma2VisionModel(cfg)).forward({
      patches: input([p, 3 * cfg.patch * cfg.patch]),
      xs: input([p, 1]),
      ys: input([p, 1]),
      cos: input([p, cfg.headDim]),
      sin: input([p, cfg.headDim]),
      pool: input([1, p]),
    });
  };

  // HF's Gemma attention scales by 1: q and k are RMS-normed per head.
  it('runs every layer at scale 1', () => {
    expect(attentionScales(forward())).toEqual([1, 1]);
  });

  // HF's rotate_half splits in half, and the 2D rope applies it to the x
  // half and the y half of a head on their own. One rope a layer covers q
  // and k.
  it('ropes q and k in the half layout, two ropes a head', () => {
    expect(ropeLayouts(forward())).toEqual(Array(2).fill({ layout: 'half', ropesPerHead: 2 }));
  });
});
