import { describe, expect, it } from 'vitest';
import { EMBEDDINGGEMMA2_VISION } from '../../../src/zoo/embeddinggemma2/config.ts';
import { EmbeddingGemma2VisionModel } from '../../../src/zoo/embeddinggemma2/visionModel.ts';
import { input } from '../../core/graph/values.ts';
import { attentionScales, standInWeights } from '../attentionScales.ts';

// HF's Gemma attention scales by 1: q and k are RMS-normed per head.
describe('EmbeddingGemma2VisionModel attention', () => {
  it('runs every layer at scale 1', () => {
    const cfg = { ...EMBEDDINGGEMMA2_VISION, hidden: 32, ffn: 64, layers: 2, heads: 2, headDim: 8 };
    const model = standInWeights(new EmbeddingGemma2VisionModel(cfg));
    // 9 patches pool into one image token.
    const p = 9;
    const out = model.forward({
      patches: input([p, 3 * cfg.patch * cfg.patch]),
      xs: input([p, 1]),
      ys: input([p, 1]),
      cos: input([p, cfg.headDim]),
      sin: input([p, cfg.headDim]),
      pool: input([1, p]),
    });
    expect(attentionScales(out)).toEqual([1, 1]);
  });
});
