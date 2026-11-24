import { describe, expect, it } from 'vitest';
import { EmbeddingGemma2VisionModel } from '../../../src/zoo/embeddinggemma2/visionModel.ts';
import { stackedParts } from '../stackedParts.ts';
import { VISION, visionFile } from './towerCheckpoints.ts';

describe('EmbeddingGemma2VisionModel', () => {
  it('stacks q, k and v in that order', async () => {
    expect(await stackedParts(new EmbeddingGemma2VisionModel(VISION), visionFile())).toEqual(
      Object.fromEntries(
        Array.from({ length: VISION.layers }, (_, i) => [
          `vision_tower.encoder.layers.${i}.self_attn.qkv`,
          ['q_proj.linear', 'k_proj.linear', 'v_proj.linear'],
        ]),
      ),
    );
  });
});
