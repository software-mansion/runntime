import { describe, expect, it } from 'vitest';
import { EmbeddingGemma2AudioModel } from '../../../src/zoo/embeddinggemma2/audioModel.ts';
import { stackedParts } from '../stackedParts.ts';
import { AUDIO, audioFile } from './towerCheckpoints.ts';

describe('EmbeddingGemma2AudioModel', () => {
  it('stacks q, k and v in that order', async () => {
    expect(await stackedParts(new EmbeddingGemma2AudioModel(AUDIO), audioFile())).toEqual(
      Object.fromEntries(
        Array.from({ length: AUDIO.layers }, (_, i) => [
          `audio_tower.layers.${i}.self_attn.qkv`,
          ['q_proj.linear', 'k_proj.linear', 'v_proj.linear'],
        ]),
      ),
    );
  });
});
