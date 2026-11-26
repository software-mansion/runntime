/** Small vision and audio tower configs and zero checkpoints with the
 *  tensor names of the hosted file, for the tower load tests. */
import {
  EMBEDDINGGEMMA2_AUDIO,
  EMBEDDINGGEMMA2_VISION,
  type EmbeddingGemma2AudioConfig,
  type EmbeddingGemma2VisionConfig,
} from '../../../src/zoo/embeddinggemma2/config.ts';
import { EmbeddingGemma2AudioModel } from '../../../src/zoo/embeddinggemma2/audioModel.ts';
import { EmbeddingGemma2VisionModel } from '../../../src/zoo/embeddinggemma2/visionModel.ts';
import { zeroCheckpoint } from '../chunkedLoad.ts';

export const VISION: EmbeddingGemma2VisionConfig = {
  ...EMBEDDINGGEMMA2_VISION,
  hidden: 16,
  ffn: 32,
  layers: 2,
  heads: 2,
  headDim: 8,
  patch: 2,
  positions: 8,
};

export const AUDIO: EmbeddingGemma2AudioConfig = {
  ...EMBEDDINGGEMMA2_AUDIO,
  hidden: 16,
  ffn: 32,
  layers: 2,
  heads: 2,
  headDim: 8,
  mels: 16,
  convChannels: [4, 4],
  window: 4,
  convKernel: 3,
  outputDim: 16,
};

const layers = (n: number) => Array.from({ length: n }, (_, i) => i);

/** A zero checkpoint the vision tower of `cfg` loads: the file's position
 *  table, q_norm and k_norm in place of what the hook builds from them. */
export function visionFile(cfg: EmbeddingGemma2VisionConfig = VISION): Uint8Array {
  const attn = (i: number) => `vision_tower.encoder.layers.${i}.self_attn`;
  const embed = 'vision_tower.patch_embedder';
  return zeroCheckpoint(
    new EmbeddingGemma2VisionModel(cfg),
    [
      `${embed}.pos_x.weight`,
      `${embed}.pos_y.weight`,
      ...layers(cfg.layers).map((i) => `${attn(i)}.qkv_norm`),
    ],
    {
      [`${embed}.position_embedding_table`]: [2, cfg.positions, cfg.hidden],
      ...Object.fromEntries(
        layers(cfg.layers).flatMap((i) => [
          [`${attn(i)}.q_norm.weight`, [cfg.headDim]],
          [`${attn(i)}.k_norm.weight`, [cfg.headDim]],
        ]),
      ),
    },
  );
}

/** A zero checkpoint the audio tower of `cfg` loads: the file's
 *  relative_k_proj and per_dim_scale in place of the rel and q_scale the hook
 *  builds from them. */
export function audioFile(cfg: EmbeddingGemma2AudioConfig = AUDIO): Uint8Array {
  const { hidden, headDim } = cfg;
  const layer = (i: number) => `audio_tower.layers.${i}`;
  return zeroCheckpoint(
    new EmbeddingGemma2AudioModel(cfg),
    layers(cfg.layers).flatMap((i) => [
      `${layer(i)}.self_attn.rel`,
      `${layer(i)}.self_attn.q_scale`,
    ]),
    Object.fromEntries(
      layers(cfg.layers).flatMap((i) => [
        [`${layer(i)}.self_attn.relative_k_proj.weight`, [hidden, hidden]],
        [`${layer(i)}.self_attn.per_dim_scale`, [headDim]],
      ]),
    ),
  );
}
