import { afterEach, describe, expect, it, vi } from 'vitest';
import { pending } from '../../../src/core/graph/value.ts';
import { bufferSource, fromSafetensors } from '../../../src/core/weights/safetensors.ts';
import { REVERB, type ReverbConfig } from '../../../src/zoo/reverb/config.ts';
import { ReverbModel } from '../../../src/zoo/reverb/model.ts';
import { input, materializedInput } from '../../core/graph/values.ts';
import { attentionScales, standInWeights } from '../attentionScales.ts';
import { chunksFetched, onceInFileOrder, zeroCheckpoint } from '../chunkedLoad.ts';
import { ropeLayouts } from '../ropeLayouts.ts';

const cfg: ReverbConfig = {
  ...REVERB,
  layers: 4,
  hidden: 32,
  ffn: 64,
  heads: 2,
  kvHeads: 1,
  headDim: 16,
  vocab: 40,
  embeddingRank: 8,
  codebooks: 3,
  codes: 5,
};
const { hidden, ffn, heads, kvHeads, headDim: d, codebooks, codes } = cfg;
const layers = Array.from({ length: cfg.layers }, (_, i) => `llm.layers.${i}`);

/** A zero checkpoint with the tensor names of a Reverb release file,
 *  `codebook_layer_offsets` included, and `extra` on top. */
const hostedFile = (extra: Record<string, number[]> = {}) =>
  zeroCheckpoint(
    new ReverbModel(cfg),
    [
      'audio_heads.coarse.weight',
      'audio_heads.fine.weight',
      ...layers.flatMap((L) => [
        `${L}.self_attn.qk.weight`,
        `${L}.self_attn.qk_norm`,
        `${L}.mlp.up_gate.weight`,
      ]),
    ],
    {
      codebook_layer_offsets: [codebooks],
      'audio_heads.weight': [codebooks * codes, hidden],
      ...Object.fromEntries(
        layers.flatMap((L) => [
          [`${L}.self_attn.q_proj.weight`, [heads * d, hidden]],
          [`${L}.self_attn.k_proj.weight`, [kvHeads * d, hidden]],
          [`${L}.self_attn.q_norm.weight`, [d]],
          [`${L}.self_attn.k_norm.weight`, [d]],
          [`${L}.mlp.up_proj.weight`, [ffn, hidden]],
          [`${L}.mlp.gate_proj.weight`, [ffn, hidden]],
        ]),
      ),
      ...extra,
    },
  );

describe('ReverbModel', () => {
  afterEach(() => vi.restoreAllMocks());

  // The hook fuses q/k and up/gate in every layer and splits the audio heads
  // in two. Each must bind where the file holds its sources, or the loader
  // fetches an evicted chunk again.
  it('fetches each chunk of its checkpoint once, in file order', async () => {
    // A [32, 32] projection is 4 KB: two to a chunk, and the file spans far
    // more chunks than the 4 the source keeps.
    const chunks = await chunksFetched(new ReverbModel(cfg), hostedFile(), 8192);
    expect(chunks.length).toBeGreaterThan(12);
    expect(chunks).toEqual(onceInFileOrder(chunks));
  });

  it('loads the release file without a warning: its hook drops the tensor it never reads', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await chunksFetched(new ReverbModel(cfg), hostedFile(), 8192);
    expect(warn).not.toHaveBeenCalled();
  });

  it('warns about a tensor it does not know', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await chunksFetched(new ReverbModel(cfg), hostedFile({ 'llm.extra': [2] }), 8192);
    expect(warn).toHaveBeenCalledExactlyOnceWith(
      'loadStateDict: state dict has tensors the model does not read: llm.extra.',
    );
  });

  it('throws CHECKPOINT_MISMATCH for a tensor it does not know when strict', async () => {
    const sd = await fromSafetensors(bufferSource(hostedFile({ 'llm.extra': [2] })));
    await expect(
      new ReverbModel(cfg).loadStateDict(sd, {
        strict: true,
        upload: (_data, shape) => pending(shape, 'reshape', []),
      }),
    ).rejects.toMatchObject({ code: 'CHECKPOINT_MISMATCH', message: /unexpected: llm\.extra\./ });
  });
});

describe('ReverbModel attention', () => {
  /** The output of every layer for 6 rows. */
  const forward = () => {
    const t = 6;
    return standInWeights(new ReverbModel(cfg)).forward(input([t, hidden]), {
      cos: input([t, d]),
      sin: input([t, d]),
      cache: layers.map(() => ({
        k: materializedInput([t, kvHeads * d]),
        v: materializedInput([t, kvHeads * d]),
      })),
      start: 0,
    });
  };

  // HF's Qwen3 attention scales by 1/√headDim after the per-head q and k norms.
  it("scales every layer's scores by 1/√headDim", () => {
    expect(attentionScales(forward())).toEqual(Array(cfg.layers).fill(1 / Math.sqrt(d)));
  });

  // HF's Qwen3 ropes q and k with rotate_half.
  it('ropes q and k of every layer in the half layout', () => {
    expect(ropeLayouts(forward())).toEqual(
      Array(2 * cfg.layers).fill({ layout: 'half', ropesPerHead: 1 }),
    );
  });
});
