import { describe, expect, it } from 'vitest';
import { MOONSHINE_TINY } from '../../../src/zoo/moonshine/config.ts';
import { transformMoonshineStateDict } from '../../../src/zoo/moonshine/stateDictHooks.ts';
import { checkpoint, contents, type Entry } from '../../core/weights/checkpoint.ts';

// The hook reads only encLayers, decLayers and headDim. headDim 4 makes the
// cross-attention q scale 1/√4 = 0.5, exact in f16.
const CFG = { ...MOONSHINE_TINY, encLayers: 1, decLayers: 1, headDim: 4 };

const square = (values: number[]): Entry => ({ values, shape: [2, 2] });
const ZERO = square([0, 0, 0, 0]);

/** Tensors the hook only renames, HF name → ours. */
const RENAMED: Record<string, string> = {
  'model.decoder.embed_tokens.weight': 'embedding.weight',
  'model.encoder.conv1.weight': 'stem.conv1.weight',
  'model.encoder.groupnorm.weight': 'stem.groupnorm.weight',
  'model.encoder.groupnorm.bias': 'stem.groupnorm.bias',
  'model.encoder.conv2.weight': 'stem.conv2.weight',
  'model.encoder.conv2.bias': 'stem.conv2.bias',
  'model.encoder.conv3.weight': 'stem.conv3.weight',
  'model.encoder.conv3.bias': 'stem.conv3.bias',
  'model.encoder.layers.0.input_layernorm.weight': 'encoder.layers.0.norm1.weight',
  'model.encoder.layers.0.self_attn.o_proj.weight': 'encoder.layers.0.attn.out.weight',
  'model.encoder.layers.0.post_attention_layernorm.weight': 'encoder.layers.0.norm2.weight',
  'model.encoder.layers.0.mlp.fc1.weight': 'encoder.layers.0.mlp.fc1.weight',
  'model.encoder.layers.0.mlp.fc1.bias': 'encoder.layers.0.mlp.fc1.bias',
  'model.encoder.layers.0.mlp.fc2.weight': 'encoder.layers.0.mlp.fc2.weight',
  'model.encoder.layers.0.mlp.fc2.bias': 'encoder.layers.0.mlp.fc2.bias',
  'model.encoder.layer_norm.weight': 'encoder.norm.weight',
  'model.decoder.layers.0.input_layernorm.weight': 'decoder.layers.0.norm1.weight',
  'model.decoder.layers.0.self_attn.o_proj.weight': 'decoder.layers.0.selfAttn.out.weight',
  'model.decoder.layers.0.post_attention_layernorm.weight': 'decoder.layers.0.norm2.weight',
  'model.decoder.layers.0.encoder_attn.o_proj.weight': 'decoder.layers.0.crossAttn.out.weight',
  'model.decoder.layers.0.final_layernorm.weight': 'decoder.layers.0.norm3.weight',
  'model.decoder.layers.0.mlp.fc1.weight': 'decoder.layers.0.mlp.fc1.weight',
  'model.decoder.layers.0.mlp.fc1.bias': 'decoder.layers.0.mlp.fc1.bias',
  'model.decoder.layers.0.mlp.fc2.weight': 'decoder.layers.0.mlp.fc2.weight',
  'model.decoder.layers.0.mlp.fc2.bias': 'decoder.layers.0.mlp.fc2.bias',
  'model.decoder.norm.weight': 'decoder.norm.weight',
};

/** A one-layer Moonshine checkpoint with HF names. Every value is exact in f16. */
const CHECKPOINT: Record<string, Entry> = {
  ...Object.fromEntries(Object.keys(RENAMED).map((name) => [name, ZERO])),
  'proj_out.weight': ZERO,
  'model.encoder.layers.0.self_attn.q_proj.weight': square([1, 2, 3, 4]),
  'model.encoder.layers.0.self_attn.k_proj.weight': square([5, 6, 7, 8]),
  'model.encoder.layers.0.self_attn.v_proj.weight': square([9, 10, 11, 12]),
  'model.decoder.layers.0.self_attn.q_proj.weight': square([-1, -2, -3, -4]),
  'model.decoder.layers.0.self_attn.k_proj.weight': square([-5, -6, -7, -8]),
  'model.decoder.layers.0.self_attn.v_proj.weight': square([-9, -10, -11, -12]),
  'model.decoder.layers.0.encoder_attn.q_proj.weight': square([1, 3, 5, 7]),
  'model.decoder.layers.0.encoder_attn.k_proj.weight': square([2, 4, 6, 8]),
  'model.decoder.layers.0.encoder_attn.v_proj.weight': square([10, 20, 30, 40]),
};

describe('transformMoonshineStateDict', () => {
  it.for([
    { dtype: 'F32', type: 'f32', elemBytes: 4 },
    { dtype: 'F16', type: 'f16', elemBytes: 2 },
  ] as const)('maps a $dtype checkpoint', async ({ dtype, type, elemBytes }) => {
    const sd = await checkpoint(dtype, CHECKPOINT);
    const total = sd.totalBytes();
    const before = new Map(sd.tensors);
    transformMoonshineStateDict(sd, '', CFG);
    const get = (name: string) => sd.tensors.get(name)!;

    const fused = [
      'encoder.layers.0.attn.qkv.weight',
      'decoder.layers.0.selfAttn.qkv.weight',
      'decoder.layers.0.crossAttn.q.weight',
      'decoder.layers.0.crossAttn.kv.weight',
    ];
    // proj_out is tied to the embedding and dropped.
    expect([...sd.tensors.keys()].sort()).toEqual([...Object.values(RENAMED), ...fused].sort());
    for (const [hf, ours] of Object.entries(RENAMED)) expect(get(ours)).toBe(before.get(hf));

    // Self-attention q|k|v rows in that order, unscaled: rope carries the scale.
    const encQkv = get('encoder.layers.0.attn.qkv.weight');
    expect(encQkv.shape).toEqual([6, 2]);
    expect(encQkv.byteLength).toBe(12 * elemBytes);
    expect(await contents(encQkv)).toEqual({
      type,
      values: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    });
    const decQkv = get('decoder.layers.0.selfAttn.qkv.weight');
    expect(decQkv.shape).toEqual([6, 2]);
    expect(decQkv.byteLength).toBe(12 * elemBytes);
    expect(await contents(decQkv)).toEqual({
      type,
      values: [-1, -2, -3, -4, -5, -6, -7, -8, -9, -10, -11, -12],
    });

    // Cross-attention q times 1/√headDim = 0.5; k|v stacked as is.
    const crossQ = get('decoder.layers.0.crossAttn.q.weight');
    expect(crossQ.shape).toEqual([2, 2]);
    expect(crossQ.byteLength).toBe(4 * elemBytes);
    expect(await contents(crossQ)).toEqual({ type, values: [0.5, 1.5, 2.5, 3.5] });
    const crossKv = get('decoder.layers.0.crossAttn.kv.weight');
    expect(crossKv.shape).toEqual([4, 2]);
    expect(crossKv.byteLength).toBe(8 * elemBytes);
    expect(await contents(crossKv)).toEqual({ type, values: [2, 4, 6, 8, 10, 20, 30, 40] });

    // The dropped proj_out (4 elems) is the only byte loss.
    expect(sd.totalBytes()).toBe(total - 4 * elemBytes);
  });
});
