import { describe, expect, it } from 'vitest';
import { transformMoonshineStateDict } from '../../../src/zoo/moonshine/stateDictHooks.ts';
import { checkpoint, type Entry } from '../../core/weights/checkpoint.ts';

const square = (values: number[]): Entry => ({ values, shape: [2, 2] });
const ZERO = square([0, 0, 0, 0]);

/** Tensors the hook leaves as they are, under their own names. */
const UNTOUCHED = [
  'model.decoder.embed_tokens.weight',
  'model.encoder.conv1.weight',
  'model.encoder.groupnorm.weight',
  'model.encoder.groupnorm.bias',
  'model.encoder.conv2.weight',
  'model.encoder.conv2.bias',
  'model.encoder.conv3.weight',
  'model.encoder.conv3.bias',
  'model.encoder.layers.0.input_layernorm.weight',
  'model.encoder.layers.0.self_attn.o_proj.weight',
  'model.encoder.layers.0.post_attention_layernorm.weight',
  'model.encoder.layers.0.mlp.fc1.weight',
  'model.encoder.layers.0.mlp.fc1.bias',
  'model.encoder.layers.0.mlp.fc2.weight',
  'model.encoder.layers.0.mlp.fc2.bias',
  'model.encoder.layer_norm.weight',
  'model.decoder.layers.0.input_layernorm.weight',
  'model.decoder.layers.0.self_attn.o_proj.weight',
  'model.decoder.layers.0.post_attention_layernorm.weight',
  'model.decoder.layers.0.encoder_attn.o_proj.weight',
  'model.decoder.layers.0.final_layernorm.weight',
  'model.decoder.layers.0.mlp.fc1.weight',
  'model.decoder.layers.0.mlp.fc1.bias',
  'model.decoder.layers.0.mlp.fc2.weight',
  'model.decoder.layers.0.mlp.fc2.bias',
  'model.decoder.norm.weight',
  ...['encoder.layers.0.self_attn', 'decoder.layers.0.self_attn'].flatMap((attn) =>
    ['q', 'k', 'v'].map((p) => `model.${attn}.${p}_proj.weight`),
  ),
  ...['q', 'k', 'v'].map((p) => `model.decoder.layers.0.encoder_attn.${p}_proj.weight`),
];

/** A one-layer Moonshine checkpoint with HF names. Every value is exact in f16. */
const CHECKPOINT: Record<string, Entry> = {
  ...Object.fromEntries(UNTOUCHED.map((name) => [name, ZERO])),
  // The tied LM head, which the model never reads.
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
    { dtype: 'F32', elemBytes: 4 },
    { dtype: 'F16', elemBytes: 2 },
  ] as const)('maps a $dtype checkpoint', async ({ dtype, elemBytes }) => {
    const sd = await checkpoint(dtype, CHECKPOINT);
    const total = sd.totalBytes();
    const before = new Map(sd.tensors);
    transformMoonshineStateDict(sd, '');

    // Every attention projection stays as the checkpoint stores it: the
    // attention applies the score scale, and the model's FusedLinear stacks
    // the projections.
    expect([...sd.tensors.keys()].sort()).toEqual([...UNTOUCHED].sort());
    for (const name of UNTOUCHED) expect(sd.tensors.get(name)).toBe(before.get(name));

    // The unread proj_out.weight is the only loss.
    expect(sd.totalBytes()).toBe(total - ZERO.values.length * elemBytes);
  });
});
