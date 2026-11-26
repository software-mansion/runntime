import { describe, expect, it } from 'vitest';
import { MOONSHINE_STREAMING_SMALL } from '../../../src/zoo/moonshine-streaming/config.ts';
import { transformStreamingStateDict } from '../../../src/zoo/moonshine-streaming/stateDictHooks.ts';
import { checkpoint, contents, type Entry } from '../../core/weights/checkpoint.ts';

// The hook reads only the encoder layer count. The hidden sizes differ, as in
// the small model, so the decoder projection is there.
const SMALL = MOONSHINE_STREAMING_SMALL;
const CFG = {
  ...SMALL,
  enc: { ...SMALL.enc, hidden: 2, layers: 1 },
  dec: { ...SMALL.dec, hidden: 3, layers: 1 },
  frameLen: 3,
};

const square = (values: number[]): Entry => ({ values, shape: [2, 2] });
/** A decoder projection: 2 rows of the decoder's hidden size 3. */
const wide = (values: number[]): Entry => ({ values, shape: [2, 3] });
const ZERO = square([0, 0, 0, 0]);

/** Tensors the hook leaves as they are, under their own names. */
const UNTOUCHED = [
  'model.decoder.embed_tokens.weight',
  'model.encoder.embedder.linear.weight',
  'model.encoder.embedder.conv1.weight',
  'model.encoder.embedder.conv1.bias',
  'model.encoder.embedder.conv2.weight',
  'model.encoder.embedder.conv2.bias',
  'model.encoder.layers.0.self_attn.o_proj.weight',
  'model.encoder.layers.0.mlp.fc1.weight',
  'model.encoder.layers.0.mlp.fc1.bias',
  'model.encoder.layers.0.mlp.fc2.weight',
  'model.encoder.layers.0.mlp.fc2.bias',
  'model.decoder.pos_emb.weight',
  'model.decoder.proj.weight',
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
  ...[
    'encoder.layers.0.self_attn',
    'decoder.layers.0.self_attn',
    'decoder.layers.0.encoder_attn',
  ].flatMap((attn) => ['q', 'k', 'v'].map((p) => `model.${attn}.${p}_proj.weight`)),
];

/** A one-layer Moonshine v2 checkpoint with HF names. Every value is exact in
 *  f16. */
const CHECKPOINT: Record<string, Entry> = {
  ...Object.fromEntries(UNTOUCHED.map((name) => [name, ZERO])),
  // The tied LM head, which the model never reads.
  'proj_out.weight': ZERO,
  'model.encoder.embedder.comp.log_k': { values: [1], shape: [1] },
  'model.encoder.layers.0.input_layernorm.gamma': { values: [0.5, -0.25], shape: [2] },
  'model.encoder.layers.0.post_attention_layernorm.gamma': { values: [0, 2], shape: [2] },
  'model.encoder.final_norm.gamma': { values: [-1, 0.125], shape: [2] },
  'model.encoder.layers.0.self_attn.q_proj.weight': square([1, 2, 3, 4]),
  'model.encoder.layers.0.self_attn.k_proj.weight': square([5, 6, 7, 8]),
  'model.encoder.layers.0.self_attn.v_proj.weight': square([9, 10, 11, 12]),
  'model.decoder.layers.0.self_attn.q_proj.weight': wide([-1, -2, -3, -4, -5, -6]),
  'model.decoder.layers.0.self_attn.k_proj.weight': wide([-7, -8, -9, -10, -11, -12]),
  'model.decoder.layers.0.self_attn.v_proj.weight': wide([-13, -14, -15, -16, -17, -18]),
  'model.decoder.layers.0.encoder_attn.q_proj.weight': wide([1, 3, 5, 7, 9, 11]),
  'model.decoder.layers.0.encoder_attn.k_proj.weight': wide([2, 4, 6, 8, 10, 12]),
  'model.decoder.layers.0.encoder_attn.v_proj.weight': wide([10, 20, 30, 40, 50, 60]),
};

describe('transformStreamingStateDict', () => {
  it.for([
    { dtype: 'F32', elemBytes: 4 },
    { dtype: 'F16', elemBytes: 2 },
  ] as const)('maps a $dtype checkpoint', async ({ dtype, elemBytes }) => {
    const sd = await checkpoint(dtype, CHECKPOINT);
    const total = sd.totalBytes();
    const before = new Map(sd.tensors);
    transformStreamingStateDict(sd, '', CFG);
    const get = (name: string) => sd.tensors.get(name)!;

    const derived = [
      'model.encoder.embedder.comp.k',
      'model.encoder.layers.0.input_layernorm.weight',
      'model.encoder.layers.0.post_attention_layernorm.weight',
      'model.encoder.final_norm.weight',
    ];
    expect([...sd.tensors.keys()].sort()).toEqual([...UNTOUCHED, ...derived].sort());
    for (const name of UNTOUCHED) expect(get(name)).toBe(before.get(name));

    // k = exp(log_k) = e, in f32 for any checkpoint.
    const k = get('model.encoder.embedder.comp.k');
    expect(k.shape).toEqual([1]);
    expect(k.byteLength).toBe(elemBytes);
    expect(await contents(k)).toEqual({ type: 'f32', values: [Math.fround(Math.E)] });

    // Norm gammas gain back their −1 offset, in f32 for any checkpoint.
    for (const [name, values] of [
      ['model.encoder.layers.0.input_layernorm.weight', [1.5, 0.75]],
      ['model.encoder.layers.0.post_attention_layernorm.weight', [1, 3]],
      ['model.encoder.final_norm.weight', [0, 1.125]],
    ] as const) {
      const gamma = get(name);
      expect(gamma.shape).toEqual([2]);
      expect(gamma.byteLength).toBe(2 * elemBytes);
      expect(await contents(gamma)).toEqual({ type: 'f32', values });
    }

    // The derived tensors carry their sources' bytes, so the unread
    // proj_out.weight is the only loss.
    expect(sd.totalBytes()).toBe(total - ZERO.values.length * elemBytes);
  });
});
