import { describe, expect, it } from 'vitest';
import { MOONSHINE_STREAMING_SMALL } from '../../../src/zoo/moonshine-streaming/config.ts';
import { transformStreamingStateDict } from '../../../src/zoo/moonshine-streaming/stateDictHooks.ts';
import { checkpoint, contents, type Entry } from '../../core/weights/checkpoint.ts';

// The hook reads only the layer counts, head dims and hidden sizes. Encoder
// headDim 4 makes the q scale 1/√4 = 0.5, exact in f16. The hidden sizes
// differ, as in the small model, so the decoder projection is there.
const SMALL = MOONSHINE_STREAMING_SMALL;
const CFG = {
  ...SMALL,
  enc: { ...SMALL.enc, hidden: 2, layers: 1, headDim: 4 },
  dec: { ...SMALL.dec, hidden: 3, layers: 1, headDim: 4 },
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
  'proj_out.weight',
];

/** A one-layer Moonshine v2 checkpoint with HF names. Every value is exact in
 *  f16. */
const CHECKPOINT: Record<string, Entry> = {
  ...Object.fromEntries(UNTOUCHED.map((name) => [name, ZERO])),
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
    { dtype: 'F32', type: 'f32', elemBytes: 4 },
    { dtype: 'F16', type: 'f16', elemBytes: 2 },
  ] as const)('maps a $dtype checkpoint', async ({ dtype, type, elemBytes }) => {
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
      'model.encoder.layers.0.self_attn.qkv.weight',
      'model.decoder.layers.0.self_attn.qkv.weight',
      'model.decoder.layers.0.encoder_attn.q_proj.weight',
      'model.decoder.layers.0.encoder_attn.kv.weight',
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

    // Encoder q|k|v with q times 1/√headDim = 0.5: the encoder has no rope.
    // The scaled part makes it f32 on every checkpoint.
    const encQkv = get('model.encoder.layers.0.self_attn.qkv.weight');
    expect(encQkv.shape).toEqual([6, 2]);
    expect(encQkv.byteLength).toBe(12 * elemBytes);
    expect(await contents(encQkv)).toEqual({
      type: 'f32',
      values: [0.5, 1, 1.5, 2, 5, 6, 7, 8, 9, 10, 11, 12],
    });

    // Decoder q|k|v unscaled: its rope tables carry the scale.
    const decQkv = get('model.decoder.layers.0.self_attn.qkv.weight');
    expect(decQkv.shape).toEqual([6, 3]);
    expect(decQkv.byteLength).toBe(18 * elemBytes);
    expect(await contents(decQkv)).toEqual({
      type,
      values: Array.from({ length: 18 }, (_, i) => -(i + 1)),
    });

    // Cross-attention q times 1/√headDim = 0.5, f32 on every checkpoint;
    // k|v stacked as is.
    const crossQ = get('model.decoder.layers.0.encoder_attn.q_proj.weight');
    expect(crossQ.shape).toEqual([2, 3]);
    expect(crossQ.byteLength).toBe(6 * elemBytes);
    expect(await contents(crossQ)).toEqual({ type: 'f32', values: [0.5, 1.5, 2.5, 3.5, 4.5, 5.5] });
    const crossKv = get('model.decoder.layers.0.encoder_attn.kv.weight');
    expect(crossKv.shape).toEqual([4, 3]);
    expect(crossKv.byteLength).toBe(12 * elemBytes);
    expect(await contents(crossKv)).toEqual({
      type,
      values: [2, 4, 6, 8, 10, 12, 10, 20, 30, 40, 50, 60],
    });

    // Nothing is dropped: every derived tensor carries its source's bytes.
    expect(sd.totalBytes()).toBe(total);
  });
});
