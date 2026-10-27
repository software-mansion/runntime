import { describe, expect, it } from 'vitest';
import { MOONSHINE_STREAMING_SMALL } from '../../../src/zoo/moonshine-streaming/config.ts';
import { transformStreamingStateDict } from '../../../src/zoo/moonshine-streaming/stateDictHooks.ts';
import { checkpoint, contents, type Entry } from '../../core/weights/checkpoint.ts';

// The hook reads only frameLen and the layer counts, head dims and hidden
// sizes. Encoder headDim 4 makes the q scale 1/√4 = 0.5, exact in f16. The
// hidden sizes differ, as in the small model, so the adapter projection maps.
const SMALL = MOONSHINE_STREAMING_SMALL;
const CFG = {
  ...SMALL,
  enc: { ...SMALL.enc, hidden: 2, layers: 1, headDim: 4 },
  dec: { ...SMALL.dec, hidden: 3, layers: 1, headDim: 4 },
  frameLen: 3,
};

const square = (values: number[]): Entry => ({ values, shape: [2, 2] });
const ZERO = square([0, 0, 0, 0]);

/** Tensors the hook only renames, HF name → ours. */
const RENAMED: Record<string, string> = {
  'model.decoder.embed_tokens.weight': 'embedding.weight',
  'model.encoder.embedder.linear.weight': 'frontend.linear.weight',
  'model.encoder.embedder.conv1.weight': 'frontend.conv1.weight',
  'model.encoder.embedder.conv1.bias': 'frontend.conv1.bias',
  'model.encoder.embedder.conv2.weight': 'frontend.conv2.weight',
  'model.encoder.embedder.conv2.bias': 'frontend.conv2.bias',
  'model.encoder.layers.0.self_attn.o_proj.weight': 'encoder.layers.0.attn.out.weight',
  'model.encoder.layers.0.mlp.fc1.weight': 'encoder.layers.0.mlp.fc1.weight',
  'model.encoder.layers.0.mlp.fc1.bias': 'encoder.layers.0.mlp.fc1.bias',
  'model.encoder.layers.0.mlp.fc2.weight': 'encoder.layers.0.mlp.fc2.weight',
  'model.encoder.layers.0.mlp.fc2.bias': 'encoder.layers.0.mlp.fc2.bias',
  'model.decoder.pos_emb.weight': 'adapter.posEmb',
  'model.decoder.proj.weight': 'adapter.proj.weight',
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

/** A one-layer Moonshine v2 checkpoint with HF names. Every value is exact in
 *  f16. */
const CHECKPOINT: Record<string, Entry> = {
  ...Object.fromEntries(Object.keys(RENAMED).map((name) => [name, ZERO])),
  'proj_out.weight': ZERO,
  'model.encoder.embedder.comp.log_k': { values: [1], shape: [1] },
  'model.encoder.layers.0.input_layernorm.gamma': { values: [0.5, -0.25], shape: [2] },
  'model.encoder.layers.0.post_attention_layernorm.gamma': { values: [0, 2], shape: [2] },
  'model.encoder.final_norm.gamma': { values: [-1, 0.125], shape: [2] },
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
      'frontend.cmvn.weight',
      'frontend.k',
      'encoder.layers.0.norm1.weight',
      'encoder.layers.0.norm2.weight',
      'encoder.norm.weight',
      'encoder.layers.0.attn.qkv.weight',
      'decoder.layers.0.selfAttn.qkv.weight',
      'decoder.layers.0.crossAttn.q.weight',
      'decoder.layers.0.crossAttn.kv.weight',
    ];
    // proj_out is tied to the embedding and dropped.
    expect([...sd.tensors.keys()].sort()).toEqual([...Object.values(RENAMED), ...derived].sort());
    for (const [hf, ours] of Object.entries(RENAMED)) expect(get(ours)).toBe(before.get(hf));

    // CMVN has no learned scale: ones, with no file bytes behind them.
    const cmvn = get('frontend.cmvn.weight');
    expect(cmvn.shape).toEqual([3]);
    expect(cmvn.byteLength).toBe(0);
    expect(await contents(cmvn)).toEqual({ type: 'f32', values: [1, 1, 1] });

    // k = exp(log_k) = e, in f32 for any checkpoint.
    const k = get('frontend.k');
    expect(k.shape).toEqual([1]);
    expect(k.byteLength).toBe(elemBytes);
    expect(await contents(k)).toEqual({ type: 'f32', values: [Math.fround(Math.E)] });

    // Norm gammas gain back their −1 offset, in f32 for any checkpoint.
    for (const [name, values] of [
      ['encoder.layers.0.norm1.weight', [1.5, 0.75]],
      ['encoder.layers.0.norm2.weight', [1, 3]],
      ['encoder.norm.weight', [0, 1.125]],
    ] as const) {
      const gamma = get(name);
      expect(gamma.shape).toEqual([2]);
      expect(gamma.byteLength).toBe(2 * elemBytes);
      expect(await contents(gamma)).toEqual({ type: 'f32', values });
    }

    // Encoder q|k|v with q times 1/√headDim = 0.5: the encoder has no rope.
    const encQkv = get('encoder.layers.0.attn.qkv.weight');
    expect(encQkv.shape).toEqual([6, 2]);
    expect(encQkv.byteLength).toBe(12 * elemBytes);
    expect(await contents(encQkv)).toEqual({
      type,
      values: [0.5, 1, 1.5, 2, 5, 6, 7, 8, 9, 10, 11, 12],
    });

    // Decoder q|k|v unscaled: its rope tables carry the scale.
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
