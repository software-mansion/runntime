import { describe, expect, it } from 'vitest';
import { transformHfPrivacyFilterStateDict } from '../../../src/zoo/privacy-filter/stateDictHooks.ts';
import { checkpoint, contents, type Entry } from '../../core/weights/checkpoint.ts';

const ZERO = { values: [0, 0], shape: [2] };

/** Tensors the hook only renames, HF name → ours. */
const RENAMED: Record<string, string> = {
  'model.layers.0.input_layernorm.weight': 'block.0.attn.norm.scale',
  'model.layers.0.post_attention_layernorm.weight': 'block.0.mlp.norm.scale',
  'model.layers.0.self_attn.o_proj.weight': 'block.0.attn.out.weight',
  'model.layers.0.self_attn.o_proj.bias': 'block.0.attn.out.bias',
  'model.layers.0.mlp.router.weight': 'block.0.mlp.gate.weight',
  'model.layers.0.mlp.router.bias': 'block.0.mlp.gate.bias',
  'model.layers.0.self_attn.sinks': 'block.0.attn.sinks',
  'model.layers.0.mlp.experts.down_proj_bias': 'block.0.mlp.outBias',
  'model.embed_tokens.weight': 'embedding.weight',
  'model.norm.weight': 'norm.scale',
  'score.weight': 'unembedding.weight',
};

/** A one-layer checkpoint in the transformers layout, with 2 experts, hidden
 *  1 and expert width 2. Every value is exact in f16. */
const CHECKPOINT: Record<string, Entry> = {
  ...Object.fromEntries(Object.keys(RENAMED).map((name) => [name, ZERO])),
  'score.bias': ZERO,
  // Grouped-query shapes: one q row pair, one k row, one v row.
  'model.layers.0.self_attn.q_proj.weight': { values: [1, 2, 3, 4], shape: [2, 2] },
  'model.layers.0.self_attn.k_proj.weight': { values: [5, 6], shape: [1, 2] },
  'model.layers.0.self_attn.v_proj.weight': { values: [7, 8], shape: [1, 2] },
  'model.layers.0.self_attn.q_proj.bias': { values: [1, 2], shape: [2] },
  'model.layers.0.self_attn.k_proj.bias': { values: [3], shape: [1] },
  'model.layers.0.self_attn.v_proj.bias': { values: [4], shape: [1] },
  // [E, K, 2H] = [2, 1, 4]: expert 0 is 1 2 | 3 4, expert 1 is 5 6 | 7 8.
  'model.layers.0.mlp.experts.gate_up_proj': {
    values: [1, 2, 3, 4, 5, 6, 7, 8],
    shape: [2, 1, 4],
  },
  // [E, 2H] = [2, 4]: expert 0 is 10 11 | 12 13, expert 1 is 14 15 | 16 17.
  'model.layers.0.mlp.experts.gate_up_proj_bias': {
    values: [10, 11, 12, 13, 14, 15, 16, 17],
    shape: [2, 4],
  },
  // [E, H, K] = [2, 2, 1].
  'model.layers.0.mlp.experts.down_proj': { values: [1, 2, 3, 4], shape: [2, 2, 1] },
};

describe('transformHfPrivacyFilterStateDict', () => {
  it.for([
    { dtype: 'F32', type: 'f32', elemBytes: 4 },
    { dtype: 'F16', type: 'f16', elemBytes: 2 },
  ] as const)('maps a $dtype checkpoint', async ({ dtype, type, elemBytes }) => {
    const sd = await checkpoint(dtype, CHECKPOINT);
    const total = sd.totalBytes();
    const before = new Map(sd.tensors);
    transformHfPrivacyFilterStateDict(sd);
    const get = (name: string) => sd.tensors.get(name)!;

    const mapped = [
      'block.0.attn.qkv.weight',
      'block.0.attn.qkv.bias',
      'block.0.mlp.gluWeight',
      'block.0.mlp.linWeight',
      'block.0.mlp.gluBias',
      'block.0.mlp.linBias',
      'block.0.mlp.outWeight',
    ];
    // score.bias is dropped.
    expect([...sd.tensors.keys()].sort()).toEqual([...Object.values(RENAMED), ...mapped].sort());
    for (const [hf, ours] of Object.entries(RENAMED)) expect(get(ours)).toBe(before.get(hf));
    expect(sd.metadata['model']).toBe('privacy-filter');

    // q|k|v rows in that order.
    const qkvWeight = get('block.0.attn.qkv.weight');
    expect(qkvWeight.shape).toEqual([4, 2]);
    expect(qkvWeight.byteLength).toBe(8 * elemBytes);
    expect(await contents(qkvWeight)).toEqual({ type, values: [1, 2, 3, 4, 5, 6, 7, 8] });
    const qkvBias = get('block.0.attn.qkv.bias');
    expect(qkvBias.shape).toEqual([4]);
    expect(qkvBias.byteLength).toBe(4 * elemBytes);
    expect(await contents(qkvBias)).toEqual({ type, values: [1, 2, 3, 4] });

    // Each expert's first half of columns is glu, the second lin, with the
    // expert axis folded into rows.
    for (const [name, shape, values] of [
      ['block.0.mlp.gluWeight', [2, 2], [1, 2, 5, 6]],
      ['block.0.mlp.linWeight', [2, 2], [3, 4, 7, 8]],
      ['block.0.mlp.gluBias', [2, 2], [10, 11, 14, 15]],
      ['block.0.mlp.linBias', [2, 2], [12, 13, 16, 17]],
    ] as const) {
      const half = get(name);
      expect(half.shape).toEqual(shape);
      expect(half.byteLength).toBe(4 * elemBytes);
      expect(await contents(half)).toEqual({ type, values });
    }

    // down_proj only folds [2, 2, 1] → [4, 1]: the source tensor, reshaped.
    const source = before.get('model.layers.0.mlp.experts.down_proj')!;
    const outWeight = get('block.0.mlp.outWeight');
    expect(outWeight.shape).toEqual([4, 1]);
    expect(outWeight.byteLength).toBe(4 * elemBytes);
    expect([outWeight.kind, outWeight.dtype]).toEqual([source.kind, source.dtype]);
    expect([...(await outWeight.f32())]).toEqual([1, 2, 3, 4]);

    // The dropped score.bias (2 elems) is the only byte loss.
    expect(sd.totalBytes()).toBe(total - 2 * elemBytes);
  });
});
