import { describe, expect, it } from 'vitest';
import {
  renameExportedNormScales,
  transformHfPrivacyFilterStateDict,
} from '../../../src/zoo/privacy-filter/stateDictHooks.ts';
import { EagerModel, HIDDEN } from '../../../src/zoo/privacy-filter/model.ts';
import {
  checkpoint,
  contents,
  safetensorsFile,
  storeAsQuantizedRows,
  type Entry,
} from '../../core/weights/checkpoint.ts';
import { chunksFetched, onceInFileOrder, zeroCheckpoint } from '../chunkedLoad.ts';

const ZERO = { values: [0, 0], shape: [2] };

/** Tensors the hook only renames, HF name → ours. */
const RENAMED: Record<string, string> = {
  'model.layers.0.input_layernorm.weight': 'block.0.attn.norm.weight',
  'model.layers.0.post_attention_layernorm.weight': 'block.0.mlp.norm.weight',
  'model.layers.0.self_attn.o_proj.weight': 'block.0.attn.out.weight',
  'model.layers.0.self_attn.o_proj.bias': 'block.0.attn.out.bias',
  'model.layers.0.mlp.router.weight': 'block.0.mlp.gate.weight',
  'model.layers.0.mlp.router.bias': 'block.0.mlp.gate.bias',
  'model.layers.0.self_attn.sinks': 'block.0.attn.sinks',
  'model.layers.0.mlp.experts.down_proj_bias': 'block.0.mlp.outBias',
  'model.embed_tokens.weight': 'embedding.weight',
  'model.norm.weight': 'norm.weight',
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
    if (source.kind === 'quant' || outWeight.kind === 'quant') {
      throw new Error('expected float tensors');
    }
    expect([outWeight.kind, outWeight.dtype]).toEqual([source.kind, source.dtype]);
    expect([...(await outWeight.f32())]).toEqual([1, 2, 3, 4]);

    // The dropped score.bias (2 elems) is the only byte loss.
    expect(sd.totalBytes()).toBe(total - 2 * elemBytes);
  });

  // A file stores a quantized [E, K, 2H] as rows [E, K·2H], which the hook
  // cannot split into glu and lin halves per expert.
  it.for([
    'model.layers.0.mlp.experts.gate_up_proj',
    'model.layers.0.mlp.experts.gate_up_proj_bias',
    'model.layers.0.mlp.experts.down_proj',
  ])('refuses a quantized %s with CHECKPOINT_MISMATCH', async (name) => {
    const sd = await checkpoint('F32', CHECKPOINT);
    await storeAsQuantizedRows(sd, name, CHECKPOINT[name]!.values, 2);
    expect(() => transformHfPrivacyFilterStateDict(sd)).toThrow(
      expect.objectContaining({
        code: 'CHECKPOINT_MISMATCH',
        message:
          `privacy-filter state dict: '${name}' is quantized, and a quantized expert ` +
          'tensor cannot be split into experts: keep it float in the file',
      }),
    );
  });
});

describe('the official checkpoint', () => {
  // The hook renames, stacks and splits every tensor before the load. Each
  // must still bind where the file holds it, or the loader fetches an
  // evicted chunk again.
  it('lets the model fetch each chunk once', async () => {
    const [blocks, experts, vocab, logits] = [2, 2, 8, 4];
    const model = () => new EagerModel({ blocks, experts, vocab, logits, weightFormat: 'f32' });
    // q is 14 heads of 64, k and v 2 heads each.
    const [q, kv] = [896, 128];
    const shapes: Record<string, number[]> = {
      'model.embed_tokens.weight': [vocab, HIDDEN],
      'model.norm.weight': [HIDDEN],
      'score.weight': [logits, HIDDEN],
      'score.bias': [logits],
    };
    for (let n = 0; n < blocks; n++) {
      const layer = `model.layers.${n}`;
      Object.assign(shapes, {
        [`${layer}.input_layernorm.weight`]: [HIDDEN],
        [`${layer}.post_attention_layernorm.weight`]: [HIDDEN],
        [`${layer}.self_attn.q_proj.weight`]: [q, HIDDEN],
        [`${layer}.self_attn.q_proj.bias`]: [q],
        [`${layer}.self_attn.k_proj.weight`]: [kv, HIDDEN],
        [`${layer}.self_attn.k_proj.bias`]: [kv],
        [`${layer}.self_attn.v_proj.weight`]: [kv, HIDDEN],
        [`${layer}.self_attn.v_proj.bias`]: [kv],
        [`${layer}.self_attn.o_proj.weight`]: [HIDDEN, q],
        [`${layer}.self_attn.o_proj.bias`]: [HIDDEN],
        [`${layer}.self_attn.sinks`]: [14],
        [`${layer}.mlp.router.weight`]: [experts, HIDDEN],
        [`${layer}.mlp.router.bias`]: [experts],
        [`${layer}.mlp.experts.gate_up_proj`]: [experts, HIDDEN, 2 * HIDDEN],
        [`${layer}.mlp.experts.gate_up_proj_bias`]: [experts, 2 * HIDDEN],
        [`${layer}.mlp.experts.down_proj`]: [experts, HIDDEN, HIDDEN],
        [`${layer}.mlp.experts.down_proj_bias`]: [experts, HIDDEN],
      });
    }
    const file = safetensorsFile(
      Object.fromEntries(
        Object.keys(shapes)
          .sort()
          .map((name) => {
            const shape = shapes[name]!;
            return [name, { data: new Float32Array(shape.reduce((a, b) => a * b, 1)), shape }];
          }),
      ),
    );
    // A block is about 15 MB, so the file spans far more chunks than the 4
    // the source keeps. A chunk holds o_proj.weight, which sits between the
    // k and q biases that one tensor reads.
    const chunks = await chunksFetched(model(), file, 2 ** 21, transformHfPrivacyFilterStateDict);
    expect(chunks.length).toBeGreaterThan(12);
    // Once each, but not in order: the stacked q, k and v bias reads the v
    // bias, a chunk past the q weight that binds next.
    expect([...chunks].sort((a, b) => a - b)).toEqual(onceInFileOrder(chunks));
  });
});

describe('renameExportedNormScales', () => {
  it('renames every RMSNorm scale of an exported file to weight', async () => {
    const sd = await checkpoint('F32', {
      'block.0.attn.norm.scale': ZERO,
      'block.0.mlp.norm.scale': ZERO,
      'norm.scale': ZERO,
      'block.0.attn.qkv.weight': ZERO,
    });
    const before = new Map(sd.tensors);
    renameExportedNormScales(sd);
    expect([...sd.tensors.keys()].sort()).toEqual([
      'block.0.attn.norm.weight',
      'block.0.attn.qkv.weight',
      'block.0.mlp.norm.weight',
      'norm.weight',
    ]);
    expect(sd.tensors.get('norm.weight')).toBe(before.get('norm.scale'));
  });

  // The norms sit all over an exported file. Each must bind where the file
  // holds its scale, or the loader fetches an evicted chunk again.
  it('lets the model fetch each chunk of an exported file once, in file order', async () => {
    const model = () =>
      new EagerModel({ blocks: 2, experts: 2, vocab: 8, logits: 4, weightFormat: 'f32' });
    const norms = [...model().namedParameters()]
      .map(([name]) => name)
      .filter((name) => name.endsWith('norm.weight'));
    const scales = norms.map((name) => `${name.slice(0, -'weight'.length)}scale`);
    const file = zeroCheckpoint(
      model(),
      norms,
      Object.fromEntries(scales.map((name) => [name, [HIDDEN]])),
    );
    // The file is about 30 MB, far more chunks than the 4 the source keeps.
    const chunks = await chunksFetched(model(), file, 2 ** 20, renameExportedNormScales);
    expect(chunks.length).toBeGreaterThan(12);
    expect(chunks).toEqual(onceInFileOrder(chunks));
  });
});
