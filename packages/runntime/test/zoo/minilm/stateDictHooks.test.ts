import { describe, expect, it } from 'vitest';
import { MINILM_L6 } from '../../../src/zoo/minilm/config.ts';
import { transformMinilmStateDict } from '../../../src/zoo/minilm/stateDictHooks.ts';
import { checkpoint, contents, type Entry } from '../../core/weights/checkpoint.ts';

// The hook reads only hidden.
const CFG = { ...MINILM_L6, hidden: 2, layers: 1 };

/** A one-layer BertModel checkpoint with HF names. Every value is exact in f16. */
const CHECKPOINT: Record<string, Entry> = {
  'embeddings.word_embeddings.weight': { values: [0, 0, 0, 0, 0, 0], shape: [3, 2] },
  'embeddings.position_embeddings.weight': { values: [1, 2, 3, 4], shape: [2, 2] },
  'embeddings.token_type_embeddings.weight': { values: [10, 20, 30, 40], shape: [2, 2] },
  'embeddings.position_ids': { values: [0, 1], shape: [1, 2] },
  'embeddings.LayerNorm.weight': { values: [1, 1], shape: [2] },
  'embeddings.LayerNorm.bias': { values: [0, 0], shape: [2] },
  'encoder.layer.0.attention.self.query.weight': { values: [1, 2, 3, 4], shape: [2, 2] },
  'encoder.layer.0.attention.self.query.bias': { values: [1, 2], shape: [2] },
  'encoder.layer.0.attention.self.key.weight': { values: [5, 6, 7, 8], shape: [2, 2] },
  'encoder.layer.0.attention.self.key.bias': { values: [3, 4], shape: [2] },
  'encoder.layer.0.attention.self.value.weight': { values: [9, 10, 11, 12], shape: [2, 2] },
  'encoder.layer.0.attention.self.value.bias': { values: [5, 6], shape: [2] },
  'encoder.layer.0.attention.output.dense.weight': { values: [0, 0, 0, 0], shape: [2, 2] },
  'encoder.layer.0.attention.output.dense.bias': { values: [0, 0], shape: [2] },
  'encoder.layer.0.attention.output.LayerNorm.weight': { values: [1, 1], shape: [2] },
  'encoder.layer.0.attention.output.LayerNorm.bias': { values: [0, 0], shape: [2] },
  'encoder.layer.0.intermediate.dense.weight': { values: [0, 0, 0, 0], shape: [2, 2] },
  'encoder.layer.0.intermediate.dense.bias': { values: [0, 0], shape: [2] },
  'encoder.layer.0.output.dense.weight': { values: [0, 0, 0, 0], shape: [2, 2] },
  'encoder.layer.0.output.dense.bias': { values: [0, 0], shape: [2] },
  'encoder.layer.0.output.LayerNorm.weight': { values: [1, 1], shape: [2] },
  'encoder.layer.0.output.LayerNorm.bias': { values: [0, 0], shape: [2] },
  'pooler.dense.weight': { values: [0, 0, 0, 0], shape: [2, 2] },
  'pooler.dense.bias': { values: [0, 0], shape: [2] },
};

/** The checkpoint's tensors the model never reads. */
const UNREAD = ['embeddings.position_ids', 'pooler.dense.weight', 'pooler.dense.bias'];

describe('transformMinilmStateDict', () => {
  it.for([
    { dtype: 'F32', elemBytes: 4 },
    { dtype: 'F16', elemBytes: 2 },
  ] as const)('maps a $dtype checkpoint', async ({ dtype, elemBytes }) => {
    const sd = await checkpoint(dtype, CHECKPOINT);
    const total = sd.totalBytes();
    const before = new Map(sd.tensors);
    transformMinilmStateDict(sd, '', CFG);
    const get = (name: string) => sd.tensors.get(name)!;

    // The unread tensors are gone and the token-type table is consumed.
    // Every other tensor stays as it is, value too: query and key load as
    // the checkpoint stores them.
    const untouched = Object.keys(CHECKPOINT).filter(
      (name) =>
        !UNREAD.includes(name) &&
        name !== 'embeddings.token_type_embeddings.weight' &&
        name !== 'embeddings.position_embeddings.weight',
    );
    expect([...sd.tensors.keys()].sort()).toEqual(
      [...untouched, 'embeddings.position_embeddings.weight'].sort(),
    );
    for (const name of untouched) expect(get(name)).toBe(before.get(name));

    // position + tokenType row 0: [1 + 10, 2 + 20, 3 + 10, 4 + 20].
    const position = get('embeddings.position_embeddings.weight');
    expect(position.shape).toEqual([2, 2]);
    expect(position.byteLength).toBe(8 * elemBytes);
    expect(await contents(position)).toEqual({ type: 'f32', values: [11, 22, 13, 24] });

    // Every derived tensor carries its sources' bytes, so the 8 values of the
    // unread tensors are the only bytes lost.
    expect(sd.totalBytes()).toBe(total - 8 * elemBytes);
  });
});
