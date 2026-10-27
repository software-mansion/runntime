import { describe, expect, it } from 'vitest';
import { f16ToF32, f32ToF16Bits } from '../../../src/core/weights/convert.ts';
import {
  bufferSource,
  fromSafetensors,
  memoryStateDict,
  type LazyStateDict,
  type LazyTensor,
} from '../../../src/core/weights/safetensors.ts';
import { MINILM_L6 } from '../../../src/zoo/minilm/config.ts';
import { transformMinilmStateDict } from '../../../src/zoo/minilm/stateDictHooks.ts';

// The hook reads only hidden, headDim and layers. headDim 16 makes the q/k
// scale 16^-0.25 = 0.5, exact in f16.
const CFG = { ...MINILM_L6, hidden: 2, layers: 1, headDim: 16 };

type Entry = { values: number[]; shape: number[] };

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

/** An F16 safetensors file, built from the format spec: an 8-byte
 *  little-endian header length, the JSON header, then the payload. */
function f16File(entries: Record<string, Entry>): Uint8Array {
  const header: Record<string, unknown> = {};
  const halves: number[] = [];
  for (const [name, { values, shape }] of Object.entries(entries)) {
    const begin = halves.length * 2;
    halves.push(...values.map(f32ToF16Bits));
    header[name] = { dtype: 'F16', shape, data_offsets: [begin, halves.length * 2] };
  }
  const json = new TextEncoder().encode(JSON.stringify(header));
  const payload = new Uint8Array(new Uint16Array(halves).buffer);
  const out = new Uint8Array(8 + json.length + payload.length);
  new DataView(out.buffer).setBigUint64(0, BigInt(json.length), true);
  out.set(json, 8);
  out.set(payload, 8 + json.length);
  return out;
}

async function checkpoint(dtype: 'F32' | 'F16'): Promise<LazyStateDict> {
  if (dtype === 'F16') {
    return fromSafetensors(bufferSource(f16File(CHECKPOINT)), { chunkBytes: 0 });
  }
  return memoryStateDict(
    Object.fromEntries(
      Object.entries(CHECKPOINT).map(([name, { values, shape }]) => [
        name,
        { data: new Float32Array(values), shape },
      ]),
    ),
  );
}

/** What a derived tensor holds: f16 halfwords or f32, decoded to numbers. */
async function contents(t: LazyTensor): Promise<{ type: 'f16' | 'f32'; values: number[] }> {
  if (t.kind !== 'derived') throw new Error('expected a derived tensor');
  const data = await t.data();
  if (data instanceof Uint16Array) return { type: 'f16', values: [...f16ToF32(data)] };
  if (data instanceof Float32Array) return { type: 'f32', values: [...data] };
  throw new Error('expected floats');
}

describe('transformMinilmStateDict', () => {
  it.for([
    { dtype: 'F32', type: 'f32', elemBytes: 4 },
    { dtype: 'F16', type: 'f16', elemBytes: 2 },
  ] as const)('maps a $dtype checkpoint', async ({ dtype, type, elemBytes }) => {
    const sd = await checkpoint(dtype);
    const total = sd.totalBytes();
    transformMinilmStateDict(sd, '', CFG);
    const get = (name: string) => sd.tensors.get(name)!;

    // Renamed keys only: pooler and position_ids are gone.
    expect([...sd.tensors.keys()].sort()).toEqual(
      [
        'embeddings.word.weight',
        'embeddings.position.weight',
        'embeddings.norm.weight',
        'embeddings.norm.bias',
        ...['qkv', 'out', 'attnNorm', 'up', 'down', 'ffnNorm'].flatMap((m) => [
          `layers.0.${m}.weight`,
          `layers.0.${m}.bias`,
        ]),
      ].sort(),
    );

    // q|k|v rows in that order, q and k times 0.5, v as is.
    const qkvWeight = get('layers.0.qkv.weight');
    expect(qkvWeight.shape).toEqual([6, 2]);
    expect(qkvWeight.byteLength).toBe(12 * elemBytes);
    expect(await contents(qkvWeight)).toEqual({
      type,
      values: [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 9, 10, 11, 12],
    });
    const qkvBias = get('layers.0.qkv.bias');
    expect(qkvBias.shape).toEqual([6]);
    expect(qkvBias.byteLength).toBe(6 * elemBytes);
    expect(await contents(qkvBias)).toEqual({ type, values: [0.5, 1, 1.5, 2, 5, 6] });

    // position + tokenType row 0: [1 + 10, 2 + 20, 3 + 10, 4 + 20].
    const position = get('embeddings.position.weight');
    expect(position.shape).toEqual([2, 2]);
    expect(position.byteLength).toBe(8 * elemBytes);
    expect(await contents(position)).toEqual({ type, values: [11, 22, 13, 24] });

    // The dropped pooler (6 elems) and position_ids (2) are the only bytes lost.
    expect(sd.totalBytes()).toBe(total - 8 * elemBytes);
  });
});
