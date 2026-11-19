import { describe, expect, it } from 'vitest';
import { pending, type ValueMeta } from '../../../src/core/graph/value.ts';
import {
  bufferSource,
  fromSafetensors,
  type LazyStateDict,
} from '../../../src/core/weights/safetensors.ts';
import type { ReverbConfig } from '../../../src/zoo/reverb/config.ts';
import { ReverbModel } from '../../../src/zoo/reverb/model.ts';
import { transformReverbStateDict } from '../../../src/zoo/reverb/stateDictHooks.ts';
import {
  checkpoint,
  contents,
  quantContents,
  quantFile,
  quantizeWeight,
  type Entry,
  type QuantEntry,
} from '../../core/weights/checkpoint.ts';

// One layer: hidden 4, 2 q heads and 1 kv head of size 4, so d^-½ = 0.5 is
// exact in f16. Three codebooks of three codes: two coarse, one fine.
const CFG: ReverbConfig = {
  layers: 1,
  hidden: 4,
  ffn: 2,
  heads: 2,
  kvHeads: 1,
  headDim: 4,
  eps: 1e-6,
  ropeTheta: 1e6,
  vocab: 5,
  embeddingRank: 2,
  codebooks: 3,
  codes: 3,
};

const DTYPES = [
  { dtype: 'F32', type: 'f32' },
  { dtype: 'F16', type: 'f16' },
] as const;

/** `rows` rows of `cols`: row r holds base + 10r + c. Exact in f16. */
const grid = (base: number, rows: number, cols: number): Entry => ({
  values: Array.from(
    { length: rows * cols },
    (_, i) => base + 10 * Math.floor(i / cols) + (i % cols),
  ),
  shape: [rows, cols],
});

/** The rows of `grid(base, …, cols)` listed in `order`. */
const gridRows = (base: number, order: number[], cols: number) =>
  order.flatMap((r) => Array.from({ length: cols }, (_, c) => base + 10 * r + c));

const vector = (values: number[]): Entry => ({ values, shape: [values.length] });

const ramp = (start: number, n: number) => vector(Array.from({ length: n }, (_, i) => start + i));

/** A release checkpoint of `cfg`'s shapes, under the release names. */
function release(cfg = CFG): Record<string, Entry> {
  const { hidden, ffn, heads, kvHeads, headDim: d, vocab, embeddingRank, codebooks, codes } = cfg;
  const S = 'llm.layers.0';
  return {
    codebook_layer_offsets: vector(Array.from({ length: codebooks }, (_, b) => b * codes)),
    'audio_embeddings.weight': grid(0, codebooks * codes, hidden),
    'audio_heads.weight': grid(0, codebooks * codes, hidden),
    'llm.embed_tokens.table.weight': grid(0, vocab, embeddingRank),
    'llm.embed_tokens.project.weight': grid(0, hidden, embeddingRank),
    'llm.norm.weight': vector(Array<number>(hidden).fill(1)),
    [`${S}.input_layernorm.weight`]: ramp(1, hidden),
    [`${S}.post_attention_layernorm.weight`]: ramp(5, hidden),
    [`${S}.self_attn.q_proj.weight`]: grid(100, heads * d, hidden),
    [`${S}.self_attn.k_proj.weight`]: grid(200, kvHeads * d, hidden),
    [`${S}.self_attn.v_proj.weight`]: grid(300, kvHeads * d, hidden),
    [`${S}.self_attn.o_proj.weight`]: grid(400, hidden, heads * d),
    [`${S}.self_attn.q_norm.weight`]: ramp(1, d),
    [`${S}.self_attn.k_norm.weight`]: ramp(5, d),
    [`${S}.mlp.up_proj.weight`]: grid(500, ffn, hidden),
    [`${S}.mlp.gate_proj.weight`]: grid(600, ffn, hidden),
    [`${S}.mlp.down_proj.weight`]: grid(700, hidden, ffn),
  };
}

/** `release(cfg)` as an int file: the Linear weights under llm.layers
 *  quantized, the rest float. */
function quantRelease(bits: 4 | 8, groupSize: number, cfg = CFG) {
  const quant: Record<string, QuantEntry> = {};
  const floats: Record<string, { data: Float32Array; shape: number[] }> = {};
  for (const [name, { values, shape }] of Object.entries(release(cfg))) {
    const data = Float32Array.from(values);
    if (name.startsWith('llm.layers.') && name.endsWith('_proj.weight')) {
      quant[name] = quantizeWeight(data, shape as [number, number], bits, groupSize);
    } else floats[name] = { data, shape };
  }
  const file = quantFile(bits, groupSize, quant, floats);
  return { quant, sd: () => fromSafetensors(bufferSource(file), { chunkBytes: 0 }) };
}

/** Tensors the hook only renames, release name → model name. */
const RENAMED = {
  'audio_embeddings.weight': 'audio_table.weight',
  'llm.embed_tokens.table.weight': 'text_table.weight',
  'llm.embed_tokens.project.weight': 'text_project.weight',
  'llm.norm.weight': 'norm.weight',
  'llm.layers.0.input_layernorm.weight': 'layers.0.input_norm.weight',
  'llm.layers.0.post_attention_layernorm.weight': 'layers.0.post_norm.weight',
  'llm.layers.0.self_attn.v_proj.weight': 'layers.0.v.weight',
  'llm.layers.0.self_attn.o_proj.weight': 'layers.0.o.weight',
  'llm.layers.0.mlp.down_proj.weight': 'layers.0.down.weight',
};

const MAPPED = [
  'heads_coarse.weight',
  'heads_fine.weight',
  'layers.0.qk.weight',
  'layers.0.qk_norm',
  'layers.0.up_gate.weight',
];

/** Each head row of grid(0, 9, 4) minus its codebook's mean row: −10, 0, 10. */
const centered = (books: number, padRows: number) => [
  ...Array.from({ length: books }, () =>
    [-10, 0, 10].flatMap((v) => Array<number>(4).fill(v)),
  ).flat(),
  ...Array<number>(padRows * 4).fill(0),
];

describe('transformReverbStateDict', () => {
  it.each(DTYPES)('maps a $dtype release checkpoint', async ({ dtype, type }) => {
    const sd = await checkpoint(dtype, release());
    const before = new Map(sd.tensors);
    transformReverbStateDict(sd, '', CFG);
    expect([...sd.tensors.keys()].sort()).toEqual([...Object.values(RENAMED), ...MAPPED].sort());
    for (const [from, to] of Object.entries(RENAMED)) {
      expect(sd.tensors.get(to)).toBe(before.get(from));
    }
    const get = (name: string) => contents(sd.tensors.get(name)!);

    // Each 4-row head goes from the half split to interleaved pairs.
    const rope = [0, 2, 1, 3];
    expect(await get('layers.0.qk.weight')).toEqual({
      type,
      values: [
        ...gridRows(100, [...rope, ...rope.map((r) => r + 4)], 4),
        ...gridRows(200, rope, 4),
      ],
    });
    expect(await get('layers.0.qk_norm')).toEqual({
      type: 'f32',
      values: [0.5, 1.5, 1, 2, 0.5, 1.5, 1, 2, 5, 7, 6, 8],
    });
    expect(await get('layers.0.up_gate.weight')).toEqual({
      type,
      values: [...gridRows(500, [0, 1], 4), ...gridRows(600, [0, 1], 4)],
    });
    // Two coarse codebooks: 6 rows padded to 8. One fine: 3 padded to 4.
    expect(await get('heads_coarse.weight')).toEqual({ type: 'f32', values: centered(2, 2) });
    expect(await get('heads_fine.weight')).toEqual({ type: 'f32', values: centered(1, 1) });
  });

  it.each([8, 4] as const)('maps an int%i release checkpoint', async (bits) => {
    const groupSize = 2;
    const S = 'llm.layers.0';
    const { quant, sd: load } = quantRelease(bits, groupSize);
    const sd = await load();
    const before = new Map(sd.tensors);
    transformReverbStateDict(sd, '', CFG);
    for (const [from, to] of Object.entries(RENAMED)) {
      expect(sd.tensors.get(to)).toBe(before.get(from));
    }

    /** The ints and scales of `entry`'s rows in `order`. */
    const rows = ({ ints, scales, shape }: QuantEntry, order: number[]) => {
      const groups = shape[1] / groupSize;
      return {
        ints: order.flatMap((r) => ints.slice(r * shape[1], (r + 1) * shape[1])),
        scales: order.flatMap((r) => scales.slice(r * groups, (r + 1) * groups)),
      };
    };
    const joined = (...parts: { ints: number[]; scales: number[] }[]) => ({
      ints: parts.flatMap((p) => p.ints),
      scaleType: 'f16',
      scales: parts.flatMap((p) => p.scales),
    });
    const rope = [0, 2, 1, 3];
    const q = quant[`${S}.self_attn.q_proj.weight`]!;
    const k = quant[`${S}.self_attn.k_proj.weight`]!;
    expect(await quantContents(sd.tensors.get('layers.0.qk.weight')!)).toEqual(
      joined(rows(q, [...rope, ...rope.map((r) => r + 4)]), rows(k, rope)),
    );
    const up = quant[`${S}.mlp.up_proj.weight`]!;
    const gate = quant[`${S}.mlp.gate_proj.weight`]!;
    expect(await quantContents(sd.tensors.get('layers.0.up_gate.weight')!)).toEqual(
      joined(rows(up, [0, 1]), rows(gate, [0, 1])),
    );
  });

  it.each<[string, (sd: LazyStateDict) => void]>([
    ['a missing tensor', (sd) => sd.tensors.delete('llm.layers.0.mlp.up_proj.weight')],
    ['an unused tensor', (sd) => sd.tensors.set('llm.extra', sd.tensors.get('llm.norm.weight')!)],
    [
      'audio heads of another shape',
      (sd) => {
        const t = sd.tensors.get('audio_heads.weight')!;
        sd.tensors.set('audio_heads.weight', { ...t, shape: [12, 3] });
      },
    ],
  ])('throws CHECKPOINT_MISMATCH for %s', async (_, edit) => {
    const sd = await checkpoint('F32', release());
    edit(sd);
    expect(() => transformReverbStateDict(sd, '', CFG)).toThrow(
      expect.objectContaining({ code: 'CHECKPOINT_MISMATCH' }),
    );
  });
});

describe('ReverbModel', () => {
  it('binds every parameter from a release checkpoint', async () => {
    const model = new ReverbModel(CFG);
    const sd = await checkpoint('F32', release());
    await model.loadStateDict(sd, {
      upload: (_data, shape: ValueMeta) => pending(shape, 'reshape', []),
    });
    expect(model.text_table.gather([4])).toEqual(new Float32Array([40, 41]));
  });

  // Wide enough for int4, which packs 8 outputs to a word.
  const WIDE: ReverbConfig = { ...CFG, hidden: 8, ffn: 8, headDim: 8 };

  it.each([8, 4] as const)(
    'binds every Linear under llm.layers from an int%i checkpoint',
    async (bits) => {
      const model = new ReverbModel(WIDE);
      await model.loadStateDict(await quantRelease(bits, 4, WIDE).sd(), {
        upload: (_data, shape: ValueMeta) => pending(shape, 'reshape', []),
      });
      const scales = [...model.namedParameters()]
        .map(([name]) => name)
        .filter((name) => name.endsWith('weightScale'));
      expect(scales).toEqual(
        ['qk', 'v', 'o', 'up_gate', 'down'].map((m) => `layers.0.${m}.weightScale`),
      );
    },
  );

  it('puts codebook c of frame t at row t·codebooks + c of the audio table', () => {
    const model = new ReverbModel(CFG);
    // Codes [codebooks 3, frames 2].
    const ids = model.audioIds(Int32Array.from([0, 1, 2, 0, 1, 2]), 2);
    expect([...ids]).toEqual([0, 5, 7, 1, 3, 8]);
  });
});
