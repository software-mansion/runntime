import { describe, expect, it } from 'vitest';
import type { LazyStateDict } from '../../../src/core/weights/safetensors.ts';
import {
  EMBEDDINGGEMMA2_AUDIO,
  EMBEDDINGGEMMA2_TEXT,
  EMBEDDINGGEMMA2_VISION,
  type EmbeddingGemma2AudioConfig,
  type EmbeddingGemma2Config,
  type EmbeddingGemma2VisionConfig,
} from '../../../src/zoo/embeddinggemma2/config.ts';
import {
  transformEmbeddingGemma2AudioStateDict,
  transformEmbeddingGemma2StateDict,
  transformEmbeddingGemma2VisionStateDict,
} from '../../../src/zoo/embeddinggemma2/stateDictHooks.ts';
import { checkpoint, contents, type Entry } from '../../core/weights/checkpoint.ts';

const DTYPES = [
  { dtype: 'F32', type: 'f32', elemBytes: 4 },
  { dtype: 'F16', type: 'f16', elemBytes: 2 },
] as const;

const ZERO: Entry = { values: [0], shape: [1] };

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

const ones = (n: number) => new Array<number>(n).fill(1);

/** Maps `sd` and returns a getter over the result, after checking that the
 *  result holds exactly `untouched` plus `mapped`, with each untouched tensor
 *  passed through as it was. */
function mapAndCheck(
  sd: LazyStateDict,
  run: (sd: LazyStateDict) => void,
  untouched: string[],
  mapped: string[],
) {
  const before = new Map(sd.tensors);
  run(sd);
  expect([...sd.tensors.keys()].sort()).toEqual([...untouched, ...mapped].sort());
  for (const name of untouched) expect(sd.tensors.get(name)).toBe(before.get(name));
  return (name: string) => sd.tensors.get(name)!;
}

describe('transformEmbeddingGemma2StateDict', () => {
  // A sliding layer, then a full one: hidden 4, 2 q heads and 1 kv head,
  // head size 4 for sliding and 8 for full, so each layer type gets its own
  // rope block. hidden^-½ = 0.5, so the PLE scale is exact in f16.
  const cfg: EmbeddingGemma2Config = {
    ...EMBEDDINGGEMMA2_TEXT,
    hidden: 4,
    heads: 2,
    layerTypes: ['sliding', 'full'],
    attention: {
      sliding: { headDim: 4, kvHeads: 1, theta: 10000 },
      full: { headDim: 8, kvHeads: 1, theta: 10000 },
    },
  };

  const S = 'language_model.';
  const L = (i: number) => `language_model.layers.${i}`;
  /** Tensors the hook leaves as they are, under their own names. */
  const layerUntouched = (i: number) =>
    [
      'self_attn.o_proj.weight',
      'mlp.gate_proj.weight',
      'mlp.up_proj.weight',
      'mlp.down_proj.weight',
      'ple_block.per_layer_input_gate.weight',
      'ple_block.per_layer_projection.weight',
      'ple_block.post_per_layer_input_norm.weight',
      'input_layernorm.weight',
      'post_attention_layernorm.weight',
      'pre_feedforward_layernorm.weight',
      'post_feedforward_layernorm.weight',
    ].map((n) => `${L(i)}.${n}`);
  const UNTOUCHED = [
    ...layerUntouched(0),
    ...layerUntouched(1),
    `${S}ple.per_layer_projection_norm.weight`,
    `${S}embed_tokens.weight`,
    `${S}norm.weight`,
    `${S}embedding_projection.weight`,
  ];
  /** Layer `i`'s attention: q over `heads` heads of `d` rows, k and v over
   *  one, norms 1 … d and d + 1 … 2d. */
  const attention = (i: number, d: number): Record<string, Entry> => ({
    [`${L(i)}.self_attn.q_proj.weight`]: grid(0, 2 * d, 4),
    [`${L(i)}.self_attn.k_proj.weight`]: grid(100, d, 4),
    [`${L(i)}.self_attn.v_proj.weight`]: grid(200, d, 4),
    [`${L(i)}.self_attn.q_norm.weight`]: {
      values: Array.from({ length: d }, (_, r) => r + 1),
      shape: [d],
    },
    [`${L(i)}.self_attn.k_norm.weight`]: {
      values: Array.from({ length: d }, (_, r) => d + r + 1),
      shape: [d],
    },
    [`${L(i)}.layer_scalar`]: { values: [0.25], shape: [1] },
  });
  const CHECKPOINT: Record<string, Entry> = {
    ...Object.fromEntries(UNTOUCHED.map((name) => [name, ZERO])),
    ...attention(0, 4),
    ...attention(1, 8),
    [`${S}ple.per_layer_model_projection.weight`]: grid(0, 2, 4),
  };

  it.for(DTYPES)('maps a $dtype checkpoint', async ({ dtype, type, elemBytes }) => {
    const sd = await checkpoint(dtype, CHECKPOINT);
    const total = sd.totalBytes();
    const get = mapAndCheck(sd, (s) => transformEmbeddingGemma2StateDict(s, '', cfg), UNTOUCHED, [
      `${L(0)}.self_attn.qkv.weight`,
      `${L(0)}.self_attn.qkv_norm`,
      `${L(0)}.layer_scalar`,
      `${L(1)}.self_attn.qkv.weight`,
      `${L(1)}.self_attn.qkv_norm`,
      `${L(1)}.layer_scalar`,
      `${S}ple.per_layer_model_projection.weight`,
    ]);

    // Rope pairs (p, p + d/2) of each d-row head become rows (2p, 2p + 1):
    // rows 0 2 1 3 of a 4-row head, 0 4 1 5 2 6 3 7 of an 8-row one. The
    // layer's type picks d, so the full layer must not reorder like the
    // sliding one. v keeps its order.
    const layers: [i: number, d: number, head: number[]][] = [
      [0, 4, [0, 2, 1, 3]],
      [1, 8, [0, 4, 1, 5, 2, 6, 3, 7]],
    ];
    for (const [i, d, head] of layers) {
      const qkv = get(`${L(i)}.self_attn.qkv.weight`);
      expect(qkv.shape).toEqual([4 * d, 4]);
      expect(qkv.byteLength).toBe(16 * d * elemBytes);
      expect(await contents(qkv)).toEqual({
        type,
        values: [
          ...gridRows(0, [...head, ...head.map((r) => r + d)], 4),
          ...gridRows(100, head, 4),
          ...gridRows(
            200,
            Array.from({ length: d }, (_, r) => r),
            4,
          ),
        ],
      });

      // q_norm reordered like q, once per q head; k_norm once; ones for v.
      // Each norm's bytes count once.
      const norm = get(`${L(i)}.self_attn.qkv_norm`);
      expect(norm.shape).toEqual([4 * d]);
      expect(norm.byteLength).toBe(2 * d * elemBytes);
      const qn = head.map((r) => r + 1);
      const kn = head.map((r) => d + r + 1);
      expect(await contents(norm)).toEqual({
        type,
        values: [...qn, ...qn, ...kn, ...ones(d)],
      });

      // The CPU parameter reads f32 whatever the checkpoint holds.
      const scalar = get(`${L(i)}.layer_scalar`);
      expect(scalar.byteLength).toBe(elemBytes);
      expect(await contents(scalar)).toEqual({ type: 'f32', values: [0.25] });
    }

    // Scaled, so f32 on every checkpoint.
    const ple = get(`${S}ple.per_layer_model_projection.weight`);
    expect(ple.shape).toEqual([2, 4]);
    expect(ple.byteLength).toBe(8 * elemBytes);
    expect(await contents(ple)).toEqual({ type: 'f32', values: [0, 0.5, 1, 1.5, 5, 5.5, 6, 6.5] });

    expect(sd.totalBytes()).toBe(total);
  });
});

describe('transformEmbeddingGemma2VisionStateDict', () => {
  // One layer with one head of size 8: each half head of 4 rows has its own
  // rope, so q and k reorder per 4 rows.
  const cfg: EmbeddingGemma2VisionConfig = {
    ...EMBEDDINGGEMMA2_VISION,
    hidden: 2,
    heads: 1,
    headDim: 8,
    positions: 2,
    layers: 1,
  };

  const P = 'vision_tower.patch_embedder';
  const T = 'vision_tower.encoder.layers.0';
  /** Tensors the hook leaves as they are, under their own names. */
  const UNTOUCHED = [
    `${P}.input_proj.weight`,
    `${T}.self_attn.o_proj.linear.weight`,
    `${T}.mlp.gate_proj.linear.weight`,
    `${T}.mlp.up_proj.linear.weight`,
    `${T}.mlp.down_proj.linear.weight`,
    `${T}.input_layernorm.weight`,
    `${T}.post_attention_layernorm.weight`,
    `${T}.pre_feedforward_layernorm.weight`,
    `${T}.post_feedforward_layernorm.weight`,
    'embed_vision.embedding_projection.weight',
  ];
  const CHECKPOINT: Record<string, Entry> = {
    ...Object.fromEntries(UNTOUCHED.map((name) => [name, ZERO])),
    // [2, positions, hidden]: x rows 1 2 | 3 4, y rows 5 6 | 7 8.
    'vision_tower.patch_embedder.position_embedding_table': {
      values: [1, 2, 3, 4, 5, 6, 7, 8],
      shape: [2, 2, 2],
    },
    [`${T}.self_attn.q_proj.linear.weight`]: grid(0, 8, 2),
    [`${T}.self_attn.k_proj.linear.weight`]: grid(100, 8, 2),
    [`${T}.self_attn.v_proj.linear.weight`]: grid(200, 8, 2),
    [`${T}.self_attn.q_norm.weight`]: { values: [1, 2, 3, 4, 5, 6, 7, 8], shape: [8] },
    [`${T}.self_attn.k_norm.weight`]: { values: [11, 12, 13, 14, 15, 16, 17, 18], shape: [8] },
  };

  it.for(DTYPES)('maps a $dtype checkpoint', async ({ dtype, type, elemBytes }) => {
    const sd = await checkpoint(dtype, CHECKPOINT);
    const total = sd.totalBytes();
    const get = mapAndCheck(
      sd,
      (s) => transformEmbeddingGemma2VisionStateDict(s, '', cfg),
      UNTOUCHED,
      [
        `${P}.pos_x.weight`,
        `${P}.pos_y.weight`,
        `${T}.self_attn.qkv.weight`,
        `${T}.self_attn.qkv_norm`,
      ],
    );

    for (const [name, values] of [
      [`${P}.pos_x.weight`, [1, 2, 3, 4]],
      [`${P}.pos_y.weight`, [5, 6, 7, 8]],
    ] as const) {
      const table = get(name);
      expect(table.shape).toEqual([2, 2]);
      expect(table.byteLength).toBe(4 * elemBytes);
      expect(await contents(table)).toEqual({ type, values });
    }

    // Rows 0 2 1 3 of each half head.
    const halfHeads = [0, 2, 1, 3, 4, 6, 5, 7];
    const qkv = get(`${T}.self_attn.qkv.weight`);
    expect(qkv.shape).toEqual([24, 2]);
    expect(qkv.byteLength).toBe(48 * elemBytes);
    expect(await contents(qkv)).toEqual({
      type,
      values: [
        ...gridRows(0, halfHeads, 2),
        ...gridRows(100, halfHeads, 2),
        ...gridRows(200, [0, 1, 2, 3, 4, 5, 6, 7], 2),
      ],
    });

    const norm = get(`${T}.self_attn.qkv_norm`);
    expect(norm.shape).toEqual([24]);
    expect(norm.byteLength).toBe(16 * elemBytes);
    expect(await contents(norm)).toEqual({
      type,
      values: [1, 3, 2, 4, 5, 7, 6, 8, 11, 13, 12, 14, 15, 17, 16, 18, ...ones(8)],
    });

    expect(sd.totalBytes()).toBe(total);
  });
});

describe('transformEmbeddingGemma2AudioStateDict', () => {
  // One layer, hidden 4, one 4-wide head (scale 4^-½ = 0.5), conv channels
  // 2 then 1, 2 conv taps, window 2.
  const cfg: EmbeddingGemma2AudioConfig = {
    ...EMBEDDINGGEMMA2_AUDIO,
    hidden: 4,
    headDim: 4,
    convKernel: 2,
    window: 2,
    convChannels: [2, 1],
    layers: 1,
    residualWeight: 0.5,
    outputDim: 2,
  };

  const A = 'audio_tower.';
  const sub = `${A}subsample_conv_projection.`;
  const T = `${A}layers.0`;
  /** Every clipped linear but q, k and v. */
  const CLIPPED = [
    `${T}.feed_forward1.ffw_layer_1`,
    `${T}.feed_forward1.ffw_layer_2`,
    `${T}.feed_forward2.ffw_layer_1`,
    `${T}.feed_forward2.ffw_layer_2`,
    `${T}.self_attn.post`,
    `${T}.lconv1d.linear_start`,
    `${T}.lconv1d.linear_end`,
  ];
  const BOUNDS = ['input_min', 'input_max', 'output_min', 'output_max'];
  /** Tensors the hook leaves as they are, under their own names. */
  const UNTOUCHED = [
    ...CLIPPED.flatMap((c) => [`${c}.linear.weight`, ...BOUNDS.map((b) => `${c}.${b}`)]),
    `${sub}layer0.norm.weight`,
    `${sub}layer1.norm.weight`,
    `${sub}input_proj_linear.weight`,
    `${T}.feed_forward1.pre_layer_norm.weight`,
    `${T}.feed_forward2.pre_layer_norm.weight`,
    `${T}.lconv1d.pre_layer_norm.weight`,
    `${T}.lconv1d.conv_norm.weight`,
    `${T}.norm_pre_attn.weight`,
    `${T}.norm_post_attn.weight`,
    `${T}.norm_out.weight`,
    `${A}output_proj.weight`,
    `${A}output_proj.bias`,
    'embed_audio.embedding_projection.weight',
  ];
  const bound = (prefix: string, values: number[]) =>
    Object.fromEntries(
      BOUNDS.map((b, i) => [`${prefix}.${b}`, { values: [values[i]!], shape: [1] }]),
    );
  const CHECKPOINT: Record<string, Entry> = {
    ...Object.fromEntries(UNTOUCHED.map((name) => [name, ZERO])),
    ...Object.assign({}, ...CLIPPED.map((p) => bound(p, [-1, 1, -2, 2]))),
    ...Object.assign(
      {},
      ...['q_proj', 'k_proj', 'v_proj'].map((p) => bound(`${T}.self_attn.${p}`, [-3, 3, -4, 4])),
    ),
    // [c0, 1, 3, 3] and [c1, c0, 3, 3], values 1 … 18.
    [`${sub}layer0.conv.weight`]: {
      values: Array.from({ length: 18 }, (_, i) => i + 1),
      shape: [2, 1, 3, 3],
    },
    [`${sub}layer1.conv.weight`]: {
      values: Array.from({ length: 18 }, (_, i) => i + 1),
      shape: [1, 2, 3, 3],
    },
    [`${T}.feed_forward1.post_layer_norm.weight`]: { values: [2, 4, 6, 8], shape: [4] },
    [`${T}.feed_forward2.post_layer_norm.weight`]: { values: [1, 3, 5, 7], shape: [4] },
    [`${T}.self_attn.q_proj.linear.weight`]: grid(0, 4, 4),
    [`${T}.self_attn.k_proj.linear.weight`]: grid(100, 4, 4),
    [`${T}.self_attn.v_proj.linear.weight`]: grid(200, 4, 4),
    // Identity, so each bias row is the sinusoid row itself.
    [`${T}.self_attn.relative_k_proj.weight`]: {
      values: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1],
      shape: [4, 4],
    },
    // softplus(0) = ln 2, so each q scale is 4^-½ = 0.5.
    [`${T}.self_attn.per_dim_scale`]: { values: [0, 0, 0, 0], shape: [4] },
    // [hidden, 1, K]: channel c holds taps 2c + 1, 2c + 2.
    [`${T}.lconv1d.depthwise_conv1d.weight`]: {
      values: [1, 2, 3, 4, 5, 6, 7, 8],
      shape: [4, 1, 2],
    },
  };
  it.for(DTYPES)('maps a $dtype checkpoint', async ({ dtype, type, elemBytes }) => {
    const sd = await checkpoint(dtype, CHECKPOINT);
    const total = sd.totalBytes();
    const before = new Map(sd.tensors);
    const get = mapAndCheck(
      sd,
      (s) => transformEmbeddingGemma2AudioStateDict(s, '', cfg),
      UNTOUCHED,
      [
        `${sub}layer0.conv.weight`,
        `${sub}layer1.conv.weight`,
        `${T}.feed_forward1.post_layer_norm.weight`,
        `${T}.feed_forward2.post_layer_norm.weight`,
        `${T}.self_attn.qkv.linear.weight`,
        ...BOUNDS.map((b) => `${T}.self_attn.qkv.${b}`),
        `${T}.self_attn.rel`,
        `${T}.self_attn.q_scale`,
        `${T}.lconv1d.depthwise_conv1d.weight`,
      ],
    );

    // conv0 only folds its taps: the source tensor, reshaped.
    const source = before.get(`${sub}layer0.conv.weight`)!;
    const conv0 = get(`${sub}layer0.conv.weight`);
    expect(conv0.shape).toEqual([2, 9]);
    expect(conv0.byteLength).toBe(18 * elemBytes);
    if (source.kind === 'quant' || conv0.kind === 'quant')
      throw new Error('expected float tensors');
    expect([conv0.kind, conv0.dtype]).toEqual([source.kind, source.dtype]);

    // Tap k of channel c moves from c·9 + k to k·2 + c.
    const conv1 = get(`${sub}layer1.conv.weight`);
    expect(conv1.shape).toEqual([1, 18]);
    expect(conv1.byteLength).toBe(18 * elemBytes);
    expect(await contents(conv1)).toEqual({
      type,
      values: [1, 10, 2, 11, 3, 12, 4, 13, 5, 14, 6, 15, 7, 16, 8, 17, 9, 18],
    });

    // Times the residual weight 0.5, so f32 on every checkpoint.
    for (const [name, values] of [
      [`${T}.feed_forward1.post_layer_norm.weight`, [1, 2, 3, 4]],
      [`${T}.feed_forward2.post_layer_norm.weight`, [0.5, 1.5, 2.5, 3.5]],
    ] as const) {
      const norm = get(name);
      expect(norm.shape).toEqual([4]);
      expect(norm.byteLength).toBe(4 * elemBytes);
      expect(await contents(norm)).toEqual({ type: 'f32', values });
    }

    const qkv = get(`${T}.self_attn.qkv.linear.weight`);
    expect(qkv.shape).toEqual([12, 4]);
    expect(qkv.byteLength).toBe(48 * elemBytes);
    expect(await contents(qkv)).toEqual({
      type,
      values: [0, 100, 200].flatMap((base) => gridRows(base, [0, 1, 2, 3], 4)),
    });

    // The fused projection's bounds are q's, f32, each carrying one
    // scalar's bytes.
    for (const [i, b] of BOUNDS.entries()) {
      const value = get(`${T}.self_attn.qkv.${b}`);
      expect(value.byteLength).toBe(elemBytes);
      expect(await contents(value)).toEqual({ type: 'f32', values: [[-3, 3, -4, 4][i]!] });
    }

    // Sinusoid rows for distances 0 and 1 at hidden 4: angle j·10000^-i for
    // i = 0, 1, sines then cosines.
    const rel = get(`${T}.self_attn.rel`);
    expect(rel.shape).toEqual([2, 4]);
    expect(rel.byteLength).toBe(16 * elemBytes);
    const relValues = await contents(rel);
    expect(relValues.type).toBe('f32');
    const want = [0, 0, 1, 1, Math.sin(1), Math.sin(1e-4), Math.cos(1), Math.cos(1e-4)];
    relValues.values.forEach((v, i) => expect(v).toBeCloseTo(want[i]!, 6));

    const qScale = get(`${T}.self_attn.q_scale`);
    expect(qScale.shape).toEqual([4]);
    expect(qScale.byteLength).toBe(4 * elemBytes);
    const qScaleValues = await contents(qScale);
    expect(qScaleValues.type).toBe('f32');
    qScaleValues.values.forEach((v) => expect(v).toBeCloseTo(0.5, 6));

    // [hidden, 1, K] → [K, hidden]: tap 0 of every channel, then tap 1.
    const lconv = get(`${T}.lconv1d.depthwise_conv1d.weight`);
    expect(lconv.shape).toEqual([2, 4]);
    expect(lconv.byteLength).toBe(8 * elemBytes);
    expect(await contents(lconv)).toEqual({ type, values: [1, 3, 5, 7, 2, 4, 6, 8] });

    // k's and v's bounds, 4 scalars each, are the only bytes lost.
    expect(sd.totalBytes()).toBe(total - 8 * elemBytes);
  });
});
