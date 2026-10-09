/** EmbeddingGemma 2 embed core: input rows in, pooled embeddings out, as
 *  recorded replays per padded length bucket, like MiniLM's.
 *
 *  The embedding table lives on the CPU, so a replay's input is the embedded
 *  rows themselves: a call gathers and scales them (or copies an image's soft
 *  tokens in), writes them with positions, segments and pool weights, and
 *  re-submits. A batch packs several inputs into one sequence: segments keep
 *  them from attending to each other, positions restart per input and pick
 *  rope rows from one shared table, and each pool row averages one input.
 *
 *  Two replay families: one pool row for single calls, PACK_ROWS for batches,
 *  so a single query never reads back a batch-sized result. */

import { createReplayCache, gatherRowsFrom, tensor, type Value } from '../../core/index.ts';
import {
  ATTN,
  EMBEDDINGGEMMA2_MAX_TOKENS,
  ropeTables,
  type EmbeddingGemma2TextModel,
} from './model.ts';

/** Token counts a pass pads up to. An image is ~270 tokens. */
export const EMBEDDINGGEMMA2_BUCKETS: readonly number[] = [
  16,
  32,
  64,
  128,
  192,
  256,
  288,
  320,
  384,
  448,
  EMBEDDINGGEMMA2_MAX_TOKENS,
];

/** Inputs packed into one batched pass, at most. */
export const PACK_ROWS = 64;

export interface EmbedInput {
  /** Embedded rows [length·hidden]. */
  rows: Float32Array;
  length: number;
}

export interface EmbeddingGemma2EmbedCore {
  /** One input to its pooled embedding [embeddingDim]. */
  embed(input: EmbedInput): Promise<Float32Array>;
  /** Many inputs, packed into as few passes as fit. One pooled embedding
   *  each, in order. */
  embedBatch(inputs: readonly EmbedInput[]): Promise<Float32Array[]>;
  /** Records the single-input bucket for `length` now. */
  warm(length: number): Promise<void>;
  dispose(): void;
}

type Planes = {
  x0: Float32Array;
  positions: Float32Array;
  segments: Float32Array;
  selector: Float32Array;
};

export function createEmbeddingGemma2EmbedCore(
  model: EmbeddingGemma2TextModel,
): EmbeddingGemma2EmbedCore {
  const { hidden, embeddingDim } = model.cfg;
  const act = model.norm.scale.shape.dtype === 'f16' ? 'f16' : 'f32';
  const maxT = EMBEDDINGGEMMA2_MAX_TOKENS;
  // Rope rows for every position, gathered per call by the positions plane.
  const fullTables = Object.fromEntries(
    (['sliding', 'full'] as const).map((type) => {
      const { headDim, theta } = ATTN[type];
      const { cos, sin } = ropeTables(maxT, headDim, theta);
      return [type, { cos: tensor(cos, [maxT, headDim]), sin: tensor(sin, [maxT, headDim]) }];
    }),
  ) as Record<'sliding' | 'full', { cos: Value; sin: Value }>;

  /** Packs inputs back to back into bucket T: input b covers its own
   *  segment and pool row b. Padding is one more segment, pooled by no one. */
  const fill = (T: number, inputs: readonly EmbedInput[], out: Planes) => {
    out.x0.fill(0);
    out.selector.fill(0);
    let start = 0;
    inputs.forEach(({ rows, length }, b) => {
      out.x0.set(rows.subarray(0, length * hidden), start * hidden);
      for (let i = 0; i < length; i++) {
        const t = start + i;
        out.positions[t] = i;
        out.segments[2 * t] = start;
        out.segments[2 * t + 1] = start + length;
        out.selector[b * T + t] = 1 / length;
      }
      start += length;
    });
    for (let t = start; t < T; t++) {
      out.positions[t] = t - start;
      out.segments[2 * t] = start;
      out.segments[2 * t + 1] = T;
    }
  };

  const makeCache = (poolRows: number) =>
    createReplayCache({
      buckets: EMBEDDINGGEMMA2_BUCKETS,
      inputs: (T) =>
        ({
          x0: [T, hidden],
          positions: [T, 1],
          segments: [T, 2],
          selector: [poolRows, T],
        }) as const,
      planeDtype: { x0: act, selector: act },
      // One full-length input of <pad> rows keeps the recording run finite.
      captureFill: (T, scratch) =>
        fill(T, [{ rows: model.embedRows(new Array<number>(T).fill(0)), length: T }], scratch),
      build: (T, inp) => {
        const gather = (type: 'sliding' | 'full') => ({
          cos: gatherRowsFrom(fullTables[type].cos, inp.positions),
          sin: gatherRowsFrom(fullTables[type].sin, inp.positions),
        });
        return model.forwardEmbeds(inp.x0, {
          tables: { sliding: gather('sliding'), full: gather('full') },
          selector: inp.selector,
          segments: inp.segments,
          maxSegment: T,
        });
      },
    });
  const single = makeCache(1);
  const packed = makeCache(PACK_ROWS);

  const bucket = (cache: typeof single, length: number) => {
    const T = cache.pick(length);
    if (T === undefined) throw new Error(`embeddinggemma2: ${length} tokens, expected 1..${maxT}`);
    return T;
  };

  const embed = (input: EmbedInput) => {
    const T = bucket(single, input.length);
    return single.run(T, (_, scratch) => fill(T, [input], scratch));
  };

  const embedBatch = async (inputs: readonly EmbedInput[]): Promise<Float32Array[]> => {
    if (inputs.length === 1) return [await embed(inputs[0]!)];
    // Greedy packs in order: each fills up to maxT tokens or PACK_ROWS inputs.
    const packs: EmbedInput[][] = [];
    let cur: EmbedInput[] = [];
    let tokens = 0;
    for (const input of inputs) {
      if (input.length > maxT) throw new Error(`embeddinggemma2: ${input.length} tokens > ${maxT}`);
      if (cur.length === PACK_ROWS || tokens + input.length > maxT) {
        packs.push(cur);
        cur = [];
        tokens = 0;
      }
      cur.push(input);
      tokens += input.length;
    }
    if (cur.length) packs.push(cur);
    const out: Float32Array[] = [];
    for (const pack of packs) {
      const total = pack.reduce((s, x) => s + x.length, 0);
      const T = bucket(packed, total);
      const pooled = await packed.run(T, (_, scratch) => fill(T, pack, scratch));
      pack.forEach((_, b) => out.push(pooled.slice(b * embeddingDim, (b + 1) * embeddingDim)));
    }
    return out;
  };

  return {
    embed,
    embedBatch,
    async warm(length) {
      const T = bucket(single, length);
      const rows = model.embedRows(new Array<number>(length).fill(2));
      await single.run(T, (_, scratch) => fill(T, [{ rows, length }], scratch));
    },
    dispose() {
      single.dispose();
      packed.dispose();
      for (const t of Object.values(fullTables)) {
        t.cos.buffer.destroy();
        t.sin.buffer.destroy();
      }
    },
  };
}
