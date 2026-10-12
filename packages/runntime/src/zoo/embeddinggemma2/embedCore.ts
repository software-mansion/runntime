/** EmbeddingGemma 2 embed core: token ids in, pooled embeddings out.
 *
 *  The embedding table lives on the CPU, so a forward's input is the embedded
 *  rows, written with positions, segments and pool weights. One input up to
 *  the largest bucket runs as a recorded replay per padded length; a longer
 *  input and a batch run eagerly. A batch packs its inputs into one
 *  sequence: segments keep them from attending to each other, positions
 *  restart per input and pick rope rows from one shared table, and each pool
 *  row averages one input. */

import {
  createReplayCache,
  gatherRowsFrom,
  tensor,
  toArray,
  type Value,
} from '../../core/index.ts';
import { EMBEDDINGGEMMA2_MAX_TOKENS, type LayerType } from './config.ts';
import type { EmbeddingGemma2Inputs, EmbeddingGemma2Model } from './model.ts';
import { ropeTables } from './rope.ts';

/** Token counts a single input pads up to. Longer inputs run eagerly. */
const EMBEDDINGGEMMA2_BUCKETS: readonly number[] = [16, 32, 64, 128, 256, 512];

interface Planes {
  x0: Float32Array;
  positions: Float32Array;
  segments: Float32Array;
  selector: Float32Array;
}

/** Packs inputs back to back into T rows: input b covers its own segment and
 *  pool row b. Rows past the inputs are one more segment, pooled by no one. */
function fillInputs(
  model: EmbeddingGemma2Model,
  T: number,
  idsList: readonly (readonly number[])[],
  out: Planes,
): void {
  out.x0.fill(0);
  out.selector.fill(0);
  let start = 0;
  idsList.forEach((ids, b) => {
    const len = ids.length;
    model.embedRows(ids, out.x0, start);
    for (let i = 0; i < len; i++) {
      const t = start + i;
      out.positions[t] = i;
      out.segments[2 * t] = start;
      out.segments[2 * t + 1] = start + len;
      out.selector[b * T + t] = 1 / len;
    }
    start += len;
  });
  for (let t = start; t < T; t++) {
    out.positions[t] = t - start;
    out.segments[2 * t] = start;
    out.segments[2 * t + 1] = T;
  }
}

export interface EmbeddingGemma2EmbedCore {
  /** One input's pooled embedding [embeddingDim]. */
  embedPooled(ids: readonly number[]): Promise<Float32Array>;
  /** Pooled embeddings of many inputs, [B · embeddingDim] in order. */
  embedPooledBatch(idsList: readonly (readonly number[])[]): Promise<Float32Array>;
  /** Records every bucket's replay now. */
  warmBuckets(probeIds: readonly number[]): Promise<void>;
  dispose(): void;
}

export function createEmbeddingGemma2EmbedCore(
  model: EmbeddingGemma2Model,
): EmbeddingGemma2EmbedCore {
  const { hidden, embeddingDim, attention } = model.cfg;
  // x0 and the pool weights meet the hidden states, so they follow the
  // model's dtype; the index planes stay f32.
  const act = model.norm.scale.shape.dtype === 'f16' ? 'f16' : 'f32';
  const maxT = EMBEDDINGGEMMA2_MAX_TOKENS;
  // Rope rows for every position, gathered per forward by the positions plane.
  const table = (type: LayerType) => {
    const { headDim, theta } = attention[type];
    const { cos, sin } = ropeTables(maxT, headDim, theta);
    return { cos: tensor(cos, [maxT, headDim]), sin: tensor(sin, [maxT, headDim]) };
  };
  const tables = { sliding: table('sliding'), full: table('full') };

  const inputsOf = (
    planes: Record<keyof Planes, Value>,
    maxSegment: number,
  ): EmbeddingGemma2Inputs => {
    const pick = (type: LayerType) => ({
      cos: gatherRowsFrom(tables[type].cos, planes.positions),
      sin: gatherRowsFrom(tables[type].sin, planes.positions),
    });
    return {
      x0: planes.x0,
      rope: { sliding: pick('sliding'), full: pick('full') },
      selector: planes.selector,
      segments: planes.segments,
      maxSegment,
    };
  };

  const replays = createReplayCache({
    buckets: EMBEDDINGGEMMA2_BUCKETS,
    inputs: (T) =>
      ({ x0: [T, hidden], positions: [T, 1], segments: [T, 2], selector: [1, T] }) as const,
    planeDtype: { x0: act, selector: act },
    // One full-length input of <pad> rows keeps the recording run finite.
    captureFill: (T, scratch) => fillInputs(model, T, [new Array<number>(T).fill(0)], scratch),
    build: (T, inp) => model.forward(inputsOf(inp, T)),
  });

  /** One eager forward over inputs packed back to back. T rounds up to a
   *  multiple of 4, so the attention score matmuls take the vec4 kernels. */
  const runEager = async (idsList: readonly (readonly number[])[]): Promise<Float32Array> => {
    const T = Math.ceil(idsList.reduce((s, ids) => s + ids.length, 0) / 4) * 4;
    const B = idsList.length;
    const planes: Planes = {
      x0: new Float32Array(T * hidden),
      positions: new Float32Array(T),
      segments: new Float32Array(2 * T),
      selector: new Float32Array(B * T),
    };
    fillInputs(model, T, idsList, planes);
    const upload = (data: Float32Array, rows: number, dtype: 'f32' | 'f16') =>
      tensor(data, { elems: data.length, dtype, dims: [rows, data.length / rows] });
    const values = {
      x0: upload(planes.x0, T, act),
      positions: upload(planes.positions, T, 'f32'),
      segments: upload(planes.segments, T, 'f32'),
      selector: upload(planes.selector, B, act),
    };
    try {
      const longest = Math.max(...idsList.map((ids) => ids.length));
      return await toArray(model.forward(inputsOf(values, longest)));
    } finally {
      for (const v of Object.values(values)) v.buffer.destroy();
    }
  };

  const embedPooled = (ids: readonly number[]): Promise<Float32Array> => {
    const T = replays.pick(ids.length);
    return T === undefined
      ? runEager([ids])
      : replays.run(T, (_, scratch) => fillInputs(model, T, [ids], scratch));
  };

  const embedPooledBatch = async (
    idsList: readonly (readonly number[])[],
  ): Promise<Float32Array> => {
    // Greedy packs in order, each up to maxT tokens.
    const packs: (readonly number[])[][] = [];
    let tokens = maxT;
    for (const ids of idsList) {
      if (tokens + ids.length > maxT) {
        packs.push([]);
        tokens = 0;
      }
      packs.at(-1)!.push(ids);
      tokens += ids.length;
    }
    const out = new Float32Array(idsList.length * embeddingDim);
    let row = 0;
    for (const pack of packs) {
      out.set(await runEager(pack), row * embeddingDim);
      row += pack.length;
    }
    return out;
  };

  const warmBuckets = async (probeIds: readonly number[]): Promise<void> => {
    for (const T of EMBEDDINGGEMMA2_BUCKETS) {
      await replays.run(T, (_, scratch) => fillInputs(model, T, [probeIds], scratch));
    }
  };

  return {
    embedPooled,
    embedPooledBatch,
    warmBuckets,
    dispose() {
      replays.dispose();
      for (const { cos, sin } of Object.values(tables)) {
        cos.buffer.destroy();
        sin.buffer.destroy();
      }
    },
  };
}
