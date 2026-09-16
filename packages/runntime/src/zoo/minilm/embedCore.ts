/** MiniLM embed core: token ids in, mean-pooled hidden state out.
 *
 *  A single sentence runs as a recorded replay, one per padded length bucket,
 *  so a call only rewrites four small input buffers and re-submits. A batch
 *  runs eagerly with every sentence packed into one row.
 *
 *  Pooling is a matmul against a row of 1/L over the real tokens. */

import { createReplayCache, matmul, tensor, toArray, type Value } from '../../core/index.ts';
import { MinilmModel } from './model.ts';

/** Token counts a single text pads up to; 512 is the model's position limit. */
export const MINILM_BUCKETS: readonly number[] = [16, 32, 64, 128, 192, 256, 384, 512];

export function pickBucket(len: number, buckets: readonly number[] = MINILM_BUCKETS) {
  for (const b of buckets) if (len <= b) return b;
  return undefined;
}

const bucketShapes = (T: number) =>
  ({
    ids: [T, 1],
    positions: [T, 1],
    segments: [T, 2],
    selector: [1, T],
  }) as const;

export function fillBucketInputs(
  T: number,
  ids: readonly number[],
  out: {
    ids: Float32Array;
    positions: Float32Array;
    segments: Float32Array;
    selector: Float32Array;
  },
): void {
  const L = ids.length;
  if (L < 1 || L > T) throw new Error(`fillBucketInputs: ${L} tokens do not fit bucket ${T}`);
  for (let i = 0; i < T; i++) {
    out.ids[i] = i < L ? ids[i]! : 0;
    out.positions[i] = i < L ? i : i - L;
    out.segments[2 * i] = i < L ? 0 : L;
    out.segments[2 * i + 1] = i < L ? L : T;
    out.selector[i] = i < L ? 1 / L : 0;
  }
}

export interface MinilmEmbedCore {
  embedPooled(tokenIds: readonly number[]): Promise<Float32Array>;
  embedPooledBatch(tokenIdsList: readonly (readonly number[])[]): Promise<Float32Array>;
  warmBuckets(probeIds: readonly number[]): Promise<void>;
  dispose(): void;
}

export function createMinilmEmbedCore(model: MinilmModel): MinilmEmbedCore {
  // The selector multiplies the hidden states, so it follows the model's
  // dtype; the index planes stay f32.
  const actDtype = model.embeddings.norm.weight.shape.dtype === 'f16' ? 'f16' : 'f32';
  const pooler = (data: Float32Array, dims: [number, number]): Value =>
    tensor(data, { elems: data.length, dtype: actDtype, dims });
  // The replay mechanics live in core; this is the MiniLM-specific config.
  const replays = createReplayCache({
    buckets: MINILM_BUCKETS,
    inputs: bucketShapes,
    planeDtype: { selector: actDtype },
    // A full-length all-pad input in one segment, so the recording run
    // computes finite values.
    captureFill: (T, scratch) => fillBucketInputs(T, new Array<number>(T).fill(0), scratch),
    build: (T, inp) =>
      matmul(
        inp.selector,
        model.forward(inp.ids, { positions: inp.positions, segments: inp.segments, maxSegment: T }),
      ),
  });

  const embedPooledEager = (tokenIds: readonly number[]): Promise<Float32Array> => {
    const t = tokenIds.length;
    const s = pooler(new Float32Array(t).fill(1 / t), [1, t]);
    return toArray(matmul(s, model.forward(tokenIds)));
  };

  const embedPooled = (tokenIds: readonly number[]): Promise<Float32Array> => {
    const T = replays.pick(tokenIds.length);
    return T === undefined
      ? embedPooledEager(tokenIds)
      : replays.run(T, (_, scratch) => fillBucketInputs(T, tokenIds, scratch));
  };

  const embedPooledBatch = async (
    tokenIdsList: readonly (readonly number[])[],
  ): Promise<Float32Array> => {
    const lengths = tokenIdsList.map((ids) => ids.length);
    const packed = tokenIdsList.flat() as number[];
    const total = packed.length;
    const selector = new Float32Array(tokenIdsList.length * total);
    let start = 0;
    lengths.forEach((len, b) => {
      for (let i = 0; i < len; i++) selector[b * total + start + i] = 1 / len;
      start += len;
    });
    const hidden = model.forward(packed, lengths);
    const s = pooler(selector, [tokenIdsList.length, total]);
    return toArray(matmul(s, hidden));
  };

  const warmBuckets = async (probeIds: readonly number[]): Promise<void> => {
    for (const T of MINILM_BUCKETS) {
      await replays.run(T, (_, scratch) => fillBucketInputs(T, probeIds, scratch));
    }
  };

  return { embedPooled, embedPooledBatch, warmBuckets, dispose: () => replays.dispose() };
}
