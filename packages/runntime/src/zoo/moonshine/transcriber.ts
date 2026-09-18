/** Moonshine transcriber: load once, then each transcribe(audio) encodes once
 *  and runs a greedy KV-cached decode. Audio of any length works — a long clip
 *  is split on quiet points and each chunk transcribed as its own utterance.
 *
 *  Decoding runs in bursts: within a burst each step's argmax id feeds the next
 *  step's embedding on device, so the GPU runs the whole burst with no CPU
 *  round-trip. A burst may overshoot EOS; those ids are discarded.
 *
 *  Rope tables and the per-layer K/V caches are preallocated once and written
 *  in place, so nothing re-uploads per step and no pipeline recompiles.
 *  Utterances reuse those buffers, so transcribe() serializes its calls. */

import {
  defaultRoot,
  evalValues,
  gpuExecutor,
  RunntimeError,
  slice,
  tensor,
  writeRows,
  type GpuPerfSink,
  type LazyStateDict,
  type Value,
} from '../../core/index.ts';
import { presetFromStateDict, type MoonshineConfig } from './config.ts';
import { splitIntoChunks, type ChunkOpts } from './chunking.ts';
import { buildRopeTables } from './rope.ts';
import { MIN_SAMPLES, stemFrames } from './convStem.ts';
import { MoonshineModel } from './model.ts';
import type { SelfKV } from './decoder.ts';
import { decodeTokens, type MoonshineTokenizer } from './tokenizer.ts';

export interface TranscribeResult {
  ids: number[];
  text: string;
  timings: { encodeMs: number; decodeMs: number; steps: number };
}

export interface Transcriber {
  readonly minSamples: number;
  /** Calls run one at a time: one made while another is in flight waits for
   *  it, so overlapping calls never share the decode caches. */
  transcribe(audio: Float32Array): Promise<TranscribeResult>;
  dispose(): void;
}

export interface BurstSample {
  positions: number;
  buildMs: number;
  evalMs: number;
  readbackMs: number;
}

export interface TranscriberPerf {
  burst?: number;
  syncAfterEncode?: boolean;
  gpuPerf?: GpuPerfSink;
  onBurst?(sample: BurstSample): void;
  onPhase?(phase: 'encode' | 'decode'): void;
}

const SAMPLE_RATE = 16_000;
const TOKENS_PER_SECOND = 6.5;
const BURST = 8;
const MAX_CHUNK_SECONDS = 26;
const SPLIT_SEARCH_SECONDS = 5;

export async function createTranscriber(
  sd: LazyStateDict,
  tokenizer: MoonshineTokenizer,
  opts: {
    cfg?: MoonshineConfig;
    weightDtype?: 'f32' | 'f16';
    onProgress?: (name: string, doneBytes: number, totalBytes: number) => void;
    chunking?: Partial<Pick<ChunkOpts, 'maxSeconds' | 'searchSeconds'>>;
    perf?: TranscriberPerf;
  } = {},
): Promise<Transcriber> {
  const root = defaultRoot();
  if (sd.metadata['model'] && sd.metadata['model'] !== 'moonshine') {
    throw new RunntimeError(
      'CHECKPOINT_MISMATCH',
      `moonshine weights: file is for model '${sd.metadata['model']}', expected 'moonshine'`,
    );
  }
  const cfg = opts.cfg ?? presetFromStateDict(sd);
  const perf = opts.perf;
  const burstMax = Math.max(1, Math.min(perf?.burst ?? BURST, cfg.maxPositions));
  const ex = gpuExecutor(root, { perf: perf?.gpuPerf });

  const half = opts.weightDtype === 'f16';
  const model = new MoonshineModel(cfg);
  if (half) model.half(root);
  try {
    await model.loadStateDict(sd, { root, onProgress: opts.onProgress });
  } catch (err) {
    model.dispose();
    throw err;
  }

  const actTensor = (data: Float32Array, dims: [number, number]): Value =>
    tensor(root, data, { elems: data.length, dtype: half ? 'f16' : 'f32', dims });

  const dec = buildRopeTables(cfg.maxPositions, cfg);
  const decCos = tensor(root, dec.cos, [cfg.maxPositions, cfg.headDim]);
  const decSin = tensor(root, dec.sin, [cfg.maxPositions, cfg.headDim]);

  // Preallocated per-layer caches, written in place and reused by every
  // utterance.
  const qw = cfg.heads * cfg.headDim;
  const kvInit = new Float32Array(cfg.maxPositions * qw);
  const selfKV: SelfKV[] = Array.from({ length: cfg.decLayers }, () => ({
    k: actTensor(kvInit, [cfg.maxPositions, qw]),
    v: actTensor(kvInit, [cfg.maxPositions, qw]),
  }));
  // Per-position greedy ids, read back once per burst.
  const idsBuf = tensor(root, new Float32Array(cfg.maxPositions), [cfg.maxPositions, 1]);

  const transcribeUtterance = async (audio: Float32Array): Promise<TranscribeResult> => {
    perf?.onPhase?.('encode');
    const t0 = performance.now();
    const frames = stemFrames(audio.length);
    const x = actTensor(audio, [audio.length, 1]);
    const enc = buildRopeTables(frames.conv3, cfg);
    const encCos = tensor(root, enc.cos, [frames.conv3, cfg.headDim]);
    const encSin = tensor(root, enc.sin, [frames.conv3, cfg.headDim]);

    const encOut = model.encoder.forward(model.stem.forward(x), encCos, encSin);
    const cross = model.decoder.precomputeCrossKV(encOut);
    // The K/V co-targets pin every buffer the decode loop reads.
    evalValues([encOut, ...cross.flatMap((c) => [c.k, c.v])], ex);
    // Keeps the encoder's GPU tail out of the first burst's readback wait.
    if (perf?.syncAfterEncode) await root.device.queue.onSubmittedWorkDone();
    const encodeMs = performance.now() - t0;

    const maxTokens = Math.min(
      cfg.maxPositions,
      Math.ceil((audio.length / SAMPLE_RATE) * TOKENS_PER_SECOND),
    );
    // Hand the pinned cross K/V back to the pool.
    const release = (kvs: readonly { k: Value; v: Value }[]) => {
      for (const { k, v } of kvs) {
        ex.releaseOutput(k.buffer, k.shape);
        k.markReleased();
        ex.releaseOutput(v.buffer, v.shape);
        v.markReleased();
      }
    };
    ex.releaseOutput(encOut.buffer, encOut.shape); // only the cross K/V outlive the encode
    encOut.markReleased();

    const ids: number[] = [cfg.bos];
    perf?.onPhase?.('decode');
    const t1 = performance.now();
    decode: while (ids.length < maxTokens) {
      const base = ids.length - 1; // position of the burst's first fed token
      const burst = Math.min(burstMax, maxTokens - ids.length);
      const tBuild = performance.now();
      // One DAG for the whole burst: step j's id chains into step j+1's
      // embedding on device.
      const idRows: Value[] = [];
      let x = model.embed([ids[base]!]); // burst opener: CPU-known id
      for (let j = 0; j < burst; j++) {
        const t = base + j;
        const h = model.decoder.forwardStep(
          x,
          slice(decCos, 0, t, t + 1),
          slice(decSin, 0, t, t + 1),
          cross,
          selfKV,
          t,
        );
        const idVal = model.nextTokenId(h); // [1,1] f32 greedy id
        idRows.push(writeRows(idsBuf, idVal, t));
        if (j + 1 < burst) x = model.embedFrom(idVal);
      }
      const tEval = performance.now();
      evalValues(idRows, ex);
      const tRead = performance.now();
      // One readback per burst, covering the id rows written so far.
      const data = await ex.readback(idsBuf.buffer, {
        elems: base + burst,
        dtype: 'f32',
        dims: [base + burst, 1],
      });
      perf?.onBurst?.({
        positions: burst,
        buildMs: tEval - tBuild,
        evalMs: tRead - tEval,
        readbackMs: performance.now() - tRead,
      });
      for (let j = 0; j < burst; j++) {
        const best = data[base + j]!;
        ids.push(best);
        if (best === cfg.eos) break decode;
      }
    }
    release(cross);
    return {
      ids,
      text: decodeTokens(tokenizer, ids),
      timings: { encodeMs, decodeMs: performance.now() - t1, steps: ids.length - 1 },
    };
  };

  const chunkOpts: ChunkOpts = {
    maxSeconds: opts.chunking?.maxSeconds ?? MAX_CHUNK_SECONDS,
    searchSeconds: opts.chunking?.searchSeconds ?? SPLIT_SEARCH_SECONDS,
    sampleRate: SAMPLE_RATE,
  };

  const transcribeAll = async (audio: Float32Array): Promise<TranscribeResult> => {
    const chunks = splitIntoChunks(audio, chunkOpts);
    if (chunks.length === 1) return transcribeUtterance(audio);
    const results: TranscribeResult[] = [];
    for (const chunk of chunks) results.push(await transcribeUtterance(chunk));
    return {
      ids: results.flatMap((r) => r.ids),
      text: results
        .map((r) => r.text.trim())
        .filter((t) => t.length > 0)
        .join(' '),
      timings: results.reduce(
        (sum, r) => ({
          encodeMs: sum.encodeMs + r.timings.encodeMs,
          decodeMs: sum.decodeMs + r.timings.decodeMs,
          steps: sum.steps + r.timings.steps,
        }),
        { encodeMs: 0, decodeMs: 0, steps: 0 },
      ),
    };
  };

  const dispose = () => {
    model.dispose();
    for (const v of [decCos, decSin, idsBuf, ...selfKV.flatMap((kv) => [kv.k, kv.v])]) {
      v.buffer.destroy();
    }
    ex.dispose();
  };

  // The KV caches, id buffer and rope tables are shared by every call, so
  // calls run one at a time: each waits for the previous one to settle.
  let inFlight: Promise<unknown> = Promise.resolve();
  let disposed = false;
  const transcribe = (audio: Float32Array): Promise<TranscribeResult> => {
    if (disposed)
      return Promise.reject(new RunntimeError('RESOURCE_DISPOSED', 'transcriber is disposed'));
    const job = inFlight.then(() => transcribeAll(audio));
    inFlight = job.catch(() => undefined);
    return job;
  };

  return {
    minSamples: MIN_SAMPLES,
    transcribe,
    dispose: () => {
      disposed = true;
      dispose();
    },
  };
}
