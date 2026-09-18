/** Streaming Moonshine transcriber: the v1 decode loop over the v2 frontend,
 *  encoder and adapter. Long clips split on quiet points into per-utterance
 *  chunks; each utterance is zero-padded to whole 80-sample frames.
 *
 *  Decoding runs in bursts: within a burst each step's argmax id feeds the next
 *  step's embedding on device, so the GPU runs the whole burst with no CPU
 *  round-trip. A burst may overshoot EOS; those ids are discarded.
 *
 *  Rope tables and the per-layer K/V caches are allocated once at
 *  DECODE_POSITIONS rows and written in place, so utterances reuse them and
 *  transcribe() calls must stay sequential. */

import {
  defaultRoot,
  evalValues,
  gpuExecutor,
  RunntimeError,
  slice,
  tensor,
  writeRows,
  type LazyStateDict,
  type Value,
} from '../../core/index.ts';
import { splitIntoChunks, type ChunkOpts } from '../moonshine/chunking.ts';
import { buildRopeTables } from '../moonshine/rope.ts';
import { decodeTokens, type MoonshineTokenizer } from '../moonshine/tokenizer.ts';
import type { SelfKV } from '../moonshine/decoder.ts';
import type { TranscribeResult, Transcriber, TranscriberPerf } from '../moonshine/transcriber.ts';
import { presetFromStateDict, type MoonshineStreamingConfig } from './config.ts';
import { frontendFrames } from './frontend.ts';
import { MoonshineStreamingModel, decoderView } from './model.ts';

export type { TranscribeResult, Transcriber };

const SAMPLE_RATE = 16_000;
const TOKENS_PER_SECOND = 6.5;
const BURST = 8;
const DECODE_POSITIONS = 256;
const MAX_CHUNK_SECONDS = 30;
const SPLIT_SEARCH_SECONDS = 5;

export async function createTranscriber(
  sd: LazyStateDict,
  tokenizer: MoonshineTokenizer,
  opts: {
    cfg?: MoonshineStreamingConfig;
    weightDtype?: 'f32' | 'f16';
    onProgress?: (name: string, doneBytes: number, totalBytes: number) => void;
    chunking?: Partial<Pick<ChunkOpts, 'maxSeconds' | 'searchSeconds'>>;
    perf?: TranscriberPerf;
  } = {},
): Promise<Transcriber> {
  const root = defaultRoot();
  if (sd.metadata['model'] && sd.metadata['model'] !== 'moonshine-streaming') {
    throw new RunntimeError(
      'CHECKPOINT_MISMATCH',
      `moonshine-streaming weights: file is for model '${sd.metadata['model']}', ` +
        `expected 'moonshine-streaming'`,
    );
  }
  const cfg = opts.cfg ?? presetFromStateDict(sd);
  const dec = decoderView(cfg);
  const perf = opts.perf;
  const burstMax = Math.max(1, Math.min(perf?.burst ?? BURST, DECODE_POSITIONS));
  const ex = gpuExecutor(root, { perf: perf?.gpuPerf });

  const half = opts.weightDtype === 'f16';
  const model = new MoonshineStreamingModel(cfg);
  if (half) model.half(root);
  try {
    await model.loadStateDict(sd, { root, onProgress: opts.onProgress });
  } catch (err) {
    model.dispose();
    throw err;
  }

  const actTensor = (data: Float32Array, dims: [number, number]): Value =>
    tensor(root, data, { elems: data.length, dtype: half ? 'f16' : 'f32', dims });

  const tables = buildRopeTables(DECODE_POSITIONS, dec);
  const decCos = tensor(root, tables.cos, [DECODE_POSITIONS, dec.headDim]);
  const decSin = tensor(root, tables.sin, [DECODE_POSITIONS, dec.headDim]);

  // Preallocated per-layer caches, written in place and reused by every
  // utterance.
  const qw = dec.heads * dec.headDim;
  const kvInit = new Float32Array(DECODE_POSITIONS * qw);
  const selfKV: SelfKV[] = Array.from({ length: dec.decLayers }, () => ({
    k: actTensor(kvInit, [DECODE_POSITIONS, qw]),
    v: actTensor(kvInit, [DECODE_POSITIONS, qw]),
  }));
  // Per-position greedy ids, read back once per burst.
  const idsBuf = tensor(root, new Float32Array(DECODE_POSITIONS), [DECODE_POSITIONS, 1]);

  const padToFrames = (audio: Float32Array): Float32Array => {
    if (audio.length === 0) throw new RunntimeError('INVALID_ARGUMENT', 'transcribe: empty audio');
    const frames = Math.ceil(audio.length / cfg.frameLen);
    if (frames * cfg.frameLen === audio.length) return audio;
    const padded = new Float32Array(frames * cfg.frameLen);
    padded.set(audio);
    return padded;
  };

  const transcribeUtterance = async (rawAudio: Float32Array): Promise<TranscribeResult> => {
    perf?.onPhase?.('encode');
    const t0 = performance.now();
    const audio = padToFrames(rawAudio);
    const frames = frontendFrames(audio.length, cfg.frameLen);
    const x = actTensor(audio, [frames.input, cfg.frameLen]);

    const adapterOut = model.adapter.forward(model.encoder.forward(model.frontend.forward(x)));
    const cross = model.decoder.precomputeCrossKV(adapterOut);
    // The K/V co-targets pin every buffer the decode loop reads.
    evalValues([adapterOut, ...cross.flatMap((c) => [c.k, c.v])], ex);
    // Keeps the encoder's GPU tail out of the first burst's readback wait.
    if (perf?.syncAfterEncode) await root.device.queue.onSubmittedWorkDone();
    const encodeMs = performance.now() - t0;

    const maxTokens = Math.min(
      DECODE_POSITIONS,
      Math.ceil((rawAudio.length / SAMPLE_RATE) * TOKENS_PER_SECOND),
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
    ex.releaseOutput(adapterOut.buffer, adapterOut.shape); // only the cross K/V outlive the encode
    adapterOut.markReleased();

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
      let x1 = model.embed([ids[base]!]); // burst opener: CPU-known id
      for (let j = 0; j < burst; j++) {
        const t = base + j;
        const h = model.decoder.forwardStep(
          x1,
          slice(decCos, 0, t, t + 1),
          slice(decSin, 0, t, t + 1),
          cross,
          selfKV,
          t,
        );
        const idVal = model.nextTokenId(h); // [1,1] f32 greedy id
        idRows.push(writeRows(idsBuf, idVal, t));
        if (j + 1 < burst) x1 = model.embedFrom(idVal);
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
    minSamples: cfg.frameLen,
    transcribe,
    dispose: () => {
      disposed = true;
      dispose();
    },
  };
}
