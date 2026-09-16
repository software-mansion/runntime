/** Speech to text task: 16 kHz mono audio in, as a clip or live from the
 *  microphone, text out. Runs Moonshine on the initRunntime() device. */

import { createResourceScope, defaultRoot, supportsF16, warmUp } from '../../../core/index.ts';
import {
  fetchJson,
  openWeights,
  throwIfAborted,
  type LoadOptions,
  type ModelPath,
} from '../../load.ts';
import { models } from '../../models.ts';
import { analyzeSegment } from '../../moonshine/segmenter.ts';
import { moonshineTokenizerAsset } from '../../moonshine/tokenizer.ts';
import { createTranscriber } from '../../moonshine/transcriber.ts';
import { createTranscriber as createStreamingTranscriber } from '../../moonshine-streaming/transcriber.ts';
import { SPEECH_SAMPLE_RATE } from './audioInput.ts';

/** New audio between two transcriptions of the sentence being spoken. */
const TICK_SAMPLES = SPEECH_SAMPLE_RATE / 2;
/** Speech this long commits even without a pause. */
const MAX_SEGMENT_SECONDS = 15;

/** The two Moonshine families. Each has its own weights and decoder. */
export type SpeechToTextArch = 'moonshine' | 'moonshine-streaming';

/** What to load. Every field is optional; a missing one comes from the
 *  family's default entry in models.speechToText. */
export interface SpeechToTextModel {
  /** The family the weights belong to. Default `moonshine`. */
  readonly arch?: SpeechToTextArch;
  /** The safetensors checkpoint. */
  readonly modelPath?: ModelPath;
  /** The tokenizer JSON: a Hugging Face tokenizer.json or our compact asset. */
  readonly tokenizerPath?: string;
  /** Precision on the GPU. `f16` runs the whole model in half precision,
   *  weights and activations, for half the memory traffic; the text stays
   *  the same. Needs a device with `shader-f16`. Default: `f16` where the
   *  device has it, `f32` elsewhere. Either loads any weight file. */
  readonly dtype?: 'f32' | 'f16';
}

/** One update of a live stream: the whole transcript so far. */
export interface SpeechStreamUpdate {
  /** The sentences the speaker finished. Each ended in a pause. */
  readonly committed: string;
  /** The sentence being spoken now. It changes until it commits. */
  readonly nonCommitted: string;
}

export interface SpeechToText {
  /** Mono samples at 16 kHz in, the spoken text out. Long clips are split
   *  on quiet moments. Clips shorter than one model frame give ''. */
  transcribe(audio: Float32Array): Promise<string>;
  /** Starts live transcription: feed samples with streamInsert, read the
   *  updates here. An update comes when the text changes; the iterator
   *  ends after streamStop or dispose. One stream at a time. */
  stream(): AsyncIterable<SpeechStreamUpdate>;
  /** Adds microphone samples, mono at 16 kHz, to the open stream. */
  streamInsert(samples: Float32Array): void;
  /** Commits what is left in the open stream and ends its iterator. */
  streamStop(): void;
  /** Frees the weights and every GPU buffer the runner holds. */
  dispose(): void;
}

/** The audio of one live stream. */
interface Session {
  /** The sentence being spoken, as inserted chunks. */
  chunks: Float32Array[];
  /** Samples in chunks. */
  length: number;
  /** Samples inserted since the last transcription. */
  fresh: number;
  stopped: boolean;
  /** Resumes the stream loop, set while it waits for audio. */
  wake: () => void;
}

/** Loads the checkpoint and the tokenizer, warms every GPU path once, and
 *  returns the runner. With no config, loads Moonshine tiny from the
 *  Hugging Face Hub. */
export async function createSpeechToText(
  config: SpeechToTextModel = {},
  opts: LoadOptions = {},
): Promise<SpeechToText> {
  const scope = createResourceScope();
  try {
    const device = defaultRoot().device;
    const arch = config.arch ?? 'moonshine';
    const defaults =
      arch === 'moonshine'
        ? models.speechToText.MOONSHINE.TINY.DEFAULT
        : models.speechToText.MOONSHINE_STREAMING.TINY.DEFAULT;
    const {
      modelPath = defaults.modelPath,
      tokenizerPath = defaults.tokenizerPath,
      dtype = supportsF16() ? 'f16' : 'f32',
    } = config;

    // One cache key per file, using the whole path so same-named files of two
    // sizes stay apart. Bump the version to invalidate.
    const cacheId = (path: ModelPath) =>
      typeof path === 'string' ? `${arch}-v1/${path.replace(/^https?:\/\//, '')}` : undefined;
    const sd = await openWeights(modelPath, {
      ...opts,
      cacheId: opts.cacheId ?? cacheId(modelPath),
    });
    const tokenizer = moonshineTokenizerAsset(
      await fetchJson(tokenizerPath, { ...opts, cacheId: cacheId(tokenizerPath) }),
    );

    throwIfAborted(opts.signal);
    const make = arch === 'moonshine' ? createTranscriber : createStreamingTranscriber;
    const transcriber = scope.track(
      await make(sd, tokenizer, { weightDtype: dtype, onProgress: opts.onProgress }),
    );

    // Compiles the kernels, and turns a GPU error into one load error.
    throwIfAborted(opts.signal);
    await warmUp(device, async () => {
      const silence = new Float32Array(SPEECH_SAMPLE_RATE);
      return Float32Array.from((await transcriber.transcribe(silence)).ids);
    });

    let disposed = false;
    // Calls run one after another: the model reuses its caches between clips.
    let queue: Promise<unknown> = Promise.resolve();
    const enqueue = (audio: Float32Array) => {
      const run = queue.then(() => transcriber.transcribe(audio));
      queue = run.catch(() => undefined);
      return run;
    };

    let live: Session | undefined;
    // Waits for new audio, transcribes the sentence from its start, and
    // commits on a pause. Audio arriving mid-transcription joins the next.
    async function* liveLoop(session: Session): AsyncIterable<SpeechStreamUpdate> {
      let last: SpeechStreamUpdate = { committed: '', nonCommitted: '' };
      try {
        while (true) {
          if (!session.stopped && session.fresh < TICK_SAMPLES) {
            await new Promise<void>((resolve) => {
              session.wake = resolve;
            });
            continue;
          }
          const final = session.stopped;
          session.fresh = 0;
          const audio = concat(session.chunks, session.length);
          const segment = analyzeSegment(audio, {
            sampleRate: SPEECH_SAMPLE_RATE,
            maxSegmentSeconds: MAX_SEGMENT_SECONDS,
          });
          let text = '';
          if (segment.hasSpeech && audio.length >= transcriber.minSamples) {
            text = (await enqueue(audio)).text.trim();
          } else if (segment.seconds >= MAX_SEGMENT_SECONDS) {
            // Silence is never transcribed, so only its last second stays.
            keepTail(session, SPEECH_SAMPLE_RATE);
          }
          let committed = last.committed;
          if (final || segment.shouldFinalize) {
            if (text) committed = committed ? `${committed} ${text}` : text;
            text = '';
            session.chunks = [];
            session.length = 0;
          }
          if (committed !== last.committed || text !== last.nonCommitted) {
            last = { committed, nonCommitted: text };
            yield last;
          }
          if (final) return;
        }
      } finally {
        if (live === session) live = undefined;
      }
    }

    return {
      async transcribe(audio) {
        if (disposed) throw new Error('speech to text is disposed');
        if (!(audio instanceof Float32Array)) {
          throw new Error('transcribe: audio must be a Float32Array');
        }
        if (audio.length < transcriber.minSamples) return '';
        return (await enqueue(audio)).text;
      },
      stream() {
        if (disposed) throw new Error('speech to text is disposed');
        if (live) throw new Error('stream: a stream is open, stop it first');
        live = { chunks: [], length: 0, fresh: 0, stopped: false, wake: () => {} };
        return liveLoop(live);
      },
      streamInsert(samples) {
        if (!live || live.stopped) throw new Error('streamInsert: no stream is open');
        if (!(samples instanceof Float32Array)) {
          throw new Error('streamInsert: samples must be a Float32Array');
        }
        // A copy: recorders reuse the array they hand out.
        live.chunks.push(samples.slice());
        live.length += samples.length;
        live.fresh += samples.length;
        if (live.fresh >= TICK_SAMPLES) live.wake();
      },
      streamStop() {
        if (!live) return;
        live.stopped = true;
        live.wake();
      },
      dispose() {
        disposed = true;
        if (live) {
          // Nothing left to transcribe, so the stream ends without the GPU.
          live.chunks = [];
          live.length = 0;
          live.stopped = true;
          live.wake();
        }
        scope.dispose();
      },
    };
  } catch (err) {
    scope.dispose();
    throw err;
  }
}

/** Joins the chunks into one array of `length` samples. */
function concat(chunks: Float32Array[], length: number): Float32Array {
  const out = new Float32Array(length);
  let at = 0;
  for (const c of chunks) {
    out.set(c, at);
    at += c.length;
  }
  return out;
}

/** Drops chunks from the front until about `samples` remain. */
function keepTail(session: Session, samples: number): void {
  while (session.chunks.length > 1 && session.length - session.chunks[0]!.length >= samples) {
    session.length -= session.chunks.shift()!.length;
  }
}
