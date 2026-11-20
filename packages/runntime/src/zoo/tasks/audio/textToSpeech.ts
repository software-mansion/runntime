/** Text to speech task: text in, spoken in the voice of a reference clip,
 *  mono audio out. Runs Reverb and the Higgs Audio v2 codec decoder on the
 *  initRunntime() device. A voice is a clip's codec codes, its transcript
 *  and its language; createVoiceMaker makes one from a recording with the
 *  codec's encoder. */

import {
  createResourceScope,
  defaultRoot,
  inGpuErrorScopes,
  RunntimeError,
  supportsF16,
  warmUp,
} from '../../../core/index.ts';
import {
  FRAME_RATE,
  LANGUAGES,
  MASK,
  MIN_VOICE_SECONDS,
  REVERB,
  SAMPLE_RATE,
  type ReverbLanguage,
} from '../../reverb/config.ts';
import { ReverbModel } from '../../reverb/model.ts';
import {
  createReverbSpeaker,
  DEFAULT_STEPS,
  type ReverbGenerateOptions,
  type Voice,
} from '../../reverb/speaker.ts';
import { speechFrames, trimSilence } from '../../reverb/text.ts';
import { createReverbTokenizer } from '../../reverb/tokenizer.ts';
import { asLoadError, checkpointMismatch, rethrowRunError } from '../../errors.ts';
import { HiggsDecoder } from '../../higgs/decoder.ts';
import { createHiggsEncodeCore } from '../../higgs/encodeCore.ts';
import { HiggsEncoder } from '../../higgs/encoder.ts';
import { resample } from '../../higgs/resample.ts';
import {
  combinedProgress,
  fetchJson,
  openWeights,
  throwIfAborted,
  type LoadOptions,
  type ModelPath,
} from '../../load.ts';
import { models } from '../../models.ts';

export type { ReverbLanguage } from '../../reverb/config.ts';
export type { Voice } from '../../reverb/speaker.ts';

/** What to load. Every field is optional; a missing path comes from
 *  models.textToSpeech.REVERB.DEFAULT. */
export interface TextToSpeechModel {
  /** The Reverb checkpoint. The weight format is read from the file. */
  readonly modelPath?: ModelPath;
  /** The Higgs Audio v2 tokenizer checkpoint. createTextToSpeech reads its
   *  decoder, createVoiceMaker its encoder. */
  readonly codecPath?: ModelPath;
  /** The Hugging Face tokenizer.json. */
  readonly tokenizerPath?: string;
  /** Precision on the GPU of the math and the float weights. Needs
   *  a device with `shader-f16` for `f16`. Default: `f16` where the device
   *  has it, `f32` elsewhere. Either loads any weight file. */
  readonly dtype?: 'f32' | 'f16';
  /** Rewrites the text and the voice's transcript as they are read aloud,
   *  for example numbers as words:
   *  `(text, language) => new Verbalizer(language).process(text)` from
   *  `verbalis`. Default: none. */
  readonly verbalize?: (text: string, language: ReverbLanguage) => string;
}

/** Options of synthesize(). Every field is optional. */
export interface SynthesizeOptions extends ReverbGenerateOptions {
  /** Keeps the clip as decoded, without the silence trim. */
  readonly raw?: boolean;
}

/** A loaded text to speech model, from createTextToSpeech(). Calls run one
 *  at a time, in order. */
export interface TextToSpeech {
  /** Sample rate of the audio synthesize() returns. */
  readonly sampleRate: number;
  /** `text` spoken in `voice`, mono samples at `sampleRate`. Throws
   *  INVALID_ARGUMENT for a text too long for one pass on this device. */
  synthesize(text: string, voice: Voice, options?: SynthesizeOptions): Promise<Float32Array>;
  /** Frees the weights and every GPU buffer the runner holds. */
  dispose(): void;
}

/** One cache key per file, using the whole path: the Reverb and codec files are
 *  both model.safetensors. Bump the version to invalidate. */
const cacheId = (path: ModelPath) =>
  typeof path === 'string' ? `reverb-v1/${path.replace(/^https?:\/\//, '')}` : undefined;

const REVERB_HUB = 'https://huggingface.co/software-mansion/Reverb-TTS/';
/** Hugging Face counts a repo's downloads by requests for this file, and
 *  the runner reads model.safetensors instead. */
const REVERB_DOWNLOAD_COUNTER = `${REVERB_HUB}resolve/main/model/safetensors/fp16/config.json`;

/** The text and a copy of `voice` with both texts passed through
 *  `verbalize` in the voice's language, or both as given without it. Throws
 *  INVALID_ARGUMENT for an empty result. */
export function spokenInput(
  text: string,
  voice: Voice,
  verbalize: TextToSpeechModel['verbalize'],
): { text: string; voice: Voice } {
  if (!verbalize) return { text, voice };
  const { codes, frames, transcript, language } = voice;
  const spoken = (input: string, what: string) => {
    const out = verbalize(input, language);
    if (typeof out !== 'string' || out.trim() === '') {
      throw new RunntimeError(
        'INVALID_ARGUMENT',
        `synthesize: verbalize must return a non-empty string for the ${what}`,
      );
    }
    return out;
  };
  return {
    text: spoken(text, 'text'),
    voice: { codes, frames, language, transcript: spoken(transcript, 'voice transcript') },
  };
}

/** Frames of the warm-up voice and of the text it speaks. */
const WARM_UP_FRAMES = 8;

/** Throws INVALID_ARGUMENT for an empty transcript or a language Reverb does
 *  not speak. `where` starts the message. */
function checkTranscript(where: string, transcript: string, language: ReverbLanguage): void {
  if (typeof transcript !== 'string' || transcript.trim() === '') {
    throw new RunntimeError('INVALID_ARGUMENT', `${where} needs a non-empty transcript`);
  }
  if (!LANGUAGES.includes(language)) {
    throw new RunntimeError(
      'INVALID_ARGUMENT',
      `${where} language must be one of ${LANGUAGES.join(', ')}, got ${language}`,
    );
  }
}

/** Throws INVALID_ARGUMENT unless `voice` holds [codebooks, frames] codes,
 *  none of them the mask, at least MIN_VOICE_SECONDS long, with a transcript
 *  and one of Reverb's languages. */
function checkVoice(voice: Voice): void {
  const bad = (why: string) => new RunntimeError('INVALID_ARGUMENT', `synthesize: voice ${why}`);
  const { codes, frames, transcript, language } = voice;
  checkTranscript('synthesize: voice', transcript, language);
  if (!(codes instanceof Int32Array) || !Number.isInteger(frames) || frames < 1) {
    throw bad('needs codes as an Int32Array and frames >= 1');
  }
  if (codes.length !== REVERB.codebooks * frames) {
    throw bad(`codes must be [${REVERB.codebooks}, frames], ${REVERB.codebooks * frames} values`);
  }
  if (frames < MIN_VOICE_SECONDS * FRAME_RATE) {
    throw bad(
      `is ${(frames / FRAME_RATE).toFixed(1)} s, shorter than the ${MIN_VOICE_SECONDS} s Reverb needs`,
    );
  }
  if (codes.some((c) => !(c >= 0 && c < MASK))) throw bad(`codes must be in [0, ${MASK})`);
  const { speechFrames } = voice;
  if (speechFrames !== undefined && !(Number.isInteger(speechFrames) && speechFrames >= 1)) {
    throw bad('speechFrames must be an integer >= 1');
  }
}

/** Loads Reverb, its codec decoder and tokenizer, warms the GPU, and returns
 *  the runner. With no config, loads Reverb from the Hugging Face Hub. */
export async function createTextToSpeech(
  config: TextToSpeechModel = {},
  opts: LoadOptions = {},
): Promise<TextToSpeech> {
  const scope = createResourceScope();
  try {
    const root = defaultRoot();
    const defaults = models.textToSpeech.REVERB.DEFAULT;
    const {
      modelPath = defaults.modelPath,
      codecPath = defaults.codecPath,
      tokenizerPath = defaults.tokenizerPath,
      dtype = supportsF16() ? 'f16' : 'f32',
      verbalize,
    } = config;
    if (verbalize !== undefined && typeof verbalize !== 'function') {
      throw new RunntimeError(
        'INVALID_ARGUMENT',
        'createTextToSpeech: verbalize must be a function',
      );
    }
    if (typeof modelPath === 'string' && modelPath.startsWith(REVERB_HUB)) {
      fetch(REVERB_DOWNLOAD_COUNTER, { method: 'HEAD' }).catch(() => {});
    }
    const sd = await openWeights(modelPath, {
      ...opts,
      cacheId: opts.cacheId ?? cacheId(modelPath),
    });
    const codecSd = await openWeights(codecPath, { ...opts, cacheId: cacheId(codecPath) });
    const tokenizer = createReverbTokenizer(
      await fetchJson(tokenizerPath, { ...opts, cacheId: cacheId(tokenizerPath) }),
    );

    throwIfAborted(opts.signal);
    const model = new ReverbModel();
    const codec = new HiggsDecoder();
    if (dtype === 'f16') {
      model.half(root);
      codec.half(root);
    }
    const speaker = scope.track(createReverbSpeaker(root, model, codec, tokenizer));
    // The files load at once; progress counts their bytes together.
    const track = combinedProgress(opts.onProgress);
    await Promise.all([
      model.loadStateDict(sd, { root, onProgress: track('model') }),
      codec.loadStateDict(codecSd, { root, onProgress: track('codec') }),
    ]);

    // Compiles the kernels, and turns a GPU error into one load error.
    throwIfAborted(opts.signal);
    await warmUp(root.device, async () => {
      const voice: Voice = {
        codes: new Int32Array(REVERB.codebooks * WARM_UP_FRAMES),
        frames: WARM_UP_FRAMES,
        transcript: 'a',
        language: 'en',
      };
      const { tokens, frames } = await speaker.generate('a', voice, { steps: 1 });
      return speaker.decode(tokens, frames);
    });

    let disposed = false;
    // Calls run one after another: they share the executors and their pools.
    let queue: Promise<unknown> = Promise.resolve();
    return {
      sampleRate: SAMPLE_RATE,

      async synthesize(text, voice, options = {}) {
        if (disposed) throw new RunntimeError('RESOURCE_DISPOSED', 'text to speech is disposed');
        if (typeof text !== 'string' || text.trim() === '') {
          throw new RunntimeError(
            'INVALID_ARGUMENT',
            'synthesize: text must be a non-empty string',
          );
        }
        checkVoice(voice);
        const { steps = DEFAULT_STEPS, seed, raw = false } = options;
        if (!Number.isInteger(steps) || steps < 1) {
          throw new RunntimeError('INVALID_ARGUMENT', `synthesize: steps ${steps} must be >= 1`);
        }
        if (seed !== undefined && !Number.isInteger(seed)) {
          throw new RunntimeError(
            'INVALID_ARGUMENT',
            `synthesize: seed ${seed} must be an integer`,
          );
        }
        const input = spokenInput(text, voice, verbalize);
        const run = queue.then(async () => {
          const { tokens, frames } = await speaker.generate(input.text, input.voice, {
            steps,
            seed,
          });
          const audio = await speaker.decode(tokens, frames);
          return raw ? audio : trimSilence(audio, SAMPLE_RATE);
        });
        queue = run.catch(() => undefined);
        return run.catch(rethrowRunError);
      },

      dispose() {
        disposed = true;
        scope.dispose();
      },
    };
  } catch (err) {
    scope.dispose();
    throw asLoadError(err, opts.signal);
  }
}

/** A tensor only a Higgs Audio v2 codec file with its encoder has. */
const HIGGS_ENCODER_TENSOR = 'semantic_model.feature_projection.projection.weight';

/** Length of the voice maker's warm-up recording. */
const WARM_UP_SECONDS = 2;

/** What fromRecording() needs to know about the recording. */
export interface FromRecordingOptions {
  /** Sample rate of the samples, in Hz. */
  readonly sampleRate: number;
  /** What the recording says, word for word. */
  readonly transcript: string;
  /** The recording's language. */
  readonly language: ReverbLanguage;
}

/** A loaded voice maker, from createVoiceMaker(). Calls run one at a time,
 *  in order. */
export interface VoiceMaker {
  /** The shortest recording fromRecording() takes, in seconds. */
  readonly minSeconds: number;
  /** The longest recording fromRecording() takes on this device, in
   *  seconds. */
  readonly maxSeconds: number;
  /** The voice of a mono recording of one speaker, for synthesize(). Throws
   *  INVALID_ARGUMENT for a recording shorter than minSeconds or longer than
   *  maxSeconds. */
  fromRecording(samples: Float32Array, options: FromRecordingOptions): Promise<Voice>;
  /** Frees the weights and every GPU buffer the runner holds. */
  dispose(): void;
}

/** Loads the voice encoder from `config.codecPath`, warms the GPU and
 *  returns the runner. Other fields are not used. With no config, loads
 *  Reverb's codec from the Hugging Face Hub. The encoder runs in f32 on every
 *  device. */
export async function createVoiceMaker(
  config: TextToSpeechModel = {},
  opts: LoadOptions = {},
): Promise<VoiceMaker> {
  const scope = createResourceScope();
  try {
    const root = defaultRoot();
    const { codecPath = models.textToSpeech.REVERB.DEFAULT.codecPath } = config;
    // The same cache key createTextToSpeech uses for the codec, so both
    // runners share it.
    const sd = await openWeights(codecPath, { ...opts, cacheId: cacheId(codecPath) });
    if (!sd.tensors.has(HIGGS_ENCODER_TENSOR)) {
      throw checkpointMismatch(
        'createVoiceMaker',
        'the codecPath file holds no voice encoder this runner reads',
      );
    }

    throwIfAborted(opts.signal);
    const encoder = new HiggsEncoder();
    const core = scope.track(createHiggsEncodeCore(root, encoder));
    await encoder.loadStateDict(sd, { root, onProgress: opts.onProgress });

    // Compiles the kernels and turns a GPU error into a load error. Two
    // seconds take the same matmul and attention routes as a real recording.
    // Any code is valid, so warmUp's output check does not fit here.
    throwIfAborted(opts.signal);
    await inGpuErrorScopes(root.device, 'pipeline warm-up', () =>
      core.encode(new Float32Array(WARM_UP_SECONDS * encoder.cfg.sampleRate)),
    );

    const rate = encoder.cfg.sampleRate;
    const maxSamples = core.maxFrames * encoder.hop;
    let disposed = false;
    // Calls run one after another: they share the executor and its pool.
    let queue: Promise<unknown> = Promise.resolve();
    return {
      minSeconds: MIN_VOICE_SECONDS,
      maxSeconds: maxSamples / rate,

      async fromRecording(samples, options) {
        if (disposed) throw new RunntimeError('RESOURCE_DISPOSED', 'voice maker is disposed');
        const { sampleRate, transcript, language } = options;
        if (!(samples instanceof Float32Array) || samples.length === 0) {
          throw new RunntimeError(
            'INVALID_ARGUMENT',
            'fromRecording: samples must be a non-empty Float32Array',
          );
        }
        if (!Number.isInteger(sampleRate) || sampleRate <= 0) {
          throw new RunntimeError(
            'INVALID_ARGUMENT',
            `fromRecording: sampleRate ${sampleRate} must be a positive integer`,
          );
        }
        checkTranscript('fromRecording:', transcript, language);
        // resample() gives ceil(length · to / from) samples.
        const length = Math.ceil((samples.length * rate) / sampleRate);
        if (length < MIN_VOICE_SECONDS * rate) {
          throw new RunntimeError(
            'INVALID_ARGUMENT',
            `fromRecording: the recording is ${(length / rate).toFixed(1)} s, shorter than the ${MIN_VOICE_SECONDS} s Reverb needs`,
          );
        }
        if (length > maxSamples) {
          throw new RunntimeError(
            'INVALID_ARGUMENT',
            `fromRecording: the recording is ${(length / rate).toFixed(1)} s, longer than the ${(maxSamples / rate).toFixed(1)} s this device takes at once`,
          );
        }
        const run = queue.then(async () => {
          const audio = resample(samples, sampleRate, rate);
          const { codes, frames } = await core.encode(audio);
          // Measured on the clip padded to whole frames, as the reverb-tts package does.
          const padded = new Float32Array(frames * Math.round(rate / FRAME_RATE));
          padded.set(audio.subarray(0, padded.length));
          return { codes, frames, transcript, language, speechFrames: speechFrames(padded, rate) };
        });
        queue = run.catch(() => undefined);
        return run.catch(rethrowRunError);
      },

      dispose() {
        disposed = true;
        scope.dispose();
      },
    };
  } catch (err) {
    scope.dispose();
    throw asLoadError(err, opts.signal);
  }
}
