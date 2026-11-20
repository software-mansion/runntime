/** Moonshine Streaming tiny against transformers on the same speech. Level A
 *  compares the encoder states the decoder reads and the logits of every
 *  decode step, the decoder fed the reference's greedy tokens one at a time
 *  through the cached step the runner decodes with. Level B compares the
 *  token each step picks, through the call the runner picks with, and the
 *  text createSpeechToText gives. */
import { defaultRoot, tensor } from '../../../src/core/index.ts';
import { createSpeechToText, type SpeechToText } from '../../../src/zoo/index.ts';
import { openWeights, type ModelPath } from '../../../src/zoo/load.ts';
import { presetFromStateDict } from '../../../src/zoo/moonshine-streaming/config.ts';
import { decoderView } from '../../../src/zoo/moonshine-streaming/decoder.ts';
import { frontendFrames } from '../../../src/zoo/moonshine-streaming/frontend.ts';
import { MoonshineStreamingModel } from '../../../src/zoo/moonshine-streaming/model.ts';
import { DECODE_POSITIONS, padToFrames } from '../../../src/zoo/moonshine-streaming/transcriber.ts';
import type { SpeechToTextModel } from '../../../src/zoo/tasks/audio/speechToText.ts';
import { gpu } from '../../gpu.ts';
import { runParity, transposedWeights } from '../../parity/harness.ts';
import {
  decodeState,
  decodeSteps,
  measureSpeech,
  readReference,
  type Decode,
  type Forced,
} from '../moonshine/parity.ts';
import { MOONSHINE_STREAMING_TINY_PARITY } from './parity.manifest.ts';

await gpu();

/** One set of weights loaded twice: the bare model for level A and the
 *  tokens, the public runner for the text. */
interface Loaded extends Decode {
  model: MoonshineStreamingModel;
  half: boolean;
  runner: SpeechToText;
  dispose(): void;
}

async function load(config: SpeechToTextModel, modelPath: ModelPath): Promise<Loaded> {
  const root = defaultRoot();
  const sd = await openWeights(modelPath);
  const model = new MoonshineStreamingModel(presetFromStateDict(sd));
  const half = config.dtype === 'f16';
  try {
    if (half) model.half(root);
    await model.loadStateDict(sd, { root });
    const runner = await createSpeechToText({ ...config, modelPath });
    const state = decodeState(root, decoderView(model.cfg), DECODE_POSITIONS, half);
    return {
      model,
      half,
      runner,
      ...state,
      dispose() {
        model.dispose();
        runner.dispose();
        state.dispose();
      },
    };
  } catch (err) {
    model.dispose();
    throw err;
  }
}

/** The encoder states of `samples` as the decoder reads them, through the
 *  adapter, and every decode step on them. The samples go in as the
 *  transcriber's frames. */
async function teacherForced(
  loaded: Loaded,
  samples: Float32Array,
  tokens: Int32Array,
): Promise<Forced> {
  const { cfg } = loaded.model;
  const { encoder, decoder } = loaded.model.model;
  const audio = padToFrames(samples, cfg.frameLen);
  const x = tensor(defaultRoot(), audio, {
    elems: audio.length,
    dtype: loaded.half ? 'f16' : 'f32',
    dims: [frontendFrames(audio.length, cfg.frameLen).input, cfg.frameLen],
  });
  try {
    return await decodeSteps(loaded, decoder.adapt(encoder.forward(x)), tokens);
  } finally {
    x.buffer.destroy();
  }
}

const measure = measureSpeech(
  await readReference(MOONSHINE_STREAMING_TINY_PARITY.family),
  teacherForced,
);
await runParity({
  manifest: MOONSHINE_STREAMING_TINY_PARITY,
  load,
  measure,
  broken: {
    // An encoder weight: the decoder reads the encoder states, so it
    // shapes the logits and the text too.
    label: 'one encoder weight transposed',
    weights: (modelPath) =>
      transposedWeights(modelPath, 'model.encoder.layers.0.self_attn.k_proj.weight'),
  },
});
