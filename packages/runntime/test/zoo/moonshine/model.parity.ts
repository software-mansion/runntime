/** Moonshine tiny and base against transformers on the same speech. Level A
 *  compares the encoder states and the logits of every decode step, the
 *  decoder fed the reference's greedy tokens one at a time through the cached
 *  step the runner decodes with. Level B compares the token each step picks,
 *  through the call the runner picks with, and the text createSpeechToText
 *  gives. */
import { defaultRoot, tensor } from '../../../src/core/index.ts';
import { createSpeechToText, type SpeechToText } from '../../../src/zoo/index.ts';
import { openWeights, type ModelPath } from '../../../src/zoo/load.ts';
import { presetFromStateDict } from '../../../src/zoo/moonshine/config.ts';
import { stemFrames } from '../../../src/zoo/moonshine/convStem.ts';
import { MoonshineModel } from '../../../src/zoo/moonshine/model.ts';
import { buildRopeTables } from '../../../src/zoo/moonshine/rope.ts';
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
} from './parity.ts';
import { MOONSHINE_BASE_PARITY, MOONSHINE_TINY_PARITY } from './parity.manifest.ts';

await gpu();

/** One set of weights loaded twice: the bare model for level A and the
 *  tokens, the public runner for the text. */
interface Loaded extends Decode {
  model: MoonshineModel;
  half: boolean;
  runner: SpeechToText;
  dispose(): void;
}

async function load(config: SpeechToTextModel, modelPath: ModelPath): Promise<Loaded> {
  const root = defaultRoot();
  const sd = await openWeights(modelPath);
  const model = new MoonshineModel(presetFromStateDict(sd));
  const { cfg } = model;
  const half = config.dtype === 'f16';
  try {
    if (half) model.half(root);
    await model.loadStateDict(sd, { root });
    const runner = await createSpeechToText({ ...config, modelPath });
    const state = decodeState(root, cfg, cfg.maxPositions, half);
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

/** The encoder states of `samples` and every decode step on them. */
async function teacherForced(
  loaded: Loaded,
  samples: Float32Array,
  tokens: Int32Array,
): Promise<Forced> {
  const root = defaultRoot();
  const { cfg } = loaded.model;
  const frames = stemFrames(samples.length).conv3;
  const enc = buildRopeTables(frames, cfg);
  const uploads = [
    tensor(root, samples, {
      elems: samples.length,
      dtype: loaded.half ? 'f16' : 'f32',
      dims: [samples.length, 1],
    }),
    tensor(root, enc.cos, [frames, cfg.headDim]),
    tensor(root, enc.sin, [frames, cfg.headDim]),
  ] as const;
  try {
    const [x, encCos, encSin] = uploads;
    return await decodeSteps(loaded, loaded.model.model.encoder.forward(x, encCos, encSin), tokens);
  } finally {
    for (const v of uploads) v.buffer.destroy();
  }
}

for (const manifest of [MOONSHINE_TINY_PARITY, MOONSHINE_BASE_PARITY]) {
  const measure = measureSpeech(await readReference(manifest.family), teacherForced);
  await runParity({
    manifest,
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
}
