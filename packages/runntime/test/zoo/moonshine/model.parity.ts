/** Moonshine tiny and base against transformers on the same speech. Level A
 *  compares the encoder states and the logits of every decode step, the
 *  decoder fed the reference's greedy tokens one at a time through the cached
 *  step the runner decodes with. Level B compares the token each step picks,
 *  through the call the runner picks with, and the text createSpeechToText
 *  gives. */
import {
  defaultRoot,
  evalValues,
  gpuExecutor,
  slice,
  tensor,
  type Value,
} from '../../../src/core/index.ts';
import { createSpeechToText, type SpeechToText } from '../../../src/zoo/index.ts';
import { openWeights, type ModelPath } from '../../../src/zoo/load.ts';
import { presetFromStateDict } from '../../../src/zoo/moonshine/config.ts';
import { stemFrames } from '../../../src/zoo/moonshine/convStem.ts';
import type { SelfKV } from '../../../src/zoo/moonshine/decoder.ts';
import { MoonshineModel } from '../../../src/zoo/moonshine/model.ts';
import { buildRopeTables } from '../../../src/zoo/moonshine/rope.ts';
import type { SpeechToTextModel } from '../../../src/zoo/tasks/audio/speechToText.ts';
import { gpu } from '../../gpu.ts';
import {
  readJson,
  readTensors,
  rows,
  runParity,
  summaryCell,
  transposedWeights,
  type Measured,
} from '../../parity/harness.ts';
import { cosine, maxAbsError, rankMismatches, summarize, wer } from '../../parity/metrics.ts';
import { MOONSHINE_BASE_PARITY, MOONSHINE_TINY_PARITY } from './parity.manifest.ts';

await gpu();

/** One set of weights loaded twice: the bare model for level A and the
 *  tokens, the public runner for the text. The executor, the decoder rope
 *  tables and the self-attention caches live as long as the model, as in
 *  the transcriber. */
interface Loaded {
  model: MoonshineModel;
  half: boolean;
  ex: ReturnType<typeof gpuExecutor>;
  decCos: Value;
  decSin: Value;
  selfKV: SelfKV[];
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
    const ex = gpuExecutor(root);
    const dec = buildRopeTables(cfg.maxPositions, cfg);
    const decCos = tensor(root, dec.cos, [cfg.maxPositions, cfg.headDim]);
    const decSin = tensor(root, dec.sin, [cfg.maxPositions, cfg.headDim]);
    const qw = cfg.heads * cfg.headDim;
    const cache = () =>
      tensor(root, new Float32Array(cfg.maxPositions * qw), {
        elems: cfg.maxPositions * qw,
        dtype: half ? 'f16' : 'f32',
        dims: [cfg.maxPositions, qw],
      });
    const selfKV: SelfKV[] = Array.from({ length: cfg.decLayers }, () => ({
      k: cache(),
      v: cache(),
    }));
    return {
      model,
      half,
      ex,
      decCos,
      decSin,
      selfKV,
      runner,
      dispose() {
        model.dispose();
        runner.dispose();
        for (const v of [decCos, decSin, ...selfKV.flatMap((kv) => [kv.k, kv.v])]) {
          v.buffer.destroy();
        }
        ex.dispose();
      },
    };
  } catch (err) {
    model.dispose();
    throw err;
  }
}

/** The encoder states of `samples`, and for every decode step its logits
 *  and the token it picks. The decoder takes `tokens` one at a time through
 *  forwardStep and the caches, and picks with nextTokenId, as the
 *  transcriber does. The caches need no reset between clips: step t writes
 *  row t before attention reads rows 0 to t. */
async function teacherForced(
  { model, half, ex, decCos, decSin, selfKV }: Loaded,
  samples: Float32Array,
  tokens: Int32Array,
): Promise<{ states: Float32Array; logits: Float32Array[]; picked: number[] }> {
  const root = defaultRoot();
  const { cfg } = model;
  /** Hands the buffer of an evaluated value back to the pool. */
  const release = (value: Value) => {
    ex.releaseOutput(value.buffer, value.shape);
    value.markReleased();
  };
  const read = async (value: Value) => {
    const data = await ex.readback(value.buffer, value.shape);
    release(value);
    return data;
  };

  const frames = stemFrames(samples.length).conv3;
  const enc = buildRopeTables(frames, cfg);
  const uploads = [
    tensor(root, samples, {
      elems: samples.length,
      dtype: half ? 'f16' : 'f32',
      dims: [samples.length, 1],
    }),
    tensor(root, enc.cos, [frames, cfg.headDim]),
    tensor(root, enc.sin, [frames, cfg.headDim]),
  ] as const;
  try {
    const [x, encCos, encSin] = uploads;
    const encOut = model.model.encoder.forward(x, encCos, encSin);
    const cross = model.model.decoder.precomputeCrossKV(encOut);
    evalValues([encOut, ...cross.flatMap((c) => [c.k, c.v])], ex);
    const states = await read(encOut);

    const logits: Float32Array[] = [];
    const picked: number[] = [];
    // The last token is the end of the text: nothing follows it.
    for (let t = 0; t < tokens.length - 1; t++) {
      const h = model.model.decoder.forwardStep(
        model.embed([tokens[t]!]),
        slice(decCos, 0, t, t + 1),
        slice(decSin, 0, t, t + 1),
        cross,
        selfKV,
        t,
      );
      const stepLogits = model.logitsLast(h);
      const stepToken = model.nextTokenId(h);
      evalValues([stepLogits, stepToken], ex);
      logits.push(await read(stepLogits));
      picked.push((await read(stepToken))[0]!);
    }
    for (const { k, v } of cross) {
      release(k);
      release(v);
    }
    return { states, logits, picked };
  } finally {
    for (const v of uploads) v.buffer.destroy();
  }
}

for (const manifest of [MOONSHINE_TINY_PARITY, MOONSHINE_BASE_PARITY]) {
  const { family } = manifest;
  const inputs = await readJson<{ clips: string[]; transcripts: string[] }>(
    `${family}/inputs.json`,
  );
  const { texts } = await readJson<{ texts: string[] }>(`${family}/outputs.json`);
  const reference = await readTensors(`${family}/reference.safetensors`);
  const clips = inputs.clips.map((_, i) => {
    const tokens = Int32Array.from(reference.get(`tokens.${i}`)!);
    return {
      samples: Float32Array.from(reference.get(`samples.${i}`)!),
      tokens,
      states: reference.get(`encoder.${i}`)!,
      logits: rows(reference.get(`logits.${i}`)!, tokens.length - 1),
    };
  });
  const referenceWer = wer(texts, inputs.transcripts);

  const measure = async (
    loaded: Loaded,
    margins: Readonly<Record<string, number>>,
  ): Promise<Measured> => {
    const encoderCosine: number[] = [];
    const encoderError: number[] = [];
    const logitsCosine: number[] = [];
    const logitsError: number[] = [];
    const got: string[] = [];
    let steps = 0;
    let mismatches = 0;
    let nearTies = 0;
    const margin = margins.logitGap;
    for (const clip of clips) {
      const { states, logits, picked } = await teacherForced(loaded, clip.samples, clip.tokens);
      encoderCosine.push(1 - cosine(states, clip.states));
      encoderError.push(maxAbsError(states, clip.states));
      logits.forEach((step, t) => {
        const want = clip.logits[t]!;
        logitsCosine.push(1 - cosine(step, want));
        logitsError.push(maxAbsError(step, want));
        const token = clip.tokens[t + 1]!;
        // Without a margin any differing token is a mismatch.
        const { count, nearTies: near } =
          margin === undefined
            ? { count: Number(picked[t] !== token), nearTies: 0 }
            : rankMismatches([picked[t]!], [token], want, margin);
        mismatches += count;
        nearTies += near;
      });
      steps += logits.length;
      // Level B: the text of the public runner, decoding on its own.
      got.push(await loaded.runner.transcribe(clip.samples));
    }
    const encoderCosineDistance = summarize(encoderCosine);
    const encoderErrors = summarize(encoderError);
    const logitsCosineDistance = summarize(logitsCosine);
    const logitsErrors = summarize(logitsError);
    const differing = got.filter((text, i) => text !== texts[i]).length;
    const gotWer = wer(got, inputs.transcripts);
    const percent = (value: number) => `${(100 * value).toFixed(2)}%`;
    return {
      cells: {
        'encoder: 1 − cos (mean / p99 / max)': summaryCell(encoderCosineDistance),
        'encoder: max error (mean / p99 / max)': summaryCell(encoderErrors),
        'logits: 1 − cos (mean / p99 / max)': summaryCell(logitsCosineDistance),
        'logits: max error (mean / p99 / max)': summaryCell(logitsErrors),
        'tokens differing': percent(mismatches / steps),
        'near-ties, not gated': margin === undefined ? '—' : `${nearTies}`,
        'texts differing': `${differing} / ${clips.length}`,
        'WER (reference)': `${percent(gotWer)} (${percent(referenceWer)})`,
      },
      metrics: {
        encoderCosineDistance: encoderCosineDistance.max,
        encoderError: encoderErrors.max,
        logitsCosineDistance: logitsCosineDistance.max,
        logitsError: logitsErrors.max,
        tokenMismatches: mismatches / steps,
        textMismatches: differing / clips.length,
        werDelta: gotWer - referenceWer,
      },
    };
  };

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
