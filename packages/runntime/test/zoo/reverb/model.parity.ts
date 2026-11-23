/** Reverb against reverb-tts on the same sentences and voices, one reference
 *  per weight file. Level A replays every pass of the reference's takes: the
 *  session is given the grid the reference had and its best codes and their
 *  log-probabilities are compared. Level B runs the sampler on its own with
 *  the reference's noise, from the raw sentence through the text preparation,
 *  and compares the codes it ends with; the public call is held to the
 *  reference's clip length. The codec row compares runntime's int8 codec with
 *  the reference's f32 one on the reference's codes, and is not gated. */
import { Verbalizer } from 'verbalis';
import { defaultRoot, supportsF16 } from '../../../src/core/index.ts';
import { HiggsDecoder } from '../../../src/zoo/higgs/decoder.ts';
import { createTextToSpeech, type TextToSpeech } from '../../../src/zoo/index.ts';
import { fetchJson, openWeights, type ModelPath } from '../../../src/zoo/load.ts';
import {
  COARSE_CODEBOOKS,
  MASK,
  REVERB,
  SAMPLE_RATE,
  FRAME_RATE,
  type ReverbLanguage,
} from '../../../src/zoo/reverb/config.ts';
import { ReverbModel } from '../../../src/zoo/reverb/model.ts';
import { mainPassUpdate, polishFirst, polishSecond } from '../../../src/zoo/reverb/sampler.ts';
import {
  createReverbSpeaker,
  DEFAULT_STEPS,
  type ReverbSpeaker,
  type Voice,
} from '../../../src/zoo/reverb/speaker.ts';
import { createReverbTokenizer } from '../../../src/zoo/reverb/tokenizer.ts';
import { spokenInput, type TextToSpeechModel } from '../../../src/zoo/tasks/audio/textToSpeech.ts';
import { gpu } from '../../gpu.ts';
import {
  readJson,
  readTensors,
  reversedWeights,
  runParity,
  summaryCell,
  type Measured,
} from '../../parity/harness.ts';
import { cosine, maxAbsError, summarize } from '../../parity/metrics.ts';
import {
  REVERB_FP16_PARITY,
  REVERB_FP32_PARITY,
  REVERB_INT4_PARITY,
  REVERB_INT8_PARITY,
} from './parity.manifest.ts';

await gpu();

const { codebooks } = REVERB;
/** Codebooks a polish pass scores, and of those the ones its first pass fills. */
const FINE = codebooks - COARSE_CODEBOOKS;
const FIRST_POLISH = 2;
const FRAME_SAMPLES = SAMPLE_RATE / FRAME_RATE;

/** The text preparation of the docs: verbalis in the voice's language. */
const verbalize = (text: string, language: ReverbLanguage) =>
  new Verbalizer(language).process(text);

/** One set of weights, loaded one at a time: the bare speaker for the
 *  passes, then the public runner for the clip length. */
interface Loaded {
  speaker: ReverbSpeaker;
  /** Frees the speaker and loads the public runner in its place. */
  runner(): Promise<TextToSpeech>;
  dispose(): void;
}

async function load(config: TextToSpeechModel, modelPath: ModelPath): Promise<Loaded> {
  const root = defaultRoot();
  const model = new ReverbModel();
  const codec = new HiggsDecoder();
  if ((config.dtype ?? (supportsF16() ? 'f16' : 'f32')) === 'f16') {
    model.half(root);
    codec.half(root);
  }
  const tokenizer = createReverbTokenizer(await fetchJson(config.tokenizerPath!));
  const speaker = createReverbSpeaker(root, model, codec, tokenizer);
  let live: { dispose(): void } = speaker;
  try {
    await model.loadStateDict(await openWeights(modelPath), { root });
    await codec.loadStateDict(await openWeights(config.codecPath!), { root });
  } catch (err) {
    speaker.dispose();
    throw err;
  }
  return {
    speaker,
    async runner() {
      speaker.dispose();
      live = { dispose() {} };
      const runner = await createTextToSpeech({ ...config, modelPath, verbalize });
      live = runner;
      return runner;
    },
    dispose: () => live.dispose(),
  };
}

/** Counts of differing codes: those where the reference scores its two best
 *  codes further apart than the margin, and the near-ties. */
class Differences {
  count = 0;
  nearTies = 0;
  total = 0;
  /** Without a margin any differing code counts. */
  add(got: number, want: number, gap: number, margin: number | undefined) {
    this.total++;
    if (got === want) return;
    if (margin !== undefined && gap <= margin) this.nearTies++;
    else this.count++;
  }
}

/** 10·log10 of the signal's energy over the error's. */
function snr(got: Float32Array, want: Float32Array): number {
  let signal = 0;
  let error = 0;
  for (let i = 0; i < want.length; i++) {
    signal += want[i]! ** 2;
    error += (got[i]! - want[i]!) ** 2;
  }
  return 10 * Math.log10(signal / error);
}

for (const manifest of [
  REVERB_FP32_PARITY,
  REVERB_FP16_PARITY,
  REVERB_INT8_PARITY,
  REVERB_INT4_PARITY,
]) {
  const { family } = manifest;
  const inputs = await readJson<
    {
      language: ReverbLanguage;
      text: string;
      prepared: string;
      transcript: string;
      speechFrames: number | null;
      frames: number;
    }[]
  >(`${family}/inputs.json`);
  const reference = await readTensors(`${family}/reference.safetensors`);
  const takes = inputs.map((input, i) => {
    const ints = (name: string) => Int32Array.from(reference.get(`${name}.${i}`)!);
    const floats = (name: string) => reference.get(`${name}.${i}`) as Float32Array;
    const codes = ints('voice');
    const voice: Voice = {
      codes,
      frames: codes.length / codebooks,
      transcript: input.transcript,
      language: input.language,
      speechFrames: input.speechFrames ?? undefined,
    };
    return {
      ...input,
      voice,
      /** The grid before each pass, [passes, codebooks, frames]. */
      grids: ints('grids'),
      coarse: {
        best: ints('coarseBest'),
        confidence: floats('coarseConfidence'),
        gap: floats('coarseGap'),
        held: floats('coarseHeld'),
      },
      fine: {
        best: ints('fineBest'),
        confidence: floats('fineConfidence'),
        gap: floats('fineGap'),
      },
      guided: {
        best: ints('guidedBest'),
        confidence: floats('guidedConfidence'),
        gap: floats('guidedGap'),
      },
      noise: floats('noise'),
      codes: ints('codes'),
      audio: floats('audio'),
    };
  });

  const measure = async (
    loaded: Loaded,
    margins: Readonly<Record<string, number>>,
  ): Promise<Measured> => {
    const { speaker } = loaded;
    const margin = margins.logProbGap;
    const confidenceErrors: number[] = [];
    const confidenceCosine: number[] = [];
    const heldErrors: number[] = [];
    const replayed = new Differences();
    const sampled = new Differences();
    const audioCosine: number[] = [];
    const audioError: number[] = [];
    const audioSnr: number[] = [];
    /** Per take: its sampled codes against the reference's, when it has the
     *  reference's length. */
    const own: (Differences | undefined)[] = [];
    let otherFrames = 0;
    let otherTexts = 0;

    for (const take of takes) {
      const T = take.frames;
      const slots = COARSE_CODEBOOKS * T;
      const grid = (pass: number) =>
        take.grids.subarray(pass * codebooks * T, (pass + 1) * codebooks * T);

      // Level A: every pass on the grid the reference gave it.
      const replay = speaker.open(take.prepared, take.voice);
      try {
        if (replay.frames !== T) {
          throw new Error(`${family}: ${replay.frames} frames for "${take.text}", expected ${T}`);
        }
        /** Compares `count` slots of a pass from `offset` in what it returns
         *  with the reference's from `from`. */
        const compare = (
          got: { best: Int32Array; confidence: Float32Array },
          want: { best: Int32Array; confidence: Float32Array; gap: Float32Array },
          count: number,
          from = 0,
          offset = 0,
        ) => {
          confidenceCosine.push(
            1 -
              cosine(
                got.confidence.subarray(offset, offset + count),
                want.confidence.subarray(from, from + count),
              ),
          );
          for (let i = 0; i < count; i++) {
            const best = got.best[offset + i]!;
            replayed.add(best, want.best[from + i]!, want.gap[from + i]!, margin);
            if (best === want.best[from + i]) {
              confidenceErrors.push(
                Math.abs(got.confidence[offset + i]! - want.confidence[from + i]!),
              );
            }
          }
        };
        for (let step = 0; step < DEFAULT_STEPS; step++) {
          const tokens = grid(step);
          const stats = await replay.pass(tokens, { fine: false });
          compare(stats, take.coarse, slots, step * slots);
          // The log-probability of the code each slot holds.
          const held = stats.heldConfidence!;
          for (let i = 0; i < slots; i++) {
            if (tokens[i] === MASK) continue;
            heldErrors.push(Math.abs(held[i]! - take.coarse.held[step * slots + i]!));
          }
        }
        compare(await replay.pass(grid(DEFAULT_STEPS), { fine: true }), take.fine, FINE * T);
        // The guided pass decides the codebooks after the first polish
        // pass's.
        compare(
          await replay.pass(grid(DEFAULT_STEPS + 1), { fine: true, guided: true }),
          take.guided,
          (FINE - FIRST_POLISH) * T,
          0,
          FIRST_POLISH * T,
        );
      } finally {
        replay.dispose();
      }

      // Level B: the sampler on its own, from the raw sentence, with the
      // reference's noise.
      const spoken = spokenInput(take.text, take.voice, verbalize);
      if (spoken.text !== take.prepared || spoken.voice.transcript !== take.transcript) {
        otherTexts++;
      }
      const session = speaker.open(spoken.text, spoken.voice);
      try {
        if (session.frames !== T) own.push(undefined);
        else {
          const differences = new Differences();
          const tokens = new Int32Array(codebooks * T).fill(MASK);
          let drawn = 0;
          const random = () => take.noise[drawn++]!;
          for (let step = 0; step < DEFAULT_STEPS; step++) {
            const coarse = await session.pass(tokens, { fine: false });
            mainPassUpdate(tokens, T, coarse, step, DEFAULT_STEPS, random);
          }
          polishFirst(tokens, T, await session.pass(tokens, { fine: true }));
          polishSecond(tokens, T, await session.pass(tokens, { fine: true, guided: true }));
          for (let i = 0; i < tokens.length; i++) {
            differences.add(tokens[i]!, take.codes[i]!, decidingGap(take, i), margin);
          }
          own.push(differences);
        }
      } finally {
        session.dispose();
      }

      // The codec on the reference's codes.
      const audio = await speaker.decode(take.codes, T);
      audioCosine.push(1 - cosine(audio, take.audio));
      audioError.push(maxAbsError(audio, take.audio));
      audioSnr.push(snr(audio, take.audio));
    }

    // The public call, with its own noise: only its length compares.
    const runner = await loaded.runner();
    for (const [i, take] of takes.entries()) {
      const slots = codebooks * take.frames;
      const clip = await runner.synthesize(take.text, take.voice, { raw: true });
      const differences = own[i];
      sampled.total += slots;
      if (!differences || clip.length !== take.frames * FRAME_SAMPLES) {
        // A take of another length differs in every code.
        otherFrames++;
        sampled.count += slots;
      } else {
        sampled.count += differences.count;
        sampled.nearTies += differences.nearTies;
      }
    }

    const confidence = summarize(confidenceErrors);
    const held = summarize(heldErrors);
    const confidenceCosineDistance = summarize(confidenceCosine);
    const ties = (differences: Differences) =>
      margin === undefined ? '—' : `${differences.nearTies}`;
    const percent = (value: number) => `${(100 * value).toFixed(2)}%`;
    const worst = (values: number[]) => Math.max(...values).toExponential(1);
    return {
      cells: {
        'pass: log-prob 1 − cos (mean / p99 / max)': summaryCell(confidenceCosineDistance),
        'pass: log-prob error (mean / p99 / max)': summaryCell(confidence),
        'held code: log-prob error (mean / p99 / max)': summaryCell(held),
        'pass: codes differing': percent(replayed.count / replayed.total),
        'pass: near-ties, not gated': ties(replayed),
        'sampled: codes differing': percent(sampled.count / sampled.total),
        'sampled: near-ties, not gated': ties(sampled),
        'target lengths differing': `${otherFrames} / ${takes.length}`,
        'prepared texts differing': `${otherTexts} / ${takes.length}`,
        'codec: 1 − cos / max error / SNR, worst clip': `${worst(audioCosine)} / ${worst(audioError)} / ${Math.min(...audioSnr).toFixed(1)} dB`,
      },
      metrics: {
        confidenceError: confidence.max,
        heldConfidenceError: held.max,
        codeMismatches: replayed.count / replayed.total,
        tokenMismatches: sampled.count / sampled.total,
      },
    };
  };

  /** How far the reference scores the code of slot `i` of its final grid
   *  above the second best, in the pass that decided it: for codebooks 0-1
   *  the last main pass that found the slot masked. */
  function decidingGap(take: (typeof takes)[number], i: number): number {
    const T = take.frames;
    const book = Math.floor(i / T);
    const t = i % T;
    if (book >= COARSE_CODEBOOKS + FIRST_POLISH) {
      return take.guided.gap[(book - COARSE_CODEBOOKS - FIRST_POLISH) * T + t]!;
    }
    if (book >= COARSE_CODEBOOKS) return take.fine.gap[(book - COARSE_CODEBOOKS) * T + t]!;
    let pass = DEFAULT_STEPS - 1;
    while (take.grids[pass * codebooks * T + i] !== MASK) pass--;
    return take.coarse.gap[pass * COARSE_CODEBOOKS * T + i]!;
  }

  await runParity({
    manifest,
    load,
    measure,
    broken: {
      // The norm before the heads: no weight of the model is square.
      label: 'the last norm reversed',
      weights: (modelPath) => reversedWeights(modelPath, 'llm.norm.weight'),
    },
  });
}
