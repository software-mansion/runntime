/** What the Moonshine and Moonshine Streaming parity tests share: the
 *  reference of a family, the teacher-forced decode steps and what a variant
 *  is measured on. */
import {
  evalValues,
  gpuExecutor,
  slice,
  tensor,
  type defaultRoot,
  type Value,
} from '../../../src/core/index.ts';
import type { SpeechToText } from '../../../src/zoo/index.ts';
import type { MoonshineConfig } from '../../../src/zoo/moonshine/config.ts';
import type { MoonshineDecoder, SelfKV } from '../../../src/zoo/moonshine/decoder.ts';
import { buildRopeTables } from '../../../src/zoo/moonshine/rope.ts';
import { readJson, readTensors, rows, summaryCell, type Measured } from '../../parity/harness.ts';
import { cosine, maxAbsError, rankMismatches, summarize, wer } from '../../parity/metrics.ts';

/** What the decode steps run on: the model, and the executor, the decoder
 *  rope tables and the self-attention caches, which live as long as the
 *  model, as in the transcriber. */
export interface Decode {
  model: {
    embed(ids: readonly number[]): Value;
    logitsLast(h: Value): Value;
    nextTokenId(h: Value): Value;
    model: { decoder: MoonshineDecoder };
  };
  ex: ReturnType<typeof gpuExecutor>;
  decCos: Value;
  decSin: Value;
  selfKV: SelfKV[];
}

/** What `Decode` holds besides the model, for the decoder `dec` and
 *  `positions` decode positions, with f16 caches when `half`, and what frees
 *  it. */
export function decodeState(
  root: ReturnType<typeof defaultRoot>,
  dec: MoonshineConfig,
  positions: number,
  half: boolean,
): Omit<Decode, 'model'> & { dispose(): void } {
  const ex = gpuExecutor(root);
  const tables = buildRopeTables(positions, dec);
  const decCos = tensor(root, tables.cos, [positions, dec.headDim]);
  const decSin = tensor(root, tables.sin, [positions, dec.headDim]);
  const qw = dec.heads * dec.headDim;
  const cache = () =>
    tensor(root, new Float32Array(positions * qw), {
      elems: positions * qw,
      dtype: half ? 'f16' : 'f32',
      dims: [positions, qw],
    });
  const selfKV: SelfKV[] = Array.from({ length: dec.decLayers }, () => ({
    k: cache(),
    v: cache(),
  }));
  return {
    ex,
    decCos,
    decSin,
    selfKV,
    dispose() {
      for (const v of [decCos, decSin, ...selfKV.flatMap((kv) => [kv.k, kv.v])]) {
        v.buffer.destroy();
      }
      ex.dispose();
    },
  };
}

/** The encoder states of a clip, and for every decode step its logits and
 *  the token it picks. */
export interface Forced {
  states: Float32Array;
  logits: Float32Array[];
  picked: number[];
}

/** `encoded`, the encoder states the decoder reads, and every decode step
 *  on them. The decoder takes `tokens` one at a time through forwardStep and
 *  the caches, and picks with nextTokenId, as the transcriber does. The
 *  caches need no reset between clips: step t writes row t before attention
 *  reads rows 0 to t. */
export async function decodeSteps(
  { model, ex, decCos, decSin, selfKV }: Decode,
  encoded: Value,
  tokens: Int32Array,
): Promise<Forced> {
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

  const { decoder } = model.model;
  const cross = decoder.precomputeCrossKV(encoded);
  evalValues([encoded, ...cross.flatMap((c) => [c.k, c.v])], ex);
  const states = await read(encoded);

  const logits: Float32Array[] = [];
  const picked: number[] = [];
  // The last token is the end of the text: nothing follows it.
  for (let t = 0; t < tokens.length - 1; t++) {
    const h = decoder.forwardStep(
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
}

/** The saved reference of `family`: per clip its samples, the greedy tokens,
 *  the encoder states and the logits of every step, then the texts and the
 *  transcripts. */
export async function readReference(family: string) {
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
  return { clips, texts, transcripts: inputs.transcripts };
}

/** Measures a loaded variant against `reference`: `teacherForced` gives
 *  level A and the tokens of a clip, the public runner its text. */
export function measureSpeech<L extends { runner: SpeechToText }>(
  { clips, texts, transcripts }: Awaited<ReturnType<typeof readReference>>,
  teacherForced: (loaded: L, samples: Float32Array, tokens: Int32Array) => Promise<Forced>,
) {
  const referenceWer = wer(texts, transcripts);
  return async (loaded: L, margins: Readonly<Record<string, number>>): Promise<Measured> => {
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
    const gotWer = wer(got, transcripts);
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
}
