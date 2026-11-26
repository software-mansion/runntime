/** The Privacy Filter against opf on the same texts. Level A compares the
 *  logits of every token for the reference's own token ids with opf's on
 *  the hosted file's weights, level B the spans createPrivacyFilter finds in
 *  the text with opf's on the upstream weights. */
import { describe, expect, it } from 'vitest';
import { toArray } from '../../../src/core/index.ts';
import { createPrivacyFilter, type PrivacyFilter } from '../../../src/zoo/index.ts';
import { openWeights } from '../../../src/zoo/load.ts';
import { loadPrivacyFilterModel } from '../../../src/zoo/privacy-filter/detector.ts';
import type { EagerModel } from '../../../src/zoo/privacy-filter/model.ts';
import { createTokenizer } from '../../../src/zoo/privacy-filter/tokenizer.ts';
import { gpu } from '../../gpu.ts';
import {
  loadBoth,
  readJson,
  readTensors,
  reversedWeights,
  rows,
  runParity,
  summaryCell,
  type Loaded,
  type Measured,
} from '../../parity/harness.ts';
import { cosine, maxAbsError, spanF1, summarize, topK, type Span } from '../../parity/metrics.ts';
import { PRIVACY_FILTER_PARITY } from './parity.manifest.ts';

await gpu();

const { family } = PRIVACY_FILTER_PARITY;
const { texts } = await readJson<{ texts: string[] }>(`${family}/inputs.json`);
const { spans: wantSpans } = await readJson<{ spans: Span[][] }>(`${family}/outputs.json`);
const reference = await readTensors(`${family}/reference.safetensors`);
const ids = texts.map((_, i) => Array.from(reference.get(`ids.${i}`)!));
/** The logits of each text, a row per token: opf's on the hosted file's
 *  weights, and on the upstream weights. */
const logits = (name: string) =>
  texts.map((_, i) => rows(reference.get(`${name}.${i}`)!, ids[i]!.length));
const wantLogits = logits('logits_hosted');
const upstreamLogits = logits('logits');

async function measure({ model, runner }: Loaded<EagerModel, PrivacyFilter>): Promise<Measured> {
  const errors: number[] = [];
  const upstreamErrors: number[] = [];
  const cosineDistances: number[] = [];
  let labelMismatches = 0;
  const spans: Span[][] = [];
  for (const [i, text] of texts.entries()) {
    const want = wantLogits[i]!;
    const got = rows(await toArray(model.forward(ids[i]!)), want.length);
    const error = (from: readonly Float32Array[]) =>
      Math.max(...got.map((row, t) => maxAbsError(row, from[t]!)));
    errors.push(error(want));
    upstreamErrors.push(error(upstreamLogits[i]!));
    for (const [t, row] of got.entries()) {
      cosineDistances.push(1 - cosine(row, want[t]!));
      if (topK(row, 1)[0] !== topK(want[t]!, 1)[0]) labelMismatches++;
    }
    spans.push(await runner.detect(text));
  }
  const error = summarize(errors);
  const cosineDistance = summarize(cosineDistances);
  const f1 = spanF1(spans, wantSpans);
  const count = (of: readonly (readonly Span[])[]) => of.reduce((sum, s) => sum + s.length, 0);
  return {
    cells: {
      'max logit error of a text (mean / p99 / max)': summaryCell(error),
      '1 − cos of a token (mean / p99 / max)': summaryCell(cosineDistance),
      'tokens with another best label': `${labelMismatches} of ${cosineDistances.length}`,
      'not gated: max logit error of a text from the upstream weights (mean / p99 / max)':
        summaryCell(summarize(upstreamErrors)),
      'spans (found / reference)': `${count(spans)} / ${count(wantSpans)}`,
      'span F1': f1.toFixed(4),
    },
    metrics: {
      error: error.max,
      cosineDistance: cosineDistance.max,
      labelMismatches: labelMismatches / cosineDistances.length,
      spanF1Distance: 1 - f1,
    },
  };
}

describe(`${family} tokenizer`, () => {
  it("gives the reference's token ids", () => {
    const tokenizer = createTokenizer();
    expect(texts.map((text) => tokenizer.encode(text))).toEqual(ids);
  });
});

await runParity({
  manifest: PRIVACY_FILTER_PARITY,
  load: (_config, modelPath) =>
    loadBoth(
      async () => (await loadPrivacyFilterModel(await openWeights(modelPath))).model,
      () => createPrivacyFilter({ modelPath }),
    ),
  measure,
  broken: {
    label: 'the label weights reversed',
    weights: (modelPath) => reversedWeights(modelPath, 'unembedding.weight'),
  },
});
