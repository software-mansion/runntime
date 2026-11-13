/** MobileNetV4-Conv-S against timm on the same photos. Level A compares
 *  the logits for timm's own model input, level B the top class
 *  createImageClassifier gives for the photo. */
import { createImageClassifier, type ImageClassifier } from '../../../src/zoo/index.ts';
import { openWeights, type ModelPath } from '../../../src/zoo/load.ts';
import { createClassifier, type Classifier } from '../../../src/zoo/mobilenetv4/classifier.ts';
import { gpu } from '../../gpu.ts';
import {
  readImage,
  readJson,
  readTensors,
  reversedWeights,
  rows,
  runParity,
  summaryCell,
  type Measured,
} from '../../parity/harness.ts';
import { cosine, maxAbsError, summarize } from '../../parity/metrics.ts';
import { MOBILENETV4_PARITY } from './parity.manifest.ts';

await gpu();

const { family } = MOBILENETV4_PARITY;
const inputs = await readJson<{ images: string[] }>(`${family}/inputs.json`);
const reference = await readTensors(`${family}/reference.safetensors`);
const images = await Promise.all(inputs.images.map(readImage));
const pixels = rows(reference.get('pixels')!, images.length);
const wantLogits = rows(reference.get('logits')!, images.length);
const wantTop1 = reference.get('top1')!;

/** One set of weights loaded twice: the bare model for level A, the public
 *  runner for level B. */
interface Loaded {
  model: Classifier;
  runner: ImageClassifier;
  dispose(): void;
}

async function load(modelPath: ModelPath): Promise<Loaded> {
  const model = await createClassifier(await openWeights(modelPath));
  const runner = await createImageClassifier({ modelPath }).catch((err: unknown) => {
    model.dispose();
    throw err;
  });
  return {
    model,
    runner,
    dispose() {
      model.dispose();
      runner.dispose();
    },
  };
}

/** Logits with their mean subtracted. Softmax ignores a shift of every
 *  logit by the same amount, so the cosine leaves it out too. */
function centered(logits: Float32Array): Float32Array {
  const mean = logits.reduce((sum, v) => sum + v, 0) / logits.length;
  return logits.map((v) => v - mean);
}

async function measure({ model, runner }: Loaded): Promise<Measured> {
  const logits: Float32Array[] = [];
  for (const input of pixels) logits.push(await model.run(input));
  const cosineDistance = summarize(
    logits.map((v, i) => 1 - cosine(centered(v), centered(wantLogits[i]!))),
  );
  const error = summarize(logits.map((v, i) => maxAbsError(v, wantLogits[i]!)));

  let missed = 0;
  for (const [i, image] of images.entries()) {
    const [top] = await runner.classify(image, { topk: 1 });
    if (top!.classId !== wantTop1[i]) missed++;
  }
  const top1Missed = missed / images.length;
  return {
    cells: {
      '1 − cos of centered logits (mean / p99 / max)': summaryCell(cosineDistance),
      'max error (mean / p99 / max)': summaryCell(error),
      'top-1 missed': `${(100 * top1Missed).toFixed(1)}%`,
    },
    metrics: { cosineDistance: cosineDistance.max, top1Missed },
  };
}

await runParity({
  manifest: MOBILENETV4_PARITY,
  load: (_config, modelPath) => load(modelPath),
  measure,
  broken: {
    label: 'one norm variance reversed',
    weights: (modelPath) => reversedWeights(modelPath, 'bn1.running_var'),
  },
});
