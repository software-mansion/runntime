/** DepthART S and B against the upstream model on the same photos. Level A
 *  compares the depth map for the reference's own model input, level B the
 *  map createDepthEstimator gives for the photo. Upstream's infer_image.py
 *  resizes its map to the photo's size; both sides are compared before
 *  that, at the model's output size. */
import { createDepthEstimator, type DepthEstimator } from '../../../src/zoo/index.ts';
import { openWeights, type ModelPath } from '../../../src/zoo/load.ts';
import { createEstimator, type Estimator } from '../../../src/zoo/depthart/estimator.ts';
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
import { cosine, summarize } from '../../parity/metrics.ts';
import { DEPTHART_B_PARITY, DEPTHART_S_PARITY } from './parity.manifest.ts';

await gpu();

/** One set of weights loaded twice: the bare model for level A, the public
 *  runner for level B. */
interface Loaded {
  model: Estimator;
  runner: DepthEstimator;
  dispose(): void;
}

async function load(modelPath: ModelPath): Promise<Loaded> {
  const model = await createEstimator(await openWeights(modelPath));
  const runner = await createDepthEstimator({ modelPath }).catch((err: unknown) => {
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

/** A map with its mean subtracted: relative depth has no fixed zero. */
function centered(map: Float32Array): Float32Array {
  const mean = map.reduce((sum, v) => sum + v, 0) / map.length;
  return map.map((v) => v - mean);
}

/** How far `got` is from `want`: the cosine distance of the centered maps,
 *  and the largest and the 99th percentile error as shares of the range of
 *  `want`. */
function compare(got: Float32Array, want: Float32Array) {
  let min = Infinity;
  let max = -Infinity;
  for (const v of want) {
    min = Math.min(min, v);
    max = Math.max(max, v);
  }
  const errors = got.map((v, i) => Math.abs(v - want[i]!)).sort();
  return {
    cosineDistance: 1 - cosine(centered(got), centered(want)),
    maxError: errors[errors.length - 1]! / (max - min),
    p99Error: errors[Math.ceil(0.99 * errors.length) - 1]! / (max - min),
  };
}

for (const manifest of [DEPTHART_S_PARITY, DEPTHART_B_PARITY]) {
  const { family } = manifest;
  const inputs = await readJson<{ images: string[] }>(`${family}/inputs.json`);
  const reference = await readTensors(`${family}/reference.safetensors`);
  const images = await Promise.all(inputs.images.map(readImage));
  const pixels = rows(reference.get('pixels')!, images.length);
  // runntime returns the reference's map negated: smaller is nearer.
  const want = rows(reference.get('depth')!, images.length).map((map) => map.map((v) => -v));

  const measure = async ({ model, runner }: Loaded): Promise<Measured> => {
    const { inputSize } = model;
    const fromPixels: ReturnType<typeof compare>[] = [];
    const fromImage: ReturnType<typeof compare>[] = [];
    for (const [i, image] of images.entries()) {
      const map = await model.run(pixels[i]!, inputSize, inputSize);
      fromPixels.push(compare(map, want[i]!));
      fromImage.push(compare((await runner.estimateDepth(image)).data, want[i]!));
    }
    const summary = (
      of: readonly ReturnType<typeof compare>[],
      key: keyof ReturnType<typeof compare>,
    ) => summarize(of.map((compared) => compared[key]));
    const cosineDistance = summary(fromPixels, 'cosineDistance');
    const imageCosineDistance = summary(fromImage, 'cosineDistance');
    return {
      cells: {
        '1 − cos of centered maps (mean / p99 / max)': summaryCell(cosineDistance),
        'max error / range (mean / p99 / max)': summaryCell(summary(fromPixels, 'maxError')),
        'p99 error / range (mean / p99 / max)': summaryCell(summary(fromPixels, 'p99Error')),
        'from the photo: 1 − cos (mean / p99 / max)': summaryCell(imageCosineDistance),
        'from the photo: max error / range (mean / p99 / max)': summaryCell(
          summary(fromImage, 'maxError'),
        ),
        'from the photo: p99 error / range (mean / p99 / max)': summaryCell(
          summary(fromImage, 'p99Error'),
        ),
      },
      metrics: {
        cosineDistance: cosineDistance.max,
        imageCosineDistance: imageCosineDistance.max,
      },
    };
  };

  await runParity({
    manifest,
    load: (_config, modelPath) => load(modelPath),
    measure,
    broken: {
      label: 'one norm variance reversed',
      weights: (modelPath) => reversedWeights(modelPath, 'pretrained.norm0.running_var'),
    },
  });
}
