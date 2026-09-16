/** The GPU side of MobileNetV4 image classification: weights in, a function
 *  from one normalized image to the class logits out. Used by the
 *  transformers.js plugin. */

import {
  defaultRoot,
  evalValues,
  gpuExecutor,
  materialized,
  tensor3d,
  uploadF32,
  warmUp,
  writeF32,
  type LazyStateDict,
} from '../../core/index.ts';
import { MOBILENETV4_CONV_S, type Mnv4Config } from './config.ts';
import { MobileNetV4Model } from './model.ts';

export interface Classifier {
  readonly inputSize: number;
  readonly numClasses: number;
  run(pixels: Float32Array): Promise<Float32Array>;
  dispose(): void;
}

const mismatch = (message: string) => new Error(`mobilenetv4 weights: ${message}`);

export function assertCheckpoint(sd: LazyStateDict, cfg: Mnv4Config): void {
  const stem = sd.tensors.get('conv_stem.weight');
  if (!stem) throw mismatch('not a timm MobileNetV4 checkpoint (no conv_stem.weight)');
  if (stem.shape[0] !== cfg.stemOut) {
    throw mismatch(`stem has ${stem.shape[0]} channels, expected ${cfg.stemOut}`);
  }
  const head = sd.tensors.get('classifier.weight');
  if (!head) throw mismatch('no classifier.weight');
  if (head.shape[0] !== cfg.numClasses || head.shape[1] !== cfg.headHidden) {
    throw mismatch(
      `classifier is [${head.shape}], expected [${cfg.numClasses}, ${cfg.headHidden}]`,
    );
  }
}

export async function createClassifier(
  sd: LazyStateDict,
  opts: {
    cfg?: Mnv4Config;
    onProgress?: (name: string, doneBytes: number, totalBytes: number) => void;
  } = {},
): Promise<Classifier> {
  const root = defaultRoot();
  const cfg = opts.cfg ?? MOBILENETV4_CONV_S;
  assertCheckpoint(sd, cfg);
  const model = new MobileNetV4Model(cfg, { root });
  const size = cfg.inputSize;
  const ex = gpuExecutor(root);
  const buffer = uploadF32(root, new Float32Array(3 * size * size));
  let disposed = false;
  const dispose = () => {
    disposed = true;
    model.dispose();
    buffer.destroy();
    ex.dispose();
  };
  try {
    await model.loadStateDict(sd, { onProgress: opts.onProgress });
    const input = materialized(tensor3d(3, size, size), buffer);
    const cap = ex.captureFrame(() => {
      const logits = model.forward(input);
      evalValues([logits], ex);
      return logits;
    });
    let queue: Promise<unknown> = Promise.resolve();
    const run = (pixels: Float32Array) => {
      if (disposed) return Promise.reject(new Error('classifier is disposed'));
      const job = queue.then(() => {
        writeF32(root, buffer, pixels);
        const out = ex.readbackOnSubmit(cap.targets);
        cap.replay();
        return out;
      });
      queue = job.catch(() => undefined);
      return job;
    };
    await warmUp(root.device, () => run(new Float32Array(3 * size * size)));
    return { inputSize: size, numClasses: cfg.numClasses, run, dispose };
  } catch (err) {
    dispose();
    throw err;
  }
}
