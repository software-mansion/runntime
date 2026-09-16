/** The GPU side of yolo26: weights in, one preprocessed image to raw head levels
 *  out. Shared by the detect/pose/segment runners and the plugin. */

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
  type Value,
} from '../../core/index.ts';
import { YOLO26_CONFIG, YOLO26_SCALES, scaleChannels, type Yolo26Variant } from './config.ts';
import { Yolo26Model, type Yolo26Task } from './model.ts';
import type { ProtoData, RawLevel } from './pipeline.ts';

/** What one forward pass gives back. `proto` comes with the segment head
 *  only: the mask prototypes every detection's coefficients combine. */
export interface DetectorOutput {
  levels: RawLevel[];
  proto?: ProtoData;
}

export interface Detector {
  readonly task: Yolo26Task;
  readonly inputSize: number;
  readonly numClasses: number;
  run(pixels: Float32Array): Promise<DetectorOutput>;
  dispose(): void;
}

/** Head channels past the 4 box distances and the classes: 3 per keypoint
 *  for pose, one coefficient per mask prototype for segment. */
export const POSE_KEYPOINTS = 17;
export const SEGMENT_MASKS = 32;
const extraChannels = (task: Yolo26Task) =>
  task === 'pose' ? 3 * POSE_KEYPOINTS : task === 'segment' ? SEGMENT_MASKS : 0;

/** The head the checkpoint was trained for, from the layers only that head
 *  has. */
export function taskFromStateDict(sd: LazyStateDict): Yolo26Task {
  if (sd.tensors.has('model.23.one2one_cv4_kpts.0.weight')) return 'pose';
  if (sd.tensors.has('model.23.proto.cv3.conv.weight')) return 'segment';
  return 'detect';
}

const mismatch = (message: string) => new Error(`yolo26 weights: ${message}`);

const STEM_CHANNELS = Object.fromEntries(
  (Object.keys(YOLO26_SCALES) as Yolo26Variant[]).map((v) => [
    v,
    scaleChannels(64, YOLO26_SCALES[v]),
  ]),
) as Record<Yolo26Variant, number>;

export function variantFromStateDict(sd: LazyStateDict): Yolo26Variant {
  const stem = sd.tensors.get('model.0.conv.weight');
  if (!stem) throw mismatch('not a yolo26 checkpoint (no model.0.conv.weight)');
  const channels = stem.shape[0];
  if (channels === STEM_CHANNELS.m)
    return sd.tensors.has('model.2.m.1.cv1.conv.weight') ? 'l' : 'm';
  const found = (Object.keys(STEM_CHANNELS) as Yolo26Variant[]).find(
    (v) => STEM_CHANNELS[v] === channels,
  );
  if (!found) throw mismatch(`not a yolo26 checkpoint (stem has ${channels} channels)`);
  return found;
}

export async function createDetector(
  sd: LazyStateDict,
  opts: {
    /** The head to build. Default: read from the weights. */
    task?: Yolo26Task;
    variant?: Yolo26Variant;
    inputSize?: number;
    onProgress?: (name: string, doneBytes: number, totalBytes: number) => void;
  } = {},
): Promise<Detector> {
  const root = defaultRoot();
  const { strides } = YOLO26_CONFIG;
  const size = opts.inputSize ?? YOLO26_CONFIG.inputSize;
  if (!Number.isInteger(size) || size <= 0 || size % 32 !== 0) {
    throw new Error(`inputSize: ${size} is not a positive multiple of 32`);
  }
  const found = variantFromStateDict(sd);
  const variant = opts.variant ?? found;
  if (variant !== found) {
    throw mismatch(`the file holds yolo26${found}, expected yolo26${variant}`);
  }
  const foundTask = taskFromStateDict(sd);
  const task = opts.task ?? foundTask;
  if (task !== foundTask) {
    throw mismatch(`the file holds a ${foundTask} checkpoint, expected ${task}`);
  }
  const model = new Yolo26Model<Yolo26Task>(variant, { task });
  const ex = gpuExecutor(root);
  const buffer = uploadF32(root, new Float32Array(3 * size * size));
  const dispose = () => {
    model.dispose();
    buffer.destroy();
    ex.dispose();
  };
  try {
    await model.loadStateDict(sd, { onProgress: opts.onProgress });
    const input = materialized(tensor3d(3, size, size), buffer);
    // Targets are the three levels, then the prototypes for segment.
    const cap = ex.captureFrame(() => {
      const out = model.forward(input);
      const targets: Value[] = Array.isArray(out) ? [...out] : [...out.levels, out.proto];
      evalValues(targets, ex);
      return targets;
    });
    // Each level is [4 + classes + extra, h, w].
    const numClasses = cap.targets[0]!.shape.dims![0]! - 4 - extraChannels(task);
    if (numClasses <= 0) throw mismatch(`the ${task} head has too few channels`);
    const toOutput = (outs: readonly Float32Array[]): DetectorOutput => {
      const levels = strides.map((stride, i) => {
        const p = cap.targets[i]!;
        return { data: outs[i]!, h: p.shape.dims![1]!, w: p.shape.dims![2]!, stride };
      });
      const proto = cap.targets[3];
      return proto
        ? { levels, proto: { data: outs[3]!, h: proto.shape.dims![1]!, w: proto.shape.dims![2]! } }
        : { levels };
    };
    let queue: Promise<unknown> = Promise.resolve();
    const run = (pixels: Float32Array) => {
      const job = queue.then(async () => {
        writeF32(root, buffer, pixels);
        const outs = ex.readbackManyOnSubmit(cap.targets);
        cap.replay();
        return toOutput(await outs);
      });
      queue = job.catch(() => undefined);
      return job;
    };
    await warmUp(
      root.device,
      async () => (await run(new Float32Array(3 * size * size))).levels[0]!.data,
    );
    return { task, inputSize: size, numClasses, run, dispose };
  } catch (err) {
    dispose();
    throw err;
  }
}
