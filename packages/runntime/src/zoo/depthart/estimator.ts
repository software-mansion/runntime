/** The GPU side of DepthART depth estimation: weights in, a function from
 *  one normalized image to a relative depth map out. Shared by the task
 *  runner and the transformers.js plugin. */

import {
  defaultRoot,
  evalValues,
  gpuExecutor,
  materialized,
  resizeBilinear2d,
  RunntimeError,
  tensor3d,
  toChw,
  toHwc4,
  uploadF32,
  warmUp,
  writeF32,
  type F32Buffer,
  type LazyStateDict,
  type RunntimeExecutor,
  type Value,
} from '../../core/index.ts';
import { DEPTHART_B_448, DEPTHART_S_448, DepthartModel, type DepthartConfig } from './model.ts';

export type DepthartVariant = 'b' | 's';

export const DEPTHART_CONFIGS: Record<DepthartVariant, DepthartConfig> = {
  b: DEPTHART_B_448,
  s: DEPTHART_S_448,
};

export interface Estimator {
  readonly variant: DepthartVariant;
  readonly inputSize: number;
  run(pixels: Float32Array, h: number, w: number): Promise<Float32Array>;
  dispose(): void;
}

const mismatch = (message: string) =>
  new RunntimeError('CHECKPOINT_MISMATCH', `depthart weights: ${message}`);

export function variantFromStateDict(sd: LazyStateDict): DepthartVariant {
  const width = sd.tensors.get('pretrained.network.1.proj.c.weight')?.shape[0];
  const found = (Object.keys(DEPTHART_CONFIGS) as DepthartVariant[]).find(
    (v) => DEPTHART_CONFIGS[v].dims[1] === width,
  );
  if (!found) {
    throw mismatch(
      `not a DepthART checkpoint (first downsample has ${width ?? 'no'} channels, expected 96 or 64)`,
    );
  }
  return found;
}

interface Frame {
  buffer: F32Buffer;
  target: Value;
  replay: () => void;
  ex: RunntimeExecutor;
}

export async function createEstimator(
  sd: LazyStateDict,
  opts: {
    variant?: DepthartVariant;
    onProgress?: (name: string, doneBytes: number, totalBytes: number) => void;
  } = {},
): Promise<Estimator> {
  const root = defaultRoot();
  const found = variantFromStateDict(sd);
  const variant = opts.variant ?? found;
  if (variant !== found) {
    throw mismatch(`the file holds depthart-${found}, expected depthart-${variant}`);
  }
  const model = new DepthartModel(DEPTHART_CONFIGS[variant]);
  const size = model.cfg.inputSize;
  const frames = new Map<string, Frame>();
  const dispose = () => {
    model.dispose();
    for (const f of frames.values()) {
      f.buffer.destroy();
      f.ex.dispose();
    }
    frames.clear();
  };
  try {
    await model.loadStateDict(sd, { onProgress: opts.onProgress });

    const record = (h: number, w: number): Frame => {
      const ex = gpuExecutor(root);
      const buffer = uploadF32(root, new Float32Array(3 * h * w));
      const input = materialized(tensor3d(3, h, w), buffer);
      const cap = ex.captureFrame(() => {
        const fitted =
          h === size && w === size
            ? input
            : toChw(
                resizeBilinear2d(toHwc4(input), { outH: size, outW: size, mode: 'halfPixel' }),
                'f32',
              );
        const depth = model.forward(fitted);
        evalValues([depth], ex);
        return depth;
      });
      return { buffer, target: cap.targets, replay: cap.replay, ex };
    };
    const frame = (h: number, w: number): Frame => {
      const key = `${h}x${w}`;
      let f = frames.get(key);
      if (!f) {
        f = record(h, w);
        frames.set(key, f);
      }
      return f;
    };

    let queue: Promise<unknown> = Promise.resolve();
    const run = (pixels: Float32Array, h: number, w: number) => {
      const job = queue.then(async () => {
        const f = frame(h, w);
        writeF32(root, f.buffer, pixels);
        const out = f.ex.readbackOnSubmit(f.target);
        f.replay();
        return out;
      });
      queue = job.catch(() => undefined);
      return job;
    };
    await warmUp(root.device, () => run(new Float32Array(3 * size * size), size, size));
    return { variant, inputSize: size, run, dispose };
  } catch (err) {
    dispose();
    throw err;
  }
}
