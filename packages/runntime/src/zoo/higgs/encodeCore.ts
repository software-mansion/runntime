/** The GPU side of the Higgs Audio v2 codec encoder: a loaded encoder in, a
 *  function from 24 kHz mono samples to codes [codebooks, frames] out. */

import type { TgpuRoot } from 'typegpu';
import { gpuExecutor, matrix, tensor, toArray, type Value } from '../../core/index.ts';
import type { HiggsEncoder } from './encoder.ts';

export interface HiggsEncodeCore {
  /** The most frames one encode takes on this device. */
  readonly maxFrames: number;
  /** Codes [codebooks, frames], codebook-major, of 24 kHz mono `samples`. */
  encode(samples: Float32Array): Promise<{ codes: Int32Array; frames: number }>;
  dispose(): void;
}

/** Runs a loaded `encoder` on its own executor. The caller checks the
 *  length against maxFrames. */
export function createHiggsEncodeCore(root: TgpuRoot, encoder: HiggsEncoder): HiggsEncodeCore {
  const maxFrames = encoder.maxFrames(root.device.limits.maxStorageBufferBindingSize);
  // Lengths change from call to call, so each encode ends with a trim: the
  // pool keeps only the buffers of the last length.
  const ex = gpuExecutor(root);
  return {
    maxFrames,

    async encode(samples) {
      const { acoustic, semantic, frames } = encoder.inputs(samples);
      let a: Value | undefined;
      let s: Value | undefined;
      let rows: Value | undefined;
      try {
        a = tensor(root, acoustic, matrix(acoustic.length, 1));
        s = tensor(root, semantic, matrix(semantic.length, 1));
        rows = encoder.forward(a, s);
        return { codes: encoder.codes(await toArray(rows, ex), frames), frames };
      } finally {
        if (rows) ex.releaseValues([rows]);
        a?.buffer.destroy();
        s?.buffer.destroy();
        ex.trim();
      }
    },

    dispose() {
      ex.dispose();
      encoder.dispose();
    },
  };
}
