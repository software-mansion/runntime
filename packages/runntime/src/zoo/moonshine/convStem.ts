/** Frame counts of the Moonshine conv stem (MoonshineEncoder's three convs):
 *  raw 16 kHz mono audio [L, 1] → encoder frames [T, d] (384× downsample,
 *  ~41.7 frames/s):
 *      conv1 k=127 s=64 → conv2 k=7 s=3 → conv3 k=3 s=2, valid padding. */

export const K1 = 127;
export const S1 = 64;
export const K2 = 7;
export const S2 = 3;
export const K3 = 3;
export const S3 = 2;

const convOut = (t: number, k: number, s: number) => Math.floor((t - k) / s) + 1;

export const MIN_SAMPLES = 895;

export function stemFrames(samples: number): { conv1: number; conv2: number; conv3: number } {
  if (!Number.isInteger(samples) || samples < MIN_SAMPLES) {
    throw new Error(
      `stemFrames: need ≥ ${MIN_SAMPLES} samples (~56 ms at 16 kHz) for one encoder frame, got ${samples}`,
    );
  }
  const t1 = convOut(samples, K1, S1);
  const t2 = convOut(t1, K2, S2);
  return { conv1: t1, conv2: t2, conv3: convOut(t2, K3, S3) };
}
