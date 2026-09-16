/** CPU pre/post for the depth demo. */

export function preprocessRgbaDepthart(
  rgba: Uint8ClampedArray,
  size: number,
  mean: readonly [number, number, number],
  std: readonly [number, number, number],
  out?: Float32Array,
): Float32Array {
  const hw = size * size;
  out ??= new Float32Array(3 * hw);
  for (let c = 0; c < 3; c++) {
    const m = mean[c]!;
    const inv = 1 / std[c]!;
    for (let i = 0; i < hw; i++) {
      out[c * hw + i] = (rgba[i * 4 + c]! / 255 - m) * inv;
    }
  }
  return out;
}

const TURBO = (() => {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    const t2 = t * t;
    const t3 = t2 * t;
    const t4 = t3 * t;
    const t5 = t4 * t;
    lut[i * 3] =
      255 *
      (0.13572138 +
        4.6153926 * t -
        42.66032258 * t2 +
        132.13108234 * t3 -
        152.94239396 * t4 +
        59.28637943 * t5);
    lut[i * 3 + 1] =
      255 *
      (0.09140261 +
        2.19418839 * t +
        4.84296658 * t2 -
        14.18503333 * t3 +
        4.27729857 * t4 +
        2.82956604 * t5);
    lut[i * 3 + 2] =
      255 *
      (0.1066733 +
        12.64194608 * t -
        60.58204836 * t2 +
        110.36276771 * t3 -
        89.90310912 * t4 +
        27.34824973 * t5);
  }
  return lut;
})();

export function depthToRgba(depth: Float32Array, out?: Uint8ClampedArray): Uint8ClampedArray {
  out ??= new Uint8ClampedArray(depth.length * 4);
  let min = Infinity;
  let max = -Infinity;
  for (const v of depth) {
    if (v < min) min = v;
    if (v > max) max = v;
  }
  const inv = max > min ? 255 / (max - min) : 0;
  for (let i = 0; i < depth.length; i++) {
    const t = ((depth[i]! - min) * inv) | 0;
    out[i * 4] = TURBO[t * 3]!;
    out[i * 4 + 1] = TURBO[t * 3 + 1]!;
    out[i * 4 + 2] = TURBO[t * 3 + 2]!;
    out[i * 4 + 3] = 255;
  }
  return out;
}
