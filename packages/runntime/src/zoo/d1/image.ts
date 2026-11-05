/** d1 image preprocessing on the CPU, ported from the checkpoint's vision.py:
 *  LFM2-VL's smart resize and tiling, torch's antialiased bilinear resize, and
 *  16 px patches in SigLIP2 NaFlex order. */

/** Tile side of a large image, px. */
const TILE = 512;
/** SigLIP2 patch side, px. */
export const PATCH = 16;
/** Side of the 2x2 pixel unshuffle in the projector, in patches. */
const FACTOR = 32;
/** Pixel budget of one crop: 64 to 256 projected tokens of 32x32 px. */
const MAX_PIXELS = 256 * 1024;
const MIN_PIXELS = 64 * 1024;

/** RGB bytes, row-major HWC. */
export interface RgbImage {
  width: number;
  height: number;
  data: Uint8Array | Uint8ClampedArray;
}

/** One crop as patch rows, [ph·pw, 3·16·16] in (py, px, c) order, in [-1, 1]. */
export interface Crop {
  ph: number;
  pw: number;
  patches: Float32Array;
}

/** LFM2-VL's smart resize, tile grid and thumbnail size. */
export function layout(
  width: number,
  height: number,
): { grid: [number, number]; thumbnail: [number, number]; tiled: boolean } {
  let h = Math.max(FACTOR, Math.round(height / FACTOR) * FACTOR);
  let w = Math.max(FACTOR, Math.round(width / FACTOR) * FACTOR);
  if (h * w > MAX_PIXELS) {
    const beta = Math.sqrt((height * width) / MAX_PIXELS);
    h = Math.max(FACTOR, Math.floor(height / beta / FACTOR) * FACTOR);
    w = Math.max(FACTOR, Math.floor(width / beta / FACTOR) * FACTOR);
  } else if (h * w < MIN_PIXELS) {
    const beta = Math.sqrt(MIN_PIXELS / (height * width));
    h = Math.ceil((height * beta) / FACTOR) * FACTOR;
    w = Math.ceil((width * beta) / FACTOR) * FACTOR;
  }
  const tiled =
    Math.max(16, Math.round(height / FACTOR) * FACTOR) *
      Math.max(16, Math.round(width / FACTOR) * FACTOR) >
    MAX_PIXELS * 2;
  let grid: [number, number] = [1, 1];
  if (tiled) {
    const seen = new Set<string>();
    const ratios: [number, number][] = [];
    for (let n = 2; n <= 10; n++) {
      for (let x = 1; x <= n; x++) {
        for (let y = 1; y <= n; y++) {
          if (x * y < 2 || x * y > 10 || seen.has(`${x},${y}`)) continue;
          seen.add(`${x},${y}`);
          ratios.push([x, y]);
        }
      }
    }
    // Python sorts the set by area; ties keep set order, which this matches
    // only up to tie order. The tie rule below prefers the larger grid anyway.
    ratios.sort((a, b) => a[0] * a[1] - b[0] * b[1]);
    let best = Infinity;
    for (const r of ratios) {
      const diff = Math.abs(width / height - r[0] / r[1]);
      if (diff < best || (diff === best && width * height > 0.5 * TILE * TILE * r[0] * r[1])) {
        grid = r;
        best = diff;
      }
    }
  }
  return { grid, thumbnail: [h, w], tiled };
}

/** Per-output-index source taps of torch's antialiased bilinear resize
 *  (PIL's triangle filter, support widened by the downscale factor). */
function aaTaps(inSize: number, outSize: number): { start: number; weights: Float64Array }[] {
  const scale = inSize / outSize;
  const support = scale >= 1 ? scale : 1;
  const invScale = scale >= 1 ? 1 / scale : 1;
  const taps = [];
  for (let i = 0; i < outSize; i++) {
    const center = scale * (i + 0.5);
    const start = Math.max(Math.trunc(center - support + 0.5), 0);
    const end = Math.min(Math.trunc(center + support + 0.5), inSize);
    const weights = new Float64Array(end - start);
    let sum = 0;
    for (let j = start; j < end; j++) {
      const w = Math.max(0, 1 - Math.abs((j - center + 0.5) * invScale));
      weights[j - start] = w;
      sum += w;
    }
    for (let j = 0; j < weights.length; j++) weights[j]! /= sum || 1;
    taps.push({ start, weights });
  }
  return taps;
}

/** Antialiased bilinear resize of an [h, w, c] float image, separable:
 *  width first, then height. */
export function resizeAA(
  src: Float32Array,
  h: number,
  w: number,
  c: number,
  outH: number,
  outW: number,
): Float32Array {
  const tx = aaTaps(w, outW);
  const ty = aaTaps(h, outH);
  const mid = new Float32Array(h * outW * c);
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < outW; x++) {
      const { start, weights } = tx[x]!;
      for (let ch = 0; ch < c; ch++) {
        let acc = 0;
        for (let k = 0; k < weights.length; k++)
          acc += weights[k]! * src[(y * w + start + k) * c + ch]!;
        mid[(y * outW + x) * c + ch] = acc;
      }
    }
  }
  const out = new Float32Array(outH * outW * c);
  for (let y = 0; y < outH; y++) {
    const { start, weights } = ty[y]!;
    for (let x = 0; x < outW; x++) {
      for (let ch = 0; ch < c; ch++) {
        let acc = 0;
        for (let k = 0; k < weights.length; k++)
          acc += weights[k]! * mid[((start + k) * outW + x) * c + ch]!;
        out[(y * outW + x) * c + ch] = acc;
      }
    }
  }
  return out;
}

/** Resize to uint8 levels, as torchvision does on a uint8 tensor. */
function resizeRgb(img: RgbImage, outH: number, outW: number): Float32Array {
  const out = resizeAA(Float32Array.from(img.data), img.height, img.width, 3, outH, outW);
  for (let i = 0; i < out.length; i++) out[i] = Math.min(255, Math.max(0, Math.round(out[i]!)));
  return out;
}

/** An [h, w, 3] uint8-level crop to normalized patch rows. */
function patchify(px: Float32Array, h: number, w: number): Crop {
  const ph = h / PATCH;
  const pw = w / PATCH;
  const dim = 3 * PATCH * PATCH;
  const patches = new Float32Array(ph * pw * dim);
  for (let py = 0; py < ph; py++) {
    for (let pxi = 0; pxi < pw; pxi++) {
      const base = (py * pw + pxi) * dim;
      for (let y = 0; y < PATCH; y++) {
        for (let x = 0; x < PATCH; x++) {
          const s = ((py * PATCH + y) * w + pxi * PATCH + x) * 3;
          const d = base + (y * PATCH + x) * 3;
          for (let ch = 0; ch < 3; ch++) patches[d + ch] = (px[s + ch]! - 127.5) / 127.5;
        }
      }
    }
  }
  return { ph, pw, patches };
}

/** An image to its crops: the tiles of a large image, then the thumbnail. */
export function preprocessImage(img: RgbImage): Crop[] {
  const plan = layout(img.width, img.height);
  const crops: Crop[] = [];
  if (plan.tiled) {
    const [gw, gh] = plan.grid;
    const big = resizeRgb(img, gh * TILE, gw * TILE);
    const bigW = gw * TILE;
    for (let r = 0; r < gh; r++) {
      for (let c = 0; c < gw; c++) {
        const tile = new Float32Array(TILE * TILE * 3);
        for (let y = 0; y < TILE; y++) {
          const s = ((r * TILE + y) * bigW + c * TILE) * 3;
          tile.set(big.subarray(s, s + TILE * 3), y * TILE * 3);
        }
        crops.push(patchify(tile, TILE, TILE));
      }
    }
  }
  const [h, w] = plan.thumbnail;
  crops.push(patchify(resizeRgb(img, h, w), h, w));
  return crops;
}
