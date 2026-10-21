/** EmbeddingGemma 2 image encoding: an image in, its token rows for the
 *  text model out, one eager forward of the vision tower per image. */

import { tensor, toArray, type RunntimeExecutor } from '../../core/index.ts';
import { checkImageSize, type ImageBuffer } from '../tasks/cv/image.ts';
import { resizeBicubic } from '../tasks/cv/utils/resizeBicubic.ts';
import type { EmbeddingGemma2VisionInputs, EmbeddingGemma2VisionModel } from './visionModel.ts';
import { visionRopeTables } from './rope.ts';

/** Patch grid [cols, rows] of the largest resize, aspect ratio kept, that
 *  fits `maxTokens` image tokens with each side a multiple of pool ·
 *  patch pixels. Throws INVALID_ARGUMENT unless the size is positive. */
export function patchGrid(
  width: number,
  height: number,
  maxTokens: number,
  patch: number,
  pool: number,
): [number, number] {
  checkImageSize(width, height);
  const maxPatches = maxTokens * pool * pool;
  const factor = Math.sqrt((maxPatches * patch * patch) / (width * height));
  const side = pool * patch;
  let w = Math.floor((factor * width) / side) * side;
  let h = Math.floor((factor * height) / side) * side;
  // The scaled image covers maxTokens tokens, so at most one side rounds
  // to 0: a very thin image, which keeps one row or column of tokens.
  const maxSide = maxTokens * side;
  if (h === 0) {
    h = side;
    w = Math.min(Math.floor(width / height) * side, maxSide);
  } else if (w === 0) {
    w = side;
    h = Math.min(Math.floor(height / width) * side, maxSide);
  }
  return [w / patch, h / patch];
}

/** Pixels [cols · rows, 3·patch²] in [-1, 1], patches in row-major order,
 *  each patch's pixels in (row, column, channel) order. */
function patchPixels(image: ImageBuffer, cols: number, rows: number, patch: number): Float32Array {
  const width = cols * patch;
  const height = rows * patch;
  const chw = resizeBicubic(image, width, height);
  const scaled = (v: number) => (v * 2) / 255 - 1;
  const plane = width * height;
  const out = new Float32Array(3 * plane);
  let o = 0;
  for (let py = 0; py < rows; py++) {
    for (let px = 0; px < cols; px++) {
      for (let y = py * patch; y < (py + 1) * patch; y++) {
        for (let x = px * patch; x < (px + 1) * patch; x++) {
          const i = y * width + x;
          out[o++] = scaled(chw[i]!);
          out[o++] = scaled(chw[plane + i]!);
          out[o++] = scaled(chw[2 * plane + i]!);
        }
      }
    }
  }
  return out;
}

/** Pooling weights [N, P]: each image token averages its pool×pool square
 *  of patches. Tokens are in row-major order over the squares. */
function poolWeights(cols: number, rows: number, pool: number): Float32Array {
  const tokenCols = cols / pool;
  const n = tokenCols * (rows / pool);
  const out = new Float32Array(n * cols * rows);
  for (let k = 0; k < cols * rows; k++) {
    const token =
      Math.floor((k % cols) / pool) + tokenCols * Math.floor(Math.floor(k / cols) / pool);
    out[token * cols * rows + k] = 1 / (pool * pool);
  }
  return out;
}

/** An image to its token rows [N · text hidden], N at most `maxTokens`.
 *  Runs on `ex` and trims its pool after. Throws INVALID_ARGUMENT for a
 *  malformed image. */
export async function encodeImage(
  model: EmbeddingGemma2VisionModel,
  image: ImageBuffer,
  maxTokens: number,
  ex: RunntimeExecutor,
): Promise<Float32Array> {
  const { patch, pool, headDim, ropeTheta } = model.cfg;
  const [cols, rows] = patchGrid(image.width, image.height, maxTokens, patch, pool);
  const p = cols * rows;
  const act = model.input_proj.weight.shape.dtype === 'f16' ? 'f16' : 'f32';
  const upload = (data: Float32Array, r: number, dtype: 'f32' | 'f16') =>
    tensor(data, { elems: data.length, dtype, dims: [r, data.length / r] });
  const xs = new Float32Array(p);
  const ys = new Float32Array(p);
  for (let k = 0; k < p; k++) {
    xs[k] = k % cols;
    ys[k] = Math.floor(k / cols);
  }
  const { cos, sin } = visionRopeTables(cols, rows, headDim, ropeTheta);
  const inputs: EmbeddingGemma2VisionInputs = {
    patches: upload(patchPixels(image, cols, rows, patch), p, act),
    xs: upload(xs, p, 'f32'),
    ys: upload(ys, p, 'f32'),
    cos: upload(cos, p, 'f32'),
    sin: upload(sin, p, 'f32'),
    pool: upload(poolWeights(cols, rows, pool), p / (pool * pool), act),
  };
  try {
    return await toArray(model.forward(inputs), ex);
  } finally {
    for (const v of Object.values(inputs)) v.buffer.destroy();
    ex.trim();
  }
}
