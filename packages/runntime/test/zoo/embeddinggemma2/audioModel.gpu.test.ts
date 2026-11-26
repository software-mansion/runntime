import { afterAll, describe, expect, it } from 'vitest';
import { toArray } from '../../../src/core/gpu/eval.ts';
import { toHwc, toHwc4 } from '../../../src/core/graph/ops.ts';
import { tensor3d } from '../../../src/core/graph/value.ts';
import {
  bufferSource,
  fromSafetensors,
  memoryStateDict,
} from '../../../src/core/weights/safetensors.ts';
import { EmbeddingGemma2AudioModel } from '../../../src/zoo/embeddinggemma2/audioModel.ts';
import { expectClose, gpu, randomF32, upload } from '../../gpu.ts';
import { AUDIO, audioFile } from './towerCheckpoints.ts';

const { root, ex } = await gpu();

const ATTN = 'audio_tower.layers.0.self_attn';
const PARTS = ['q_proj', 'k_proj', 'v_proj'];
const { hidden, mels, convKernel, eps } = AUDIO;
const [c0, c1] = AUDIO.convChannels;
/** Bounds q, k and v share, as the hook requires. The input range is
 *  narrower than the random input's [-2, 2), the output range wide enough
 *  that some outputs fall inside it. */
const BOUNDS = { input_min: -0.5, input_max: 0.5, output_min: -1, output_max: 1 };

const weights = PARTS.map((_, i) => randomF32(hidden * hidden, i + 1));
const SUB = 'audio_tower.subsample_conv_projection';
const DEPTHWISE = 'audio_tower.layers.0.lconv1d.depthwise_conv1d.weight';
/** The subsample convs and their norms, and a depthwise conv, in torch's
 *  layout. */
const conv0 = randomF32(c0 * 9, 10);
const conv1 = randomF32(c1 * c0 * 9, 11);
const norm0 = randomF32(c0, 12);
const norm1 = randomF32(c1, 13);
const depthwise = randomF32(hidden * convKernel, 14);
const sd = await fromSafetensors(bufferSource(audioFile()));
const tensors = memoryStateDict({
  [`${SUB}.layer0.conv.weight`]: { data: conv0, shape: [c0, 1, 3, 3] },
  [`${SUB}.layer0.norm.weight`]: norm0,
  [`${SUB}.layer1.conv.weight`]: { data: conv1, shape: [c1, c0, 3, 3] },
  [`${SUB}.layer1.norm.weight`]: norm1,
  [DEPTHWISE]: { data: depthwise, shape: [hidden, 1, convKernel] },
  ...Object.fromEntries(
    PARTS.flatMap((part, i) => [
      [`${ATTN}.${part}.linear.weight`, { data: weights[i]!, shape: [hidden, hidden] }],
      ...Object.entries(BOUNDS).map(([name, value]) => [
        `${ATTN}.${part}.${name}`,
        Float32Array.of(value),
      ]),
    ]),
  ),
}).tensors;
for (const [name, tensor] of tensors) sd.tensors.set(name, tensor);
const model = new EmbeddingGemma2AudioModel(AUDIO);
await model.loadStateDict(sd, { root });
afterAll(() => model.dispose());

const clamp = (x: number, lo: number, hi: number) => Math.min(Math.max(x, lo), hi);

describe('EmbeddingGemma2AudioModel on the GPU', () => {
  it('projects q, k and v with the input and the output clamped to the bounds', async () => {
    const rows = 3;
    const x = randomF32(rows * hidden, 9).map((v) => 2 * v);
    // Per row: q, then k, then v, each W · x with x clamped to the input
    // bounds and the result to the output bounds.
    const want: number[] = [];
    for (let r = 0; r < rows; r++) {
      for (const w of weights) {
        for (let o = 0; o < hidden; o++) {
          let s = 0;
          for (let i = 0; i < hidden; i++) {
            const xi = clamp(x[r * hidden + i]!, BOUNDS.input_min, BOUNDS.input_max);
            s += w[o * hidden + i]! * xi;
          }
          want.push(clamp(s, BOUNDS.output_min, BOUNDS.output_max));
        }
      }
    }
    const attn = model.audio_tower.layers.items[0]!.self_attn;
    const y = attn.project(upload(root, x, [rows, hidden]));
    expectClose(await toArray(y, ex), want, { atol: 1e-5, rtol: 1e-4 });
  });

  /** torch's Conv2d(3×3, stride 2, padding 1, no bias) on x [cIn, h, w],
   *  LayerNorm over channels with no bias, then ReLU, in f64. Returns
   *  [cOut, hOut, wOut]. */
  function convLayerRef(
    x: ArrayLike<number>,
    weight: Float32Array,
    norm: Float32Array,
    [cIn, h, w]: [number, number, number],
    cOut: number,
  ) {
    const hOut = Math.floor((h - 1) / 2) + 1;
    const wOut = Math.floor((w - 1) / 2) + 1;
    const out = new Float64Array(cOut * hOut * wOut);
    for (let oy = 0; oy < hOut; oy++) {
      for (let ox = 0; ox < wOut; ox++) {
        const pixel = new Float64Array(cOut);
        for (let o = 0; o < cOut; o++) {
          for (let c = 0; c < cIn; c++) {
            for (let ky = 0; ky < 3; ky++) {
              for (let kx = 0; kx < 3; kx++) {
                const [iy, ix] = [2 * oy - 1 + ky, 2 * ox - 1 + kx];
                if (iy < 0 || iy >= h || ix < 0 || ix >= w) continue;
                pixel[o]! += weight[((o * cIn + c) * 3 + ky) * 3 + kx]! * x[(c * h + iy) * w + ix]!;
              }
            }
          }
        }
        const mean = pixel.reduce((a, b) => a + b) / cOut;
        const variance = pixel.reduce((a, b) => a + (b - mean) ** 2, 0) / cOut;
        for (let o = 0; o < cOut; o++) {
          const normed = ((pixel[o]! - mean) / Math.sqrt(variance + eps)) * norm[o]!;
          out[(o * hOut + oy) * wOut + ox] = Math.max(normed, 0);
        }
      }
    }
    return { out, hOut, wOut };
  }

  // 7 frames: an odd count, so both convs round their output up.
  it('runs the two subsample convs as conv, LayerNorm over channels and ReLU', async () => {
    const frames = 7;
    const mel = randomF32(frames * mels, 20);
    const first = convLayerRef(mel, conv0, norm0, [1, frames, mels], c0);
    const second = convLayerRef(first.out, conv1, norm1, [c0, first.hOut, first.wOut], c1);
    // The model reads [token, frequency, channel].
    const want = new Float64Array(second.out.length);
    const pixels = second.hOut * second.wOut;
    for (let c = 0; c < c1; c++) {
      for (let p = 0; p < pixels; p++) want[p * c1 + c] = second.out[c * pixels + p]!;
    }
    const sub = model.audio_tower.subsample_conv_projection;
    const x = toHwc4(upload(root, mel, tensor3d(1, frames, mels)), 'f32');
    const y = toHwc(sub.layer1.forward(sub.layer0.forward(x)));
    expect(y.shape.dims).toEqual([2, mels / 4, c1]);
    expectClose(await toArray(y, ex), want, { atol: 1e-5, rtol: 1e-4 });
  });

  it('loads the depthwise conv from torch [hidden, 1, taps] as taps of every channel', async () => {
    const want = new Float32Array(depthwise.length);
    for (let c = 0; c < hidden; c++) {
      for (let k = 0; k < convKernel; k++) want[k * hidden + c] = depthwise[c * convKernel + k]!;
    }
    const { weight } = model.audio_tower.layers.items[0]!.lconv1d.depthwise_conv1d;
    expect(weight.shape.dims).toEqual([convKernel, hidden]);
    expectClose(await toArray(weight.value, ex), want, { atol: 0, rtol: 0 });
  });
});
