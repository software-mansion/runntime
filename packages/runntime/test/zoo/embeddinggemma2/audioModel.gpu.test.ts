import { afterAll, describe, it } from 'vitest';
import { toArray } from '../../../src/core/gpu/eval.ts';
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
const { hidden } = AUDIO;
/** Bounds q, k and v share, as the hook requires. The input range is
 *  narrower than the random input's [-2, 2), the output range wide enough
 *  that some outputs fall inside it. */
const BOUNDS = { input_min: -0.5, input_max: 0.5, output_min: -1, output_max: 1 };

const weights = PARTS.map((_, i) => randomF32(hidden * hidden, i + 1));
const sd = await fromSafetensors(bufferSource(audioFile()));
const tensors = memoryStateDict({
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
});
