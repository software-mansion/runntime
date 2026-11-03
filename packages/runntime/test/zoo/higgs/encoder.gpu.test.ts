import { afterAll, describe, expect, it } from 'vitest';
import { toArray } from '../../../src/core/gpu/eval.ts';
import { matrix } from '../../../src/core/graph/value.ts';
import { HiggsEncoder } from '../../../src/zoo/higgs/encoder.ts';
import { expectClose, gpu, randomF32, upload } from '../../gpu.ts';
import { ACTIVATION_REF } from '../../core/kernels/matmul/checkMatmul.ts';
import { CFG, checkpoint } from './encoderCheckpoint.ts';

const { root, ex } = await gpu();

const raw = checkpoint();
const enc = new HiggsEncoder(CFG);
await enc.loadStateDict(raw, { root });
afterAll(() => enc.dispose());

/** torch's Linear on rows x [t, cIn] with weight [cOut, cIn], in f64. */
function linearRef(x: ArrayLike<number>, t: number, name: string): Float64Array {
  const w = raw[`${name}.weight`]!;
  const b = raw[`${name}.bias`]!.data;
  const [cOut, cIn] = w.shape as [number, number];
  const out = new Float64Array(t * cOut);
  for (let r = 0; r < t; r++) {
    for (let o = 0; o < cOut; o++) {
      let s = b[o]!;
      for (let i = 0; i < cIn; i++) s += x[r * cIn + i]! * w.data[o * cIn + i]!;
      out[r * cOut + o] = s;
    }
  }
  return out;
}

describe('HiggsEncoder on the GPU', () => {
  it("runs HuBERT's grouped position conv as torch does, weight norm and the cut frame included", async () => {
    const { hidden, posGroups, posKernel: k } = CFG.hubert;
    const t = 7;
    const x = randomF32(t * hidden, 9);
    const P = 'semantic_model.encoder.pos_conv_embed.conv';
    const g = raw[`${P}.parametrizations.weight.original0`]!.data;
    const v = raw[`${P}.parametrizations.weight.original1`]!.data;
    const bias = raw[`${P}.bias`]!.data;
    const width = hidden / posGroups;
    const norm = Array.from({ length: k }, (_, j) =>
      Math.sqrt(v.reduce((s, val, i) => (i % k === j ? s + val * val : s), 0)),
    );
    // conv1d with padding k/2 on both sides, groups, then the last frame cut.
    const want = new Float64Array(t * hidden);
    for (let o = 0; o < hidden; o++) {
      const group = Math.floor(o / width);
      for (let r = 0; r < t; r++) {
        let s = bias[o]!;
        for (let i = 0; i < width; i++) {
          for (let j = 0; j < k; j++) {
            const src = r + j - k / 2;
            if (src < 0 || src >= t) continue;
            const w = (v[(o * width + i) * k + j]! * g[j]!) / norm[j]!;
            s += w * x[src * hidden + group * width + i]!;
          }
        }
        want[r * hidden + o] = ACTIVATION_REF.gelu(s);
      }
    }
    const y = enc.semantic_model.encoder.pos_conv_embed.forward(upload(root, x, [t, hidden]));
    expect(y.shape).toEqual(matrix(t, hidden));
    expectClose(await toArray(y, ex), want, { atol: 1e-5, rtol: 1e-4 });
  });

  it('quantizes residually, each codebook taking its nearest row', async () => {
    const { codebooks, codebookSize: size, codebookDim: dim } = CFG;
    const hidden = CFG.acoustic.latent + CFG.hubert.hidden;
    const t = 9;
    const x = randomF32(t * hidden, 11).map((v) => 3 * v);
    const want = new Int32Array(t * codebooks);
    let residual = Float64Array.from(x);
    for (let c = 0; c < codebooks; c++) {
      const Q = `quantizer.quantizers.${c}`;
      const z = linearRef(residual, t, `${Q}.project_in`);
      const embed = raw[`${Q}.codebook.embed`]!.data;
      const rows = new Float64Array(t * dim);
      for (let r = 0; r < t; r++) {
        let best = 0;
        let bestDist = Infinity;
        for (let i = 0; i < size; i++) {
          let dist = 0;
          for (let j = 0; j < dim; j++) dist += (z[r * dim + j]! - embed[i * dim + j]!) ** 2;
          if (dist < bestDist) [best, bestDist] = [i, dist];
        }
        want[r * codebooks + c] = best;
        rows.set(embed.subarray(best * dim, (best + 1) * dim), r * dim);
      }
      const quantized = linearRef(rows, t, `${Q}.project_out`);
      residual = residual.map((v, i) => v - quantized[i]!);
    }
    const y = enc.quantizer.forward(upload(root, x, [t, hidden]));
    expect(y.shape).toEqual(matrix(t, codebooks));
    expect(Int32Array.from(await toArray(y, ex))).toEqual(want);
  });
});
