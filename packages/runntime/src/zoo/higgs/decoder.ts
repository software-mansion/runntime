/** The Higgs Audio v2 codec decoder: codes [8, frames] at 25 frames per
 *  second to 24 kHz mono audio, loaded from the codec's transformers
 *  checkpoint.
 *
 *  A frame's input is one codebook row per codebook through that codebook's
 *  project_out, summed, then fc2. All of it is linear, so the load folds it
 *  into one table row per code plus a bias, summed on the CPU. Then DAC's
 *  decoder without the final tanh: a conv, five upsampling blocks (Snake,
 *  transposed conv, three dilated residual units), Snake and a conv to one
 *  channel. Time-major throughout. */

import {
  add,
  derivedTensor,
  nn,
  RunntimeError,
  type LazyStateDict,
  type LazyTensor,
  type Value,
} from '../../core/index.ts';
import { checkpointMismatch, requireTensor } from '../errors.ts';
import { HIGGS_DECODER, type HiggsDecoderConfig } from './config.ts';

class ResidualUnit extends nn.Module {
  readonly snake1: nn.Snake;
  readonly conv1: nn.Conv1d;
  readonly snake2: nn.Snake;
  readonly conv2: nn.Conv1d;
  constructor(channels: number, kernel: number, dilation: number) {
    super();
    this.snake1 = new nn.Snake(channels);
    this.conv1 = new nn.Conv1d(channels, channels, kernel, {
      dilation,
      padding: ((kernel - 1) / 2) * dilation,
    });
    this.snake2 = new nn.Snake(channels);
    this.conv2 = new nn.Conv1d(channels, channels, 1);
  }
  override forward(x: Value): Value {
    const y = this.conv2.forward(this.snake2.forward(this.conv1.forward(this.snake1.forward(x))));
    return add(x, y);
  }
}

/** Snake, the upsampling conv, then three residual units with dilations 1,
 *  3 and 9. */
class Block extends nn.Module {
  readonly snake1: nn.Snake;
  readonly conv_t1: nn.ConvTranspose1d;
  readonly res_unit1: ResidualUnit;
  readonly res_unit2: ResidualUnit;
  readonly res_unit3: ResidualUnit;
  constructor(inChannels: number, stride: number, kernel: number) {
    super();
    const out = inChannels / 2;
    this.snake1 = new nn.Snake(inChannels);
    this.conv_t1 = new nn.ConvTranspose1d(inChannels, out, 2 * stride, {
      stride,
      padding: Math.ceil(stride / 2),
      outputPadding: stride % 2,
    });
    this.res_unit1 = new ResidualUnit(out, kernel, 1);
    this.res_unit2 = new ResidualUnit(out, kernel, 3);
    this.res_unit3 = new ResidualUnit(out, kernel, 9);
  }
  override forward(x: Value): Value {
    const h = this.conv_t1.forward(this.snake1.forward(x));
    return this.res_unit3.forward(this.res_unit2.forward(this.res_unit1.forward(h)));
  }
}

class AcousticDecoder extends nn.Module {
  readonly conv1: nn.Conv1d;
  readonly block: nn.ModuleList<Block>;
  readonly snake1: nn.Snake;
  readonly conv2: nn.Conv1d;
  constructor(cfg: HiggsDecoderConfig) {
    super();
    const { latent, channels, strides, kernel } = cfg;
    const pad = (kernel - 1) / 2;
    this.conv1 = new nn.Conv1d(latent, channels, kernel, { padding: pad });
    this.block = new nn.ModuleList(strides.map((s, i) => new Block(channels / 2 ** i, s, kernel)));
    const last = channels / 2 ** strides.length;
    this.snake1 = new nn.Snake(last);
    this.conv2 = new nn.Conv1d(last, 1, kernel, { padding: pad });
  }
  override forward(x: Value): Value {
    return this.conv2.forward(this.snake1.forward(this.block.forward(this.conv1.forward(x))));
  }
}

const WHAT = 'higgs decoder weights';

interface Codebook {
  /** [codebookSize, dim] */
  embed: LazyTensor;
  /** project_out, torch layout [hidden, dim] and [hidden]. */
  weight: LazyTensor;
  bias: LazyTensor;
}

/** The decoder input for code i of codebook k is embed_k[i]·project_out_kᵀ·
 *  fc2ᵀ plus a bias shared by every frame: fc2's bias and each project_out
 *  bias through fc2. Both come out as derived tensors, computed in f64. */
function foldCodebooks(
  books: Codebook[],
  fc2W: LazyTensor,
  fc2B: LazyTensor,
  { codebookSize, latent }: HiggsDecoderConfig,
) {
  const dim = books[0]!.embed.shape[1]!;
  const hidden = fc2W.shape[1]!;
  const want = (t: LazyTensor, what: string, shape: number[]) => {
    if (t.shape.join() !== shape.join()) {
      throw checkpointMismatch(WHAT, `${what} has shape [${t.shape}], expected [${shape}]`);
    }
  };
  want(fc2W, 'fc2.weight', [latent, hidden]);
  want(fc2B, 'fc2.bias', [latent]);
  for (const b of books) {
    want(b.embed, 'a codebook embed', [codebookSize, dim]);
    want(b.weight, 'a project_out weight', [hidden, dim]);
    want(b.bias, 'a project_out bias', [hidden]);
  }
  // fc2 as [hidden, latent], so the inner loops run along latent. Both
  // tensors share it.
  let transposed: Promise<Float64Array> | undefined;
  const fc2T = () =>
    (transposed ??= fc2W.f32().then((w) => {
      const t = new Float64Array(hidden * latent);
      for (let o = 0; o < latent; o++) {
        for (let j = 0; j < hidden; j++) t[j * latent + o] = w[o * hidden + j]!;
      }
      return t;
    }));
  const sources = [...books.flatMap((b) => [b.embed, b.weight, b.bias]), fc2W, fc2B];
  const table = derivedTensor(
    [books.length * codebookSize, latent],
    async () => {
      const f = await fc2T();
      const out = new Float32Array(books.length * codebookSize * latent);
      const m = new Float64Array(dim * latent);
      const row = new Float64Array(latent);
      for (const [k, b] of books.entries()) {
        const w = await b.weight.f32();
        const e = await b.embed.f32();
        // m = project_outᵀ·fc2ᵀ, [dim, latent].
        m.fill(0);
        for (let j = 0; j < hidden; j++) {
          for (let a = 0; a < dim; a++) {
            const p = w[j * dim + a]!;
            for (let o = 0; o < latent; o++) m[a * latent + o]! += p * f[j * latent + o]!;
          }
        }
        for (let i = 0; i < codebookSize; i++) {
          row.fill(0);
          for (let a = 0; a < dim; a++) {
            const v = e[i * dim + a]!;
            for (let o = 0; o < latent; o++) row[o]! += v * m[a * latent + o]!;
          }
          out.set(row, (k * codebookSize + i) * latent);
        }
      }
      return out;
    },
    sources.reduce((sum, t) => sum + t.byteLength, 0),
  );
  const bias = derivedTensor([latent], async () => {
    const f = await fc2T();
    const out = Float64Array.from(await fc2B.f32());
    for (const b of books) {
      const pb = await b.bias.f32();
      for (let j = 0; j < hidden; j++) {
        for (let o = 0; o < latent; o++) out[o]! += pb[j]! * f[j * latent + o]!;
      }
    }
    return Float32Array.from(out);
  });
  return { table, bias };
}

export class HiggsDecoder extends nn.Module {
  /** [codebooks·codebookSize, latent] on the CPU, folded at load. */
  readonly codes: nn.Embedding;
  readonly codes_bias: nn.CpuParameter;
  readonly acoustic_decoder: AcousticDecoder;

  constructor(readonly cfg: HiggsDecoderConfig = HIGGS_DECODER) {
    super();
    const { codebooks, codebookSize, latent } = cfg;
    this.codes = new nn.Embedding(codebooks * codebookSize, latent, { device: 'cpu' });
    this.codes_bias = new nn.CpuParameter({ elems: latent, dtype: 'f32', dims: [latent] });
    this.acoustic_decoder = new AcousticDecoder(cfg);
  }

  /** Folds the codebooks, their project_out and fc2 into codes and
   *  codes_bias, and drops the tensors the decoder does not read: the
   *  encoder, the semantic model and the quantizer's training state. */
  override transformStateDict(sd: LazyStateDict, prefix: string): void {
    const scope = prefix ? `${prefix}.` : '';
    const get = (name: string) => requireTensor(sd, scope + name, WHAT);
    const books = Array.from({ length: this.cfg.codebooks }, (_, k) => ({
      embed: get(`quantizer.quantizers.${k}.codebook.embed`),
      weight: get(`quantizer.quantizers.${k}.project_out.weight`),
      bias: get(`quantizer.quantizers.${k}.project_out.bias`),
    }));
    const { table, bias } = foldCodebooks(books, get('fc2.weight'), get('fc2.bias'), this.cfg);
    for (const name of [...sd.tensors.keys()]) {
      if (name.startsWith(scope) && !name.startsWith(`${scope}acoustic_decoder.`)) {
        sd.tensors.delete(name);
      }
    }
    sd.tensors.set(`${scope}codes.weight`, table);
    sd.tensors.set(`${scope}codes_bias`, bias);
  }

  /** Input rows [frames, latent] for codes [codebooks, frames], each code in
   *  [0, codebookSize). */
  latents(codes: Int32Array, frames: number): Float32Array {
    const { codebooks, codebookSize, latent } = this.cfg;
    if (codes.length !== codebooks * frames) {
      throw new RunntimeError(
        'INVALID_ARGUMENT',
        `HiggsDecoder: ${codes.length} codes for ${codebooks} codebooks × ${frames} frames`,
      );
    }
    const ids = new Array<number>(codebooks * frames);
    for (let t = 0; t < frames; t++) {
      for (let c = 0; c < codebooks; c++) {
        const code = codes[c * frames + t]!;
        if (code < 0 || code >= codebookSize) {
          throw new RunntimeError(
            'INVALID_ARGUMENT',
            `HiggsDecoder: code ${code} out of range [0, ${codebookSize})`,
          );
        }
        ids[t * codebooks + c] = c * codebookSize + code;
      }
    }
    const rows = this.codes.gather(ids);
    // The fold writes codes_bias, so it is always f32.
    const bias = this.codes_bias.data as Float32Array;
    const out = new Float32Array(frames * latent);
    for (let t = 0; t < frames; t++) {
      out.set(bias, t * latent);
      for (let c = 0; c < codebooks; c++) {
        const src = (t * codebooks + c) * latent;
        for (let i = 0; i < latent; i++) out[t * latent + i]! += rows[src + i]!;
      }
    }
    return out;
  }

  /** Latent rows [frames, latent] to samples [frames·∏strides, 1]. */
  override forward(x: Value): Value {
    return this.acoustic_decoder.forward(x);
  }
}
