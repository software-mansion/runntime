/** The full Moonshine module tree. No constructor takes a sequence length —
 *  rope tables and cross K/V arrive as forward arguments — and one
 *  loadStateDict binds the whole checkpoint.
 *
 *  The embedding weight serves as both token lookup and the tied LM head. No
 *  transposed copy exists, so tying is exact by construction. */

import {
  argmaxDot,
  gatherRowsFrom,
  matmul,
  nn,
  slice,
  transpose,
  type LazyStateDict,
  type Value,
} from '../../core/index.ts';
import type { MoonshineConfig } from './config.ts';
import { ConvStem } from './convStem.ts';
import { MoonshineEncoder } from './encoder.ts';
import { MoonshineDecoder } from './decoder.ts';
import { transformMoonshineStateDict } from './stateDictHooks.ts';

export class MoonshineModel extends nn.Module {
  readonly embedding: nn.Embedding;
  readonly stem: ConvStem;
  readonly encoder: MoonshineEncoder;
  readonly decoder: MoonshineDecoder;
  constructor(readonly cfg: MoonshineConfig) {
    super();
    this.embedding = new nn.Embedding(cfg.vocab, cfg.dModel);
    this.stem = new ConvStem(cfg);
    this.encoder = new MoonshineEncoder(cfg);
    this.decoder = new MoonshineDecoder(cfg);
  }

  override transformStateDict(sd: LazyStateDict, prefix: string): void {
    transformMoonshineStateDict(sd, prefix, this.cfg);
  }

  embed(ids: readonly number[]): Value {
    return this.embedding.forward(ids);
  }

  embedFrom(ids: Value): Value {
    return gatherRowsFrom(this.embedding.weight.value, ids);
  }

  logitsLast(h: Value): Value {
    const w = this.embedding.weight.value;
    if (w.shape.dtype !== h.shape.dtype) {
      throw new Error(
        `MoonshineModel.logitsLast: embedding is ${w.shape.dtype} but h is ${h.shape.dtype}`,
      );
    }
    const t = h.shape.dims![0]!;
    return matmul(w, transpose(slice(h, 0, t - 1, t)));
  }

  nextTokenId(h1: Value): Value {
    return argmaxDot(this.embedding.weight.value, h1);
  }

  forward(): Value {
    throw new Error('MoonshineModel: use stem/encoder/decoder forwards, embed() and logitsLast()');
  }
}
