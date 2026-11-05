/** The full Moonshine module tree, named like the HF checkpoint:
 *  `model.encoder` and `model.decoder` under MoonshineForConditionalGeneration.
 *  No constructor takes a sequence length — rope tables and cross K/V arrive
 *  as forward arguments — and one loadStateDict binds the whole checkpoint.
 *
 *  The decoder's embedding weight serves as both token lookup and the tied LM
 *  head, so a checkpoint that also stores the tied `proj_out.weight` gets it
 *  reported as unread. */

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
import { MoonshineEncoder } from './encoder.ts';
import { MoonshineDecoder } from './decoder.ts';
import { transformMoonshineStateDict } from './stateDictHooks.ts';

/** HF's MoonshineModel: the encoder and decoder pair. */
export class MoonshineEncoderDecoder extends nn.Module {
  readonly encoder: MoonshineEncoder;
  readonly decoder: MoonshineDecoder;
  constructor(cfg: MoonshineConfig) {
    super();
    this.encoder = new MoonshineEncoder(cfg);
    this.decoder = new MoonshineDecoder(cfg);
  }
}

export class MoonshineModel extends nn.Module {
  readonly model: MoonshineEncoderDecoder;
  constructor(readonly cfg: MoonshineConfig) {
    super();
    this.model = new MoonshineEncoderDecoder(cfg);
  }

  override transformStateDict(sd: LazyStateDict, prefix: string): void {
    transformMoonshineStateDict(sd, prefix, this.cfg);
  }

  embed(ids: readonly number[]): Value {
    return this.model.decoder.embed_tokens.forward(ids);
  }

  embedFrom(ids: Value): Value {
    return gatherRowsFrom(this.model.decoder.embed_tokens.weight.value, ids);
  }

  logitsLast(h: Value): Value {
    const w = this.model.decoder.embed_tokens.weight.value;
    if (w.shape.dtype !== h.shape.dtype) {
      throw new Error(
        `MoonshineModel.logitsLast: embedding is ${w.shape.dtype} but h is ${h.shape.dtype}`,
      );
    }
    const t = h.shape.dims![0]!;
    return matmul(w, transpose(slice(h, 0, t - 1, t)));
  }

  nextTokenId(h1: Value): Value {
    return argmaxDot(this.model.decoder.embed_tokens.weight.value, h1);
  }

  forward(): Value {
    throw new Error(
      'MoonshineModel: use model.encoder/model.decoder forwards, embed() and logitsLast()',
    );
  }
}
