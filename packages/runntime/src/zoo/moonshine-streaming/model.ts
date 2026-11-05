/** The full Moonshine v2 module tree, named like the HF checkpoint:
 *  `model.encoder` and `model.decoder`. No constructor takes a sequence
 *  length, and one loadStateDict binds the whole checkpoint.
 *
 *  The decoder's embedding weight serves as both token lookup and the tied LM
 *  head, so the checkpoint's `proj_out.weight` is left unread. */

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
import type { MoonshineStreamingConfig } from './config.ts';
import { StreamingDecoder } from './decoder.ts';
import { StreamingEncoder } from './encoder.ts';
import { transformStreamingStateDict } from './stateDictHooks.ts';

/** HF's MoonshineStreamingModel: the encoder and decoder pair. */
export class StreamingEncoderDecoder extends nn.Module {
  readonly encoder: StreamingEncoder;
  readonly decoder: StreamingDecoder;
  constructor(cfg: MoonshineStreamingConfig) {
    super();
    this.encoder = new StreamingEncoder(cfg);
    this.decoder = new StreamingDecoder(cfg);
  }
}

export class MoonshineStreamingModel extends nn.Module {
  readonly model: StreamingEncoderDecoder;
  constructor(readonly cfg: MoonshineStreamingConfig) {
    super();
    this.model = new StreamingEncoderDecoder(cfg);
  }

  override transformStateDict(sd: LazyStateDict, prefix: string): void {
    transformStreamingStateDict(sd, prefix, this.cfg);
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
        `MoonshineStreamingModel.logitsLast: embedding is ${w.shape.dtype} but h is ${h.shape.dtype}`,
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
      'MoonshineStreamingModel: use model.encoder/model.decoder forwards, embed() and logitsLast()',
    );
  }
}
