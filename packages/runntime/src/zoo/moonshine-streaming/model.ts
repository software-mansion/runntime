/** The full Moonshine v2 module tree. No constructor takes a sequence length,
 *  and one loadStateDict binds the whole checkpoint.
 *
 *  The decoder is the v1 class: same architecture, different dims, so
 *  `decoderView` bridges the two config shapes. Cross K and V read the adapter
 *  output rather than the encoder's.
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
import type { MoonshineConfig } from '../moonshine/config.ts';
import { MoonshineDecoder } from '../moonshine/decoder.ts';
import type { MoonshineStreamingConfig } from './config.ts';
import { StreamingFrontend } from './frontend.ts';
import { StreamingAdapter, StreamingEncoder } from './encoder.ts';
import { transformStreamingStateDict } from './stateDictHooks.ts';

export function decoderView(cfg: MoonshineStreamingConfig): MoonshineConfig {
  return {
    dModel: cfg.dec.hidden,
    ffn: cfg.dec.ffn,
    encLayers: cfg.enc.layers,
    decLayers: cfg.dec.layers,
    heads: cfg.dec.heads,
    headDim: cfg.dec.headDim,
    rotaryDim: cfg.dec.rotaryDim,
    ropeTheta: cfg.dec.ropeTheta,
    vocab: cfg.vocab,
    maxPositions: cfg.maxPositions,
    bos: cfg.bos,
    eos: cfg.eos,
  };
}

export class MoonshineStreamingModel extends nn.Module {
  readonly embedding: nn.Embedding;
  readonly frontend: StreamingFrontend;
  readonly encoder: StreamingEncoder;
  readonly adapter: StreamingAdapter;
  readonly decoder: MoonshineDecoder;
  constructor(readonly cfg: MoonshineStreamingConfig) {
    super();
    this.embedding = new nn.Embedding(cfg.vocab, cfg.dec.hidden);
    this.frontend = new StreamingFrontend(cfg);
    this.encoder = new StreamingEncoder(cfg);
    this.adapter = new StreamingAdapter(cfg);
    this.decoder = new MoonshineDecoder(decoderView(cfg));
  }

  override transformStateDict(sd: LazyStateDict, prefix: string): void {
    transformStreamingStateDict(sd, prefix, this.cfg);
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
        `MoonshineStreamingModel.logitsLast: embedding is ${w.shape.dtype} but h is ${h.shape.dtype}`,
      );
    }
    const t = h.shape.dims![0]!;
    return matmul(w, transpose(slice(h, 0, t - 1, t)));
  }

  nextTokenId(h1: Value): Value {
    return argmaxDot(this.embedding.weight.value, h1);
  }

  forward(): Value {
    throw new Error(
      'MoonshineStreamingModel: use frontend/encoder/adapter/decoder forwards, embed() and logitsLast()',
    );
  }
}
