/** Moonshine v2 decoder: the v1 decoder (same architecture, different dims)
 *  plus the position embedding and, where the widths differ, the projection
 *  that bring the encoder output to its cross attention. Positions enter
 *  here, as the encoder carries none. */

import { add, nn, slice, type Value } from '../../core/index.ts';
import { MoonshineDecoder } from '../moonshine/decoder.ts';
import type { MoonshineConfig } from '../moonshine/config.ts';
import type { MoonshineStreamingConfig } from './config.ts';

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

export class StreamingDecoder extends MoonshineDecoder {
  readonly pos_emb: nn.Embedding;
  readonly proj?: nn.Linear;
  constructor(cfg: MoonshineStreamingConfig) {
    super(decoderView(cfg));
    const [eh, dh] = [cfg.enc.hidden, cfg.dec.hidden];
    this.pos_emb = new nn.Embedding(cfg.maxPositions, eh);
    this.proj = eh === dh ? undefined : new nn.Linear(eh, dh, { bias: false });
  }
  /** Encoder frames [T, enc.hidden] to what cross attention reads
   *  [T, dec.hidden]: positions added, the width projected. */
  adapt(encOut: Value): Value {
    const t = encOut.shape.dims![0]!;
    const positioned = add(encOut, slice(this.pos_emb.weight.value, 0, 0, t));
    return this.proj ? this.proj.forward(positioned) : positioned;
  }
}
