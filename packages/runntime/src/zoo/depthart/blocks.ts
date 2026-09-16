/** DepthART's two block types and the ratio-4 MLP, mirroring the checkpoint's
 *  field names. Residual structure read off the upstream ONNX graph — the two
 *  blocks differ in more than which mixer they use:
 *
 *    ConvBlock:  out = x + layer_scale * mlp(dwconv(x))      one residual
 *    SSMBlock:   h = x + op(x);  out = h + mlp(h)            two, no layer_scale
 *
 *  The depthwise conv's own identity branch lives inside RepDW, so it is not
 *  a block-level residual. */

import { add, channelAffine, nn } from '../../core/index.ts';
import type { Value } from '../../core/index.ts';
import { Conv2dBN, RepDW } from './layers.ts';
import { SS2D, type SS2DConfig } from './ss2d.ts';

const { Module, Parameter } = nn;

export class Mlp extends Module {
  readonly fc1: Conv2dBN;
  readonly fc2: Conv2dBN;
  constructor(channels: number, ratio = 4) {
    super();
    const hidden = channels * ratio;
    this.fc1 = new Conv2dBN(channels, hidden, { activation: 'gelu' });
    this.fc2 = new Conv2dBN(hidden, channels);
  }
  forward(x: Value): Value {
    return this.fc2.forward(this.fc1.forward(x));
  }
}

export class ConvBlock extends Module {
  readonly dwconv: RepDW;
  readonly mlp: Mlp;
  readonly layer_scale: InstanceType<typeof Parameter>;
  constructor(readonly channels: number) {
    super();
    this.dwconv = new RepDW(channels, { kernelSize: 3 });
    this.mlp = new Mlp(channels);
    // The checkpoint stores [C, 1, 1]; channelAffine only reads the element
    // count, so the parameter is declared flat.
    this.layer_scale = new Parameter({ elems: channels, dtype: 'f32', dims: [channels] });
  }
  forward(x: Value): Value {
    const mixed = this.dwconv.forward(x);
    // Per-channel scale in-layout — the hwc4 channelAffine kernel.
    return add(x, channelAffine(this.mlp.forward(mixed), this.layer_scale.value));
  }
}

export class SSMBlock extends Module {
  readonly op: SS2D;
  readonly mlp: Mlp;
  constructor(cfg: SS2DConfig) {
    super();
    this.op = new SS2D(cfg);
    this.mlp = new Mlp(cfg.channels);
  }
  forward(x: Value): Value {
    const mixed = add(x, this.op.forward(x));
    return add(mixed, this.mlp.forward(mixed));
  }
}
