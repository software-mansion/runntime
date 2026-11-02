/** DepthART's two block types and the ratio-4 MLP, mirroring the checkpoint's
 *  field names. Residual structure read off the upstream ONNX graph — the two
 *  blocks differ in more than which mixer they use:
 *
 *    ConvBlock:  out = x + layer_scale * mlp(dwconv(x))      one residual
 *    SSMBlock:   h = x + op(x);  out = h + mlp(h)            two, no layer_scale
 *
 *  The depthwise conv's own identity branch lives inside RepDW, so it is not
 *  a block-level residual. */

import { add, derivedTensor, nn, type LazyStateDict } from '../../core/index.ts';
import type { Value } from '../../core/index.ts';
import { requireTensor } from '../errors.ts';
import { Conv2dBN, RepDW } from './layers.ts';
import { SS2D, type SS2DConfig } from './ss2d.ts';

const { Module } = nn;

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
  constructor(readonly channels: number) {
    super();
    this.dwconv = new RepDW(channels, { kernelSize: 3 });
    this.mlp = new Mlp(channels);
  }
  forward(x: Value): Value {
    // layer_scale lives in fc2's weights, so the residual add rides fc2's
    // epilogue.
    return add(x, this.mlp.forward(this.dwconv.forward(x)));
  }

  /** Scales fc2's BatchNorm affine by layer_scale, which folds into fc2 with
   *  the BatchNorm. */
  override transformStateDict(sd: LazyStateDict, prefix: string): void {
    const key = (n: string) => (prefix ? `${prefix}.${n}` : n);
    const at = (n: string) => requireTensor(sd, key(n), `ConvBlock at '${prefix}'`);
    const scale = at('layer_scale');
    // f32 on purpose: the BatchNorm fold reads it next, and an f16 product
    // would round once more before the fold.
    for (const n of ['mlp.fc2.bn.weight', 'mlp.fc2.bn.bias']) {
      const src = at(n);
      sd.tensors.set(
        key(n),
        derivedTensor(
          src.shape,
          async () => {
            const [v, s] = await Promise.all([src.f32(), scale.f32()]);
            return v.map((x, o) => x * s[o]!);
          },
          src.byteLength + (n.endsWith('weight') ? scale.byteLength : 0),
        ),
      );
    }
    sd.tensors.delete(key('layer_scale'));
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
