/** MobileNetV4 building blocks, mirroring timm's efficientnet_blocks 1:1 —
 *  same submodule field names (conv/bn1, dw_start/pw_exp/dw_mid/pw_proj with
 *  conv/bn inside) so state-dict keys line up in the loader. nn.BatchNorm2d
 *  folds itself into the preceding conv at load (unfused checkpoints), so
 *  every conv here runs as Conv2d(+bias) with ReLU fused into the conv
 *  kernel epilogue where timm applies one. */

import { add, nn } from '../../core/index.ts';
import type { Value } from '../../core/index.ts';
import type { UirBlockSpec } from './config.ts';

const { BatchNorm2d, Conv2d, Module } = nn;

/** timm ConvBnAct: conv (field `conv`) + `bn1` + ReLU, autopad k//2. */
export class ConvBnAct extends Module {
  readonly conv: InstanceType<typeof Conv2d>;
  readonly bn1: InstanceType<typeof BatchNorm2d>;
  constructor(cIn: number, cOut: number, opts: { kernelSize: number; stride: number }) {
    super();
    this.conv = new Conv2d(cIn, cOut, {
      kernelSize: opts.kernelSize,
      stride: opts.stride,
      padding: Math.floor(opts.kernelSize / 2),
      bias: true, // bn1 folds in here
      activation: 'relu',
    });
    this.bn1 = new BatchNorm2d(cOut);
  }
  forward(x: Value): Value {
    return this.bn1.forward(this.conv.forward(x));
  }
}

/** timm ConvNormAct: conv (field `conv`) + `bn` + optional ReLU. The UIR
 *  depthwise stages set groups=channels; pointwise stages use k=1. */
class ConvNormAct extends Module {
  readonly conv: InstanceType<typeof Conv2d>;
  readonly bn: InstanceType<typeof BatchNorm2d>;
  constructor(
    cIn: number,
    cOut: number,
    opts: { kernelSize: number; stride?: number; groups?: number; act: boolean },
  ) {
    super();
    this.conv = new Conv2d(cIn, cOut, {
      kernelSize: opts.kernelSize,
      stride: opts.stride ?? 1,
      padding: Math.floor(opts.kernelSize / 2),
      groups: opts.groups ?? 1,
      bias: true, // bn folds in here
      activation: opts.act ? 'relu' : undefined,
    });
    this.bn = new BatchNorm2d(cOut);
  }
  forward(x: Value): Value {
    return this.bn.forward(this.conv.forward(x));
  }
}

/** timm UniversalInvertedResidual (MNv4's core block):
 *  dw_start (k×k depthwise, NO act, only some blocks) → pw_exp (1×1 expand,
 *  ReLU) → dw_mid (k×k depthwise carrying the stride, ReLU, only some blocks)
 *  → pw_proj (1×1 project, NO act), with a residual add when stride 1 and
 *  in==out. This variant's checkpoints have layer_scale=Identity, so it is
 *  omitted entirely (no gamma keys). */
export class UniversalInvertedResidual extends Module {
  readonly dw_start?: ConvNormAct;
  readonly pw_exp: ConvNormAct;
  readonly dw_mid?: ConvNormAct;
  readonly pw_proj: ConvNormAct;
  private readonly _hasSkip: boolean;

  constructor(cIn: number, spec: UirBlockSpec) {
    super();
    const mid = cIn * spec.expand;
    if (!Number.isInteger(mid)) {
      throw new Error(`UIR: expansion ${spec.expand}·${cIn} is not a whole channel count`);
    }
    if (spec.dwStartK > 0) {
      // dw_start never carries the stride in this variant (dw_mid does; the
      // one strided block with dw_start — 2.0 — strides in dw_mid).
      this.dw_start = new ConvNormAct(cIn, cIn, {
        kernelSize: spec.dwStartK,
        groups: cIn,
        act: false,
      });
    }
    this.pw_exp = new ConvNormAct(cIn, mid, { kernelSize: 1, act: true });
    if (spec.dwMidK > 0) {
      this.dw_mid = new ConvNormAct(mid, mid, {
        kernelSize: spec.dwMidK,
        stride: spec.stride,
        groups: mid,
        act: true,
      });
    }
    this.pw_proj = new ConvNormAct(mid, spec.out, { kernelSize: 1, act: false });
    this._hasSkip = spec.stride === 1 && cIn === spec.out;
  }

  forward(x: Value): Value {
    let y = x;
    if (this.dw_start) y = this.dw_start.forward(y);
    y = this.pw_exp.forward(y);
    if (this.dw_mid) y = this.dw_mid.forward(y);
    y = this.pw_proj.forward(y);
    return this._hasSkip ? add(x, y) : y;
  }
}
