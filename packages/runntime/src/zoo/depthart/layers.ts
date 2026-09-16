/** DepthART building blocks, named after the checkpoint's own fields. Both are
 *  load-time weight transforms rather than new math. */

import { channelAffine, nn, pad2d, type ConvAct } from '../../core/index.ts';
import type { Value } from '../../core/index.ts';
import { derivedTensor, type LazyStateDict, type LazyTensor } from '../../core/index.ts';

const { BatchNorm2d, Conv2d, Module, Parameter } = nn;

export class Conv2dBN extends Module {
  readonly c: InstanceType<typeof Conv2d>;
  readonly bn: InstanceType<typeof BatchNorm2d>;
  constructor(
    cIn: number,
    cOut: number,
    opts: {
      kernelSize?: number | [number, number];
      stride?: number;
      padding?: number;
      groups?: number;
      activation?: ConvAct;
    } = {},
  ) {
    super();
    const { kernelSize = 1, stride = 1, padding = 0, groups = 1, activation } = opts;
    this.c = new Conv2d(cIn, cOut, {
      kernelSize,
      stride,
      padding,
      groups,
      bias: true, // BN folds in here
      activation, // comptime epilogue — no separate activation dispatch
    });
    this.bn = new BatchNorm2d(cOut);
  }
  forward(x: Value): Value {
    return this.c.forward(x);
  }
}

export class Act extends Module {
  forward(x: Value): Value {
    return x;
  }
}

export class BatchNormC extends Module {
  readonly scale: InstanceType<typeof Parameter>;
  readonly shift: InstanceType<typeof Parameter>;
  constructor(
    readonly channels: number,
    readonly eps = 1e-5,
  ) {
    super();
    const p = () => new Parameter({ elems: channels, dtype: 'f32', dims: [channels] });
    this.scale = p();
    this.shift = p();
  }
  forward(x: Value): Value {
    return channelAffine(x, this.scale.value, this.shift.value);
  }
  override transformStateDict(sd: LazyStateDict, prefix: string): void {
    const key = (n: string) => (prefix ? `${prefix}.${n}` : n);
    const at = (n: string): LazyTensor => {
      const t = sd.tensors.get(key(n));
      if (!t) throw new Error(`BatchNormC at '${prefix}': missing '${n}' in the state dict`);
      return t;
    };
    const [g, b, m, v] = [at('weight'), at('bias'), at('running_mean'), at('running_var')];
    const c = this.channels;
    const eps = this.eps;
    let memo: Promise<{ scale: Float32Array; shift: Float32Array }> | undefined;
    const affine = () =>
      (memo ??= (async () => {
        const [gd, bd, md, vd] = await Promise.all([g.f32(), b.f32(), m.f32(), v.f32()]);
        const scale = new Float32Array(c);
        const shift = new Float32Array(c);
        for (let o = 0; o < c; o++) {
          scale[o] = gd[o]! / Math.sqrt(vd[o]! + eps);
          shift[o] = bd[o]! - md[o]! * scale[o]!;
        }
        return { scale, shift };
      })());
    const bytes = g.byteLength + b.byteLength + m.byteLength + v.byteLength;
    sd.tensors.set(
      key('scale'),
      derivedTensor([c], async () => (await affine()).scale, bytes),
    );
    sd.tensors.set(
      key('shift'),
      derivedTensor([c], async () => (await affine()).shift),
    );
    for (const n of ['weight', 'bias', 'running_mean', 'running_var', 'num_batches_tracked']) {
      sd.tensors.delete(key(n));
    }
  }
}

function centerTap(kH: number, kW: number): number {
  return ((kH - 1) / 2) * kW + (kW - 1) / 2;
}

export class RepDW extends Module {
  readonly merged: InstanceType<typeof Conv2d>;
  private readonly kH: number;
  private readonly kW: number;
  constructor(
    readonly channels: number,
    opts: {
      kernelSize?: number | [number, number];
      eps?: number;
      activation?: ConvAct;
    } = {},
  ) {
    super();
    const k = opts.kernelSize ?? 3;
    [this.kH, this.kW] = typeof k === 'number' ? [k, k] : k;
    this.eps = opts.eps ?? 1e-5;
    // Asymmetric kernels need per-axis padding, which the conv bakes as one
    // symmetric value, so forward() pads explicitly and the conv runs
    // unpadded.
    const square = this.kH === this.kW;
    this.merged = new Conv2d(channels, channels, {
      kernelSize: k,
      groups: channels,
      padding: square ? (this.kH - 1) / 2 : 0,
      bias: true,
      activation: opts.activation,
    });
  }
  private readonly eps: number;

  forward(x: Value): Value {
    if (this.kH === this.kW) return this.merged.forward(x);
    return this.merged.forward(pad2d(x, { padH: (this.kH - 1) / 2, padW: (this.kW - 1) / 2 }));
  }

  override transformStateDict(sd: LazyStateDict, prefix: string): void {
    // prefix is empty when this module is the root.
    const key = (name: string) => (prefix ? `${prefix}.${name}` : name);
    const at = (name: string): LazyTensor => {
      const t = sd.tensors.get(key(name));
      if (!t) throw new Error(`RepDW at '${prefix}': missing '${name}' in the state dict`);
      return t;
    };
    const kw = at('conv.c.weight');
    const [innerG, innerB, innerM, innerV] = [
      at('conv.bn.weight'),
      at('conv.bn.bias'),
      at('conv.bn.running_mean'),
      at('conv.bn.running_var'),
    ];
    const pw = at('conv1.weight');
    const pb = at('conv1.bias');
    const [outerG, outerB, outerM, outerV] = [
      at('bn.weight'),
      at('bn.bias'),
      at('bn.running_mean'),
      at('bn.running_var'),
    ];

    const c = this.channels;
    const taps = this.kH * this.kW;
    const center = centerTap(this.kH, this.kW);
    const eps = this.eps;

    let memo: Promise<{ w: Float32Array; b: Float32Array }> | undefined;
    const merged = () =>
      (memo ??= (async () => {
        const [kd, ig, ib, im, iv, pwd, pbd, og, ob, om, ov] = await Promise.all([
          kw.f32(),
          innerG.f32(),
          innerB.f32(),
          innerM.f32(),
          innerV.f32(),
          pw.f32(),
          pb.f32(),
          outerG.f32(),
          outerB.f32(),
          outerM.f32(),
          outerV.f32(),
        ]);
        const w = new Float32Array(c * taps);
        const b = new Float32Array(c);
        for (let o = 0; o < c; o++) {
          // Branch 1: k×k conv with its own BN folded in.
          const innerScale = ig[o]! / Math.sqrt(iv[o]! + eps);
          const base = o * taps;
          for (let t = 0; t < taps; t++) w[base + t] = kd[base + t]! * innerScale;
          let bias = ib[o]! - im[o]! * innerScale;
          // The 1×1 and identity branches are centre taps.
          w[base + center]! += pwd[o]! + 1;
          bias += pbd[o]!;
          // Outer BN folds into the summed branches.
          const outerScale = og[o]! / Math.sqrt(ov[o]! + eps);
          for (let t = 0; t < taps; t++) w[base + t]! *= outerScale;
          b[o] = ob[o]! + (bias - om[o]!) * outerScale;
        }
        return { w, b };
      })());

    const bytes = [kw, innerG, innerB, innerM, innerV, pw, pb, outerG, outerB, outerM, outerV]
      .map((t) => t.byteLength)
      .reduce((a, x) => a + x, 0);
    sd.tensors.set(
      key('merged.weight'),
      derivedTensor([c, 1, this.kH, this.kW], async () => (await merged()).w, bytes),
    );
    sd.tensors.set(
      key('merged.bias'),
      derivedTensor([c], async () => (await merged()).b),
    );
    for (const branch of [
      'conv.c.weight',
      'conv.bn.weight',
      'conv.bn.bias',
      'conv.bn.running_mean',
      'conv.bn.running_var',
      'conv.bn.num_batches_tracked',
      'conv1.weight',
      'conv1.bias',
      'bn.weight',
      'bn.bias',
      'bn.running_mean',
      'bn.running_var',
      'bn.num_batches_tracked',
    ]) {
      sd.tensors.delete(key(branch));
    }
  }
}
