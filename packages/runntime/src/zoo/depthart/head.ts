/** The DPT (MiDaS) refine head, without the projects and resize stack a
 *  general DPT needs: DepthART's backbone already emits four spatial scales, so
 *  each `layerN_rn` reads a tap directly.
 *
 *  Field names mirror the checkpoint. Every interpolation is linear with
 *  align_corners, which is resizeBilinear2d's default. */

import { add, clamp, nn, resizeBilinear2d } from '../../core/index.ts';
import { Act } from './layers.ts';
import type { Value } from '../../core/index.ts';

const { Conv2d, Module } = nn;

const relu = (x: Value): Value => clamp(x, 0, Infinity);

class ResConfUnit extends Module {
  readonly conv1: InstanceType<typeof Conv2d>;
  readonly conv2: InstanceType<typeof Conv2d>;
  constructor(features: number) {
    super();
    this.conv1 = new Conv2d(features, features, { kernelSize: 3, padding: 1, activation: 'relu' });
    this.conv2 = new Conv2d(features, features, { kernelSize: 3, padding: 1 });
  }
  forward(x: Value): Value {
    return add(x, this.conv2.forward(this.conv1.forward(relu(x))));
  }
}

class Fusion extends Module<[Value, Value | undefined, number], Value> {
  readonly resConfUnit1: ResConfUnit;
  readonly resConfUnit2: ResConfUnit;
  readonly out_conv: InstanceType<typeof Conv2d>;
  constructor(features: number) {
    super();
    this.resConfUnit1 = new ResConfUnit(features);
    this.resConfUnit2 = new ResConfUnit(features);
    this.out_conv = new Conv2d(features, features, { kernelSize: 1 });
  }
  forward(prev: Value, skip: Value | undefined, size: number): Value {
    const merged = skip ? add(prev, this.resConfUnit1.forward(skip)) : prev;
    const up = resizeBilinear2d(this.resConfUnit2.forward(merged), { outH: size, outW: size });
    return this.out_conv.forward(up);
  }
}

class Scratch extends Module<[readonly Value[], number], Value> {
  readonly layer1_rn: InstanceType<typeof Conv2d>;
  readonly layer2_rn: InstanceType<typeof Conv2d>;
  readonly layer3_rn: InstanceType<typeof Conv2d>;
  readonly layer4_rn: InstanceType<typeof Conv2d>;
  readonly refinenet1: Fusion;
  readonly refinenet2: Fusion;
  readonly refinenet3: Fusion;
  readonly refinenet4: Fusion;
  readonly output_conv1: InstanceType<typeof Conv2d>;
  readonly output_conv2: nn.ModuleList<InstanceType<typeof Conv2d> | Act>;

  constructor(tapChannels: readonly number[], features: number) {
    super();
    // rn convs are bias-free (the checkpoint carries weight only).
    const rn = (c: number) => new Conv2d(c, features, { kernelSize: 3, padding: 1, bias: false });
    this.layer1_rn = rn(tapChannels[0]!);
    this.layer2_rn = rn(tapChannels[1]!);
    this.layer3_rn = rn(tapChannels[2]!);
    this.layer4_rn = rn(tapChannels[3]!);
    this.refinenet4 = new Fusion(features);
    this.refinenet3 = new Fusion(features);
    this.refinenet2 = new Fusion(features);
    this.refinenet1 = new Fusion(features);
    this.output_conv1 = new Conv2d(features, features / 2, { kernelSize: 3, padding: 1 });
    // torch Sequential [conv, ReLU, conv, ReLU] — only 0 and 2 hold weights,
    // so the ReLU slots are placeholders that keep the numeric field names
    // lined up with the checkpoint. Both ReLUs ride the conv epilogues.
    this.output_conv2 = new nn.ModuleList<InstanceType<typeof Conv2d> | Act>([
      new Conv2d(features / 2, 16, { kernelSize: 3, padding: 1, activation: 'relu' }),
      new Act(),
      new Conv2d(16, 1, { kernelSize: 1, activation: 'relu' }),
      new Act(),
    ]);
  }

  forward(taps: readonly Value[], outSize: number): Value {
    const l1 = this.layer1_rn.forward(taps[0]!);
    const l2 = this.layer2_rn.forward(taps[1]!);
    const l3 = this.layer3_rn.forward(taps[2]!);
    const l4 = this.layer4_rn.forward(taps[3]!);
    const side = (v: Value) => (v.shape.dims as number[])[1]!;

    // Each fusion resizes to the next finer tap's grid; refinenet1 doubles.
    const p4 = this.refinenet4.forward(l4, undefined, side(l3));
    const p3 = this.refinenet3.forward(p4, l3, side(l2));
    const p2 = this.refinenet2.forward(p3, l2, side(l1));
    const p1 = this.refinenet1.forward(p2, l1, 2 * side(l1));

    const up = resizeBilinear2d(this.output_conv1.forward(p1), {
      outH: outSize,
      outW: outSize,
    });
    return this.output_conv2.forward(up);
  }
}

export class DepthHead extends Module<[readonly Value[], number], Value> {
  readonly scratch: Scratch;
  constructor(tapChannels: readonly number[], features = 48) {
    super();
    this.scratch = new Scratch(tapChannels, features);
  }
  forward(taps: readonly Value[], outSize: number): Value {
    return this.scratch.forward(taps, outSize);
  }
}
