/** YOLO26 building blocks. Module and parameter field names (`conv`, `cv1`,
 *  `m`, ...) are checkpoint state-dict keys and cannot be renamed. */

import { add, cat, chunk, maxPool2d, mul, nn, sigmoid } from '../../core/index.ts';
import type { Value } from '../../core/index.ts';

const { BatchNorm2d, Conv2d, Module, ModuleList } = nn;

export function silu(x: Value): Value {
  return mul(x, sigmoid(x));
}

/** Conv → BatchNorm → SiLU. BN folds into the conv bias at load, so forward()
 *  only runs the conv. */
export class ConvBlock extends Module {
  static readonly BN_EPS = 1e-3;

  readonly conv: InstanceType<typeof Conv2d>;
  readonly bn: InstanceType<typeof BatchNorm2d>;
  constructor(
    inChannels: number,
    outChannels: number,
    opts: { kernelSize?: number; stride?: number; groups?: number; act?: boolean } = {},
  ) {
    super();
    const { kernelSize = 1, stride = 1, groups = 1 } = opts;
    this.conv = new Conv2d(inChannels, outChannels, {
      kernelSize,
      stride,
      padding: Math.floor(kernelSize / 2), // keeps H/W when stride is 1
      groups,
      bias: true, // BN folds in here
      activation: opts.act === false ? undefined : 'silu', // fused into the conv epilogue
    });
    this.bn = new BatchNorm2d(outChannels, { eps: ConvBlock.BN_EPS });
  }
  forward(x: Value): Value {
    return this.conv.forward(x);
  }
}

/** Two convs with an identity skip, present only when in and out channel
 *  counts match. */
export class ResidualBlock extends Module {
  readonly cv1: ConvBlock;
  readonly cv2: ConvBlock;
  private readonly residual: boolean;
  constructor(
    inChannels: number,
    outChannels: number,
    opts: { shortcut?: boolean; kernels?: [number, number]; expansion?: number } = {},
  ) {
    super();
    const { shortcut = true, kernels = [3, 3], expansion = 0.5 } = opts;
    const hidden = Math.floor(outChannels * expansion);
    this.cv1 = new ConvBlock(inChannels, hidden, { kernelSize: kernels[0] });
    this.cv2 = new ConvBlock(hidden, outChannels, { kernelSize: kernels[1] });
    this.residual = shortcut && inChannels === outChannels;
  }
  forward(x: Value): Value {
    const out = this.cv2.forward(this.cv1.forward(x));
    return this.residual ? add(x, out) : out;
  }
}

/** Cross-stage partial: two 1×1 branches, one through `repeats` residual
 *  blocks, concatenated and mixed back to `outChannels` by a final 1×1. */
export class CspStage extends Module {
  readonly cv1: ConvBlock;
  readonly cv2: ConvBlock;
  readonly cv3: ConvBlock;
  readonly m: InstanceType<typeof ModuleList<ResidualBlock>>;
  constructor(
    inChannels: number,
    outChannels: number,
    opts: { repeats?: number; shortcut?: boolean; expansion?: number; kernelSize?: number } = {},
  ) {
    super();
    const { repeats = 2, shortcut = true, expansion = 0.5, kernelSize = 3 } = opts;
    const hidden = Math.floor(outChannels * expansion);
    this.cv1 = new ConvBlock(inChannels, hidden, { kernelSize: 1 });
    this.cv2 = new ConvBlock(inChannels, hidden, { kernelSize: 1 });
    this.cv3 = new ConvBlock(2 * hidden, outChannels, { kernelSize: 1 });
    this.m = new ModuleList(
      Array.from(
        { length: repeats },
        () =>
          new ResidualBlock(hidden, hidden, {
            shortcut,
            kernels: [kernelSize, kernelSize],
            expansion: 1.0,
          }),
      ),
    );
  }
  forward(x: Value): Value {
    const deep = this.m.forward(this.cv1.forward(x));
    return this.cv3.forward(cat([deep, this.cv2.forward(x)], 0));
  }
}

/** One 1×1 makes 2×hidden channels split in half; each of `repeats` blocks
 *  appends its output before a 1×1 fuses them. `nested`: repeats are CspStages. */
export class SplitFuseStage extends Module {
  readonly cv1: ConvBlock;
  readonly cv2: ConvBlock;
  // Repeat blocks (CspStage | ResidualBlock) are all plain Value→Value.
  readonly m: nn.ModuleList<nn.Module>;
  private readonly hidden: number;
  constructor(
    inChannels: number,
    outChannels: number,
    opts: { repeats?: number; nested?: boolean; expansion?: number; shortcut?: boolean } = {},
  ) {
    super();
    const { repeats = 1, nested = false, expansion = 0.5, shortcut = true } = opts;
    this.hidden = Math.floor(outChannels * expansion);
    this.cv1 = new ConvBlock(inChannels, 2 * this.hidden, { kernelSize: 1 });
    this.cv2 = new ConvBlock((2 + repeats) * this.hidden, outChannels, { kernelSize: 1 });
    this.m = new ModuleList<nn.Module>(
      Array.from({ length: repeats }, () =>
        nested
          ? new CspStage(this.hidden, this.hidden, { repeats: 2, shortcut })
          : new ResidualBlock(this.hidden, this.hidden, { shortcut }),
      ),
    );
  }
  forward(x: Value): Value {
    const parts: Value[] = chunk(this.cv1.forward(x), 2, 0);
    for (const block of this.m.items) parts.push(block.forward(parts[parts.length - 1]!));
    return this.cv2.forward(cat(parts, 0));
  }
}

/** Spatial pyramid pool: 1×1 down to half the channels, then `repeats` chained
 *  k×k max-pools concatenated — multi-scale for the cost of a small kernel. */
export class PyramidPool extends Module {
  readonly cv1: ConvBlock;
  readonly cv2: ConvBlock;
  private readonly kernelSize: number;
  private readonly repeats: number;
  private readonly residual: boolean;
  constructor(
    inChannels: number,
    outChannels: number,
    opts: { kernelSize?: number; repeats?: number; shortcut?: boolean } = {},
  ) {
    super();
    const { kernelSize = 5, repeats = 3, shortcut = false } = opts;
    const hidden = Math.floor(inChannels / 2);
    this.cv1 = new ConvBlock(inChannels, hidden, { kernelSize: 1, act: false });
    this.cv2 = new ConvBlock(hidden * (repeats + 1), outChannels, { kernelSize: 1 });
    this.kernelSize = kernelSize;
    this.repeats = repeats;
    this.residual = shortcut && inChannels === outChannels;
  }
  forward(x: Value): Value {
    const scales: Value[] = [this.cv1.forward(x)];
    for (let i = 0; i < this.repeats; i++) {
      scales.push(
        maxPool2d(scales[scales.length - 1]!, {
          kernelSize: this.kernelSize,
          stride: 1,
          padding: Math.floor(this.kernelSize / 2),
        }),
      );
    }
    const out = this.cv2.forward(cat(scales, 0));
    return this.residual ? add(x, out) : out;
  }
}
