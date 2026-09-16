/** MobileNetV4Model: timm's MobileNetV3 container laid out with the same field
 *  names (conv_stem/bn1 / blocks.{stage}.{i} / conv_head/norm_head /
 *  classifier) so parameter names match the checkpoint keys 1:1. The BN
 *  modules are declared like in timm and fold into their convs at load.
 *
 *  The conv trunk runs HWC4+f16 end to end (toHwc4 once at the input). The
 *  global average pool is a row-major reduction, so the head hops out of hwc4
 *  for the pool and back in for conv_head (a 1×1 conv at 1×1 spatial), then
 *  leaves for the classifier matmul. */

import { astype, mean, nn, reshape, toChw, toHwc4 } from '../../core/index.ts';
import type { Value } from '../../core/index.ts';
import type { TgpuRoot } from 'typegpu';
import { ConvBnAct, UniversalInvertedResidual } from './blocks.ts';
import { MOBILENETV4_CONV_S, type Mnv4Config } from './config.ts';

const { BatchNorm2d, Conv2d, Linear, Module, ModuleList } = nn;

export class MobileNetV4Model extends Module {
  readonly conv_stem: InstanceType<typeof Conv2d>;
  readonly bn1: InstanceType<typeof BatchNorm2d>;
  readonly blocks: nn.ModuleList<nn.ModuleList<nn.Module>>;
  readonly conv_head: InstanceType<typeof Conv2d>;
  readonly norm_head: InstanceType<typeof BatchNorm2d>;
  readonly classifier: InstanceType<typeof Linear>;
  readonly cfg: Mnv4Config;

  constructor(
    cfg: Mnv4Config = MOBILENETV4_CONV_S,
    /** `root` feeds half()'s shader-f16 check — defaults to the initRunntime()
     *  root; graph-only contexts (no device) skip the check. */
    opts: { root?: TgpuRoot } = {},
  ) {
    super();
    this.cfg = cfg;
    this.conv_stem = new Conv2d(3, cfg.stemOut, {
      kernelSize: 3,
      stride: 2,
      padding: 1,
      bias: true, // bn1 folds in here
      activation: 'relu', // bn1's ReLU, fused
    });
    this.bn1 = new BatchNorm2d(cfg.stemOut);
    let c = cfg.stemOut;
    const stages = cfg.stages.map((stage) => {
      const blocks = stage.map((spec) => {
        const block: nn.Module =
          spec.kind === 'cn'
            ? new ConvBnAct(c, spec.out, { kernelSize: spec.kernelSize, stride: spec.stride })
            : new UniversalInvertedResidual(c, spec);
        c = spec.out;
        return block;
      });
      return new ModuleList(blocks);
    });
    this.blocks = new ModuleList(stages);
    this.conv_head = new Conv2d(c, cfg.headHidden, {
      kernelSize: 1,
      bias: true, // norm_head folds in here
      activation: 'relu', // norm_head's ReLU, fused
    });
    this.norm_head = new BatchNorm2d(cfg.headHidden);
    this.classifier = new Linear(cfg.headHidden, cfg.numClasses);
    // The trunk runs HWC4+f16: nn.Conv2d packs mat4 tiles at load, and
    // half() flips the biases and the classifier to f16.
    this.half(opts.root);
  }

  /** x: [3, S, S] f32 CHW → class logits [1, numClasses] f32. */
  override forward(x: Value): Value {
    let y = this.bn1.forward(this.conv_stem.forward(toHwc4(x)));
    for (const stage of this.blocks.items) for (const block of stage.items) y = block.forward(y);
    // Global average pool: [C, H, W] → [C, 1], then a 1×1-spatial conv_head
    // (timm pools BEFORE conv_head, MobileNetV3-style).
    const [ch, h, w] = y.shape.dims as [number, number, number];
    const pooled = mean(reshape(toChw(y, 'f16'), [ch, h * w]));
    y = this.norm_head.forward(this.conv_head.forward(toHwc4(reshape(pooled, [ch, 1, 1]))));
    const feats = reshape(toChw(y, 'f16'), [1, this.cfg.headHidden]);
    return astype(this.classifier.forward(feats), 'f32');
  }
}
