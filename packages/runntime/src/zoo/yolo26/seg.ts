/** YOLO26 segmentation head: the detection head plus per-level mask-coefficient
 *  towers and one shared prototype generator, combined in the pipeline. */

import { add, nn, toChw, toHwc4, upsample2d } from '../../core/index.ts';
import type { Value } from '../../core/index.ts';
import { ConvBlock } from './blocks.ts';
import { catHeads, DetectHead, plainConv1x1 } from './head.ts';

const { ConvTranspose2d, Module, ModuleList } = nn;

/** Shared mask prototypes: coarse levels refined and upsampled onto P3's grid
 *  and summed, then a conv stack takes the fused map to quarter resolution. */
export class MaskPrototypes extends Module<[readonly Value[]], Value> {
  readonly feat_refine: nn.ModuleList<ConvBlock>;
  readonly feat_fuse: ConvBlock;
  readonly cv1: ConvBlock;
  readonly upsample: nn.ConvTranspose2d;
  readonly cv2: ConvBlock;
  readonly cv3: ConvBlock;
  constructor(levelChannels: readonly number[], hiddenChannels: number, numMaskCoeffs: number) {
    super();
    this.feat_refine = new ModuleList(
      levelChannels
        .slice(1)
        .map((channels) => new ConvBlock(channels, levelChannels[0]!, { kernelSize: 1 })),
    );
    this.feat_fuse = new ConvBlock(levelChannels[0]!, hiddenChannels, { kernelSize: 3 });
    this.cv1 = new ConvBlock(hiddenChannels, hiddenChannels, { kernelSize: 3 });
    // Learned 2× upsampler — plain ConvTranspose2d (bias, no BN/act).
    this.upsample = new ConvTranspose2d(hiddenChannels, hiddenChannels, {
      kernelSize: 2,
      stride: 2,
    });
    this.cv2 = new ConvBlock(hiddenChannels, hiddenChannels, { kernelSize: 3 });
    this.cv3 = new ConvBlock(hiddenChannels, numMaskCoeffs, { kernelSize: 1 });
  }
  override forward(feats: readonly Value[]): Value {
    let fused = feats[0]!;
    for (let level = 1; level < feats.length; level++) {
      const refined = this.feat_refine.items[level - 1]!.forward(feats[level]!);
      fused = add(fused, upsample2d(refined, { scale: 2 ** level }));
    }
    // ConvTranspose2d is a CHW kernel — step out of hwc4 for it, back in after.
    const refined = this.cv1.forward(this.feat_fuse.forward(fused));
    const upsampled = toHwc4(this.upsample.forward(toChw(refined, 'f16')));
    return this.cv3.forward(this.cv2.forward(upsampled));
  }
}

/** Detection head plus per-level mask-coefficient towers and the shared
 *  prototype generator. */
export class SegmentHead extends DetectHead {
  readonly one2one_cv4: nn.ModuleList<nn.ModuleList<nn.Module>>;
  readonly proto: MaskPrototypes;
  readonly numMaskCoeffs: number;
  constructor(
    numClasses: number,
    regMax: number,
    levelChannels: readonly number[],
    opts: { numMaskCoeffs?: number; protoChannels: number },
  ) {
    super(numClasses, regMax, levelChannels);
    const { numMaskCoeffs = 32, protoChannels } = opts;
    this.numMaskCoeffs = numMaskCoeffs;
    const coefChannels = Math.max(Math.floor(levelChannels[0]! / 4), numMaskCoeffs);
    this.one2one_cv4 = new ModuleList(
      levelChannels.map(
        (channels) =>
          new ModuleList<nn.Module>([
            new ConvBlock(channels, coefChannels, { kernelSize: 3 }),
            new ConvBlock(coefChannels, coefChannels, { kernelSize: 3 }),
            plainConv1x1(coefChannels, numMaskCoeffs),
          ]),
      ),
    );
    this.proto = new MaskPrototypes(levelChannels, protoChannels, numMaskCoeffs);
  }
  override forward(feats: readonly Value[]): Value[] {
    return feats.map((feat, level) => {
      const box = this.one2one_cv2.items[level]!.forward(feat);
      const cls = this.one2one_cv3.items[level]!.forward(feat);
      const coef = this.one2one_cv4.items[level]!.forward(feat);
      return catHeads([box, cls, coef]);
    });
  }
  forwardProto(feats: readonly Value[]): Value {
    return this.proto.forward(feats);
  }
}
