/** YOLO26 pose head: the detection head (one class, person) plus a per-level
 *  keypoint branch of [x, y, visibility-logit] channels per keypoint. */

import { nn } from '../../core/index.ts';
import type { Value } from '../../core/index.ts';
import { ConvBlock } from './blocks.ts';
import { catHeads, DetectHead, plainConv1x1 } from './head.ts';

const { ModuleList } = nn;

export class PoseHead extends DetectHead {
  readonly one2one_cv4: nn.ModuleList<nn.ModuleList<nn.Module>>;
  readonly one2one_cv4_kpts: nn.ModuleList<nn.Module>;
  readonly numKptChannels: number;
  constructor(
    numClasses: number,
    regMax: number,
    levelChannels: readonly number[],
    opts: { kptShape?: readonly [number, number] } = {},
  ) {
    super(numClasses, regMax, levelChannels);
    const [numKeypoints, channelsPerKeypoint] = opts.kptShape ?? [17, 3];
    this.numKptChannels = numKeypoints * channelsPerKeypoint;
    // The +2 matches the training-time sigma branch, which shares this width.
    const kptChannels = Math.max(
      Math.floor(levelChannels[0]! / 4),
      numKeypoints * (channelsPerKeypoint + 2),
    );
    this.one2one_cv4 = new ModuleList(
      levelChannels.map(
        (channels) =>
          new ModuleList<nn.Module>([
            new ConvBlock(channels, kptChannels, { kernelSize: 3 }),
            new ConvBlock(kptChannels, kptChannels, { kernelSize: 3 }),
          ]),
      ),
    );
    this.one2one_cv4_kpts = new ModuleList(
      levelChannels.map(() => plainConv1x1(kptChannels, this.numKptChannels)),
    );
  }
  override forward(feats: readonly Value[]): Value[] {
    return super.forward(feats).map((boxCls, level) => {
      const kpt = this.one2one_cv4_kpts.items[level]!.forward(
        this.one2one_cv4.items[level]!.forward(feats[level]!),
      );
      return catHeads([boxCls, kpt]);
    });
  }
}
