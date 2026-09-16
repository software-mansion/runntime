/** YOLO26 detection head. regMax=1, so the box branch emits raw ltrb distances
 *  with no DFL; only the one-to-one towers run at inference. */

import { cat, nn, toChw } from '../../core/index.ts';
import type { Value } from '../../core/index.ts';
import { ConvBlock } from './blocks.ts';

const { Conv2d, Module, ModuleList } = nn;
// Tower stages (ConvBlock / depthwise / plain 1x1) are all plain Value→Value.
type Tower = nn.ModuleList<nn.Module>;

function gcd(a: number, b: number): number {
  return b === 0 ? a : gcd(b, a % b);
}

/** Depthwise conv; groups is the gcd, so unequal channel counts degrade to
 *  grouped rather than failing. */
function depthwiseConv(inChannels: number, outChannels: number, kernelSize: number): ConvBlock {
  return new ConvBlock(inChannels, outChannels, {
    kernelSize,
    groups: gcd(inChannels, outChannels),
  });
}

export function plainConv1x1(inChannels: number, outChannels: number) {
  return new Conv2d(inChannels, outChannels, { kernelSize: 1, bias: true });
}

/** Concat head outputs on the channel axis, staying in hwc4 when every part is
 *  aligned and falling back to chw when one isn't. */
export function catHeads(parts: readonly Value[]): Value {
  if (
    parts.every((p) => p.shape.layout === 'hwc4') &&
    parts.every((p) => p.shape.dims![0]! % 4 === 0)
  ) {
    return cat(parts, 0);
  }
  return cat(
    parts.map((p) => (p.shape.layout === 'hwc4' ? toChw(p, 'f16') : p)),
    0,
  );
}

/** Per-level box and class towers: one tensor per level, box channels first,
 *  then class logits. */
export class DetectHead extends Module<[readonly Value[]], Value[]> {
  readonly one2one_cv2: nn.ModuleList<Tower>;
  readonly one2one_cv3: nn.ModuleList<Tower>;
  readonly numClasses: number;
  constructor(numClasses: number, regMax: number, levelChannels: readonly number[]) {
    super();
    this.numClasses = numClasses;
    const boxChannels = Math.max(16, Math.floor(levelChannels[0]! / 4), regMax * 4);
    const clsChannels = Math.max(levelChannels[0]!, Math.min(numClasses, 100));
    this.one2one_cv2 = new ModuleList(
      levelChannels.map(
        (channels) =>
          new ModuleList<nn.Module>([
            new ConvBlock(channels, boxChannels, { kernelSize: 3 }),
            new ConvBlock(boxChannels, boxChannels, { kernelSize: 3 }),
            plainConv1x1(boxChannels, 4 * regMax),
          ]),
      ),
    );
    this.one2one_cv3 = new ModuleList(
      levelChannels.map(
        (channels) =>
          new ModuleList<nn.Module>([
            new ModuleList<nn.Module>([
              depthwiseConv(channels, channels, 3),
              new ConvBlock(channels, clsChannels, { kernelSize: 1 }),
            ]),
            new ModuleList<nn.Module>([
              depthwiseConv(clsChannels, clsChannels, 3),
              new ConvBlock(clsChannels, clsChannels, { kernelSize: 1 }),
            ]),
            plainConv1x1(clsChannels, numClasses),
          ]),
      ),
    );
  }
  override forward(feats: readonly Value[]): Value[] {
    return feats.map((feat, level) => {
      const box = this.one2one_cv2.items[level]!.forward(feat);
      const cls = this.one2one_cv3.items[level]!.forward(feat);
      return catHeads([box, cls]);
    });
  }
}
