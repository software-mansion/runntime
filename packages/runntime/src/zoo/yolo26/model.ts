/** The 24-layer yolo26 graph as one flat ModuleList so parameter names match the
 *  checkpoint; parameterless layers hold their index. `task` picks layer 23. */

import { astype, cat, nn, toChw, toHwc4, upsample2d } from '../../core/index.ts';
import type { Value } from '../../core/index.ts';
import type { TgpuRoot } from 'typegpu';
import { SplitFuseStage, ConvBlock, PyramidPool } from './blocks.ts';
import { SplitAttnStage, SplitFuseAttnStage } from './psa.ts';
import { DetectHead } from './head.ts';
import { SegmentHead } from './seg.ts';
import { PoseHead } from './pose.ts';
import {
  scaleChannels,
  scaleRepeats,
  YOLO26_CONFIG,
  YOLO26_SCALES,
  type Yolo26Variant,
} from './config.ts';

const { Module, ModuleList } = nn;

class Upsample2x extends Module {
  forward(x: Value): Value {
    return upsample2d(x, { scale: 2 });
  }
}

class ConcatSlot extends Module {
  forward(x: Value): Value {
    return x;
  }
}

export type Yolo26Task = 'detect' | 'segment' | 'pose';
export type Yolo26Output<T extends Yolo26Task> = T extends 'segment'
  ? { levels: [Value, Value, Value]; proto: Value }
  : [Value, Value, Value];

export class Yolo26Model<T extends Yolo26Task = 'detect'> extends Module<[Value], Yolo26Output<T>> {
  readonly model: nn.ModuleList<nn.AnyModule>;
  readonly variant: Yolo26Variant;
  readonly task: Yolo26Task;
  constructor(
    variant: Yolo26Variant = 'n',
    opts: { numClasses?: number; task?: T; root?: TgpuRoot } = {},
  ) {
    super();
    const { task = 'detect' } = opts;
    // Pose checkpoints are single-class; detect and seg default to COCO-80.
    const { numClasses = task === 'pose' ? 1 : YOLO26_CONFIG.numClasses } = opts;
    this.variant = variant;
    this.task = task;
    const scale = YOLO26_SCALES[variant];
    const ch = (base: number) => scaleChannels(base, scale);
    const rep = (base: number) => scaleRepeats(base, scale);
    // Larger variants nest a CSP stage in every split-fuse stage regardless.
    const nestAll = variant === 'm' || variant === 'l' || variant === 'x';
    const nest = (perLayer: boolean) => perLayer || nestAll;

    // Backbone (layer indices 0–10), base channels in comments.
    const layers: nn.AnyModule[] = [
      new ConvBlock(3, ch(64), { kernelSize: 3, stride: 2 }), // 0 P1/2
      new ConvBlock(ch(64), ch(128), { kernelSize: 3, stride: 2 }), // 1 P2/4
      new SplitFuseStage(ch(128), ch(256), {
        repeats: rep(2),
        nested: nest(false),
        expansion: 0.25,
      }), // 2
      new ConvBlock(ch(256), ch(256), { kernelSize: 3, stride: 2 }), // 3 P3/8
      new SplitFuseStage(ch(256), ch(512), {
        repeats: rep(2),
        nested: nest(false),
        expansion: 0.25,
      }), // 4
      new ConvBlock(ch(512), ch(512), { kernelSize: 3, stride: 2 }), // 5 P4/16
      new SplitFuseStage(ch(512), ch(512), { repeats: rep(2), nested: nest(true) }), // 6
      new ConvBlock(ch(512), ch(1024), { kernelSize: 3, stride: 2 }), // 7 P5/32
      new SplitFuseStage(ch(1024), ch(1024), { repeats: rep(2), nested: nest(true) }), // 8
      new PyramidPool(ch(1024), ch(1024), { kernelSize: 5, repeats: 3, shortcut: true }), // 9
      new SplitAttnStage(ch(1024), ch(1024), { repeats: rep(2) }), // 10
      // Head (layer indices 11–22)
      new Upsample2x(), // 11
      new ConcatSlot(), // 12: cat [11, 6] → ch(1024)+ch(512)
      new SplitFuseStage(ch(1024) + ch(512), ch(512), { repeats: rep(2), nested: nest(true) }), // 13
      new Upsample2x(), // 14
      new ConcatSlot(), // 15: cat [14, 4] → ch(512)+ch(512)
      new SplitFuseStage(ch(512) + ch(512), ch(256), { repeats: rep(2), nested: nest(true) }), // 16 P3 out
      new ConvBlock(ch(256), ch(256), { kernelSize: 3, stride: 2 }), // 17
      new ConcatSlot(), // 18: cat [17, 13] → ch(256)+ch(512)
      new SplitFuseStage(ch(256) + ch(512), ch(512), { repeats: rep(2), nested: nest(true) }), // 19 P4 out
      new ConvBlock(ch(512), ch(512), { kernelSize: 3, stride: 2 }), // 20
      new ConcatSlot(), // 21: cat [20, 10] → ch(512)+ch(1024)
      new SplitFuseAttnStage(ch(512) + ch(1024), ch(1024), { repeats: 1, expansion: 0.5 }), // 22 P5 out
      task === 'segment'
        ? new SegmentHead(numClasses, YOLO26_CONFIG.regMax, [ch(256), ch(512), ch(1024)], {
            protoChannels: scaleChannels(256, scale),
          })
        : task === 'pose'
          ? new PoseHead(numClasses, YOLO26_CONFIG.regMax, [ch(256), ch(512), ch(1024)])
          : new DetectHead(numClasses, YOLO26_CONFIG.regMax, [ch(256), ch(512), ch(1024)]), // 23
    ];
    this.model = new ModuleList(layers);
    // HWC4 + f16 throughout: Conv2d packs mat4 tiles at load, half() flips the
    // biases and every row-major island.
    this.half(opts.root);
  }

  private outF32(v: Value): Value {
    if (v.shape.layout === 'hwc4') return toChw(v, 'f32');
    return astype(v, 'f32');
  }

  private forwardFeats(x: Value): [Value, Value, Value] {
    // Layers 0-22 are plain Value→Value; forward() calls the head separately.
    const layers = this.model.items as readonly nn.Module[];
    // Backbone
    const y0 = layers[0]!.forward(x);
    const y1 = layers[1]!.forward(y0);
    const y2 = layers[2]!.forward(y1);
    const y3 = layers[3]!.forward(y2);
    const y4 = layers[4]!.forward(y3); // → head skip (P3 features, 80×80)
    const y5 = layers[5]!.forward(y4);
    const y6 = layers[6]!.forward(y5); // → head skip (P4 features, 40×40)
    const y7 = layers[7]!.forward(y6);
    const y8 = layers[8]!.forward(y7);
    const y9 = layers[9]!.forward(y8);
    const y10 = layers[10]!.forward(y9); // → head skip (P5 features, 20×20)
    // FPN top-down
    const y13 = layers[13]!.forward(cat([layers[11]!.forward(y10), y6], 0));
    const y16 = layers[16]!.forward(cat([layers[14]!.forward(y13), y4], 0)); // P3 out
    // PAN bottom-up
    const y19 = layers[19]!.forward(cat([layers[17]!.forward(y16), y13], 0)); // P4 out
    const y22 = layers[22]!.forward(cat([layers[20]!.forward(y19), y10], 0)); // P5 out
    return [y16, y19, y22];
  }

  override forward(x: Value): Yolo26Output<T> {
    const feats = this.forwardFeats(toHwc4(x));
    const out = (v: Value) => this.outF32(v);
    if (this.task === 'segment') {
      const head = this.model.items[23] as SegmentHead;
      const [p3, p4, p5] = head.forward(feats);
      return {
        levels: [out(p3!), out(p4!), out(p5!)],
        proto: out(head.forwardProto(feats)),
      } as Yolo26Output<T>;
    }
    const [p3, p4, p5] = (this.model.items[23] as DetectHead).forward(feats);
    return [out(p3!), out(p4!), out(p5!)] as Yolo26Output<T>;
  }
}
