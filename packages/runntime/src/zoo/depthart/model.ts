/** DepthART: a MetaFormer-style conv/SSM hybrid backbone with a DPT head.
 *  Field names mirror the checkpoint's own.
 *
 *      patch_embed  3 -> 24 -> 48, two stride-2 convs + GELU   (1/4 res)
 *      network.0    stage 1, 4 blocks     -> norm0  tap
 *      network.1    downsample 48 -> 96
 *      network.2    stage 2, 3 blocks     -> norm2  tap
 *      network.3    downsample 96 -> 192
 *      network.4    stage 3, 10 blocks    -> norm4  tap
 *      network.5    downsample 192 -> 384
 *      network.6    stage 4, 5 blocks     -> norm6  tap
 *      depth_head   DPT refine cascade over the four taps
 *
 *  The last block of each stage is an SSM block, and stage 3 has a second at
 *  index 5. Everything else is a conv block. */

import { materialized, mul, nn, tensor3d, toChw, toHwc4, uploadF32 } from '../../core/index.ts';
import type { Value } from '../../core/index.ts';
import { defaultRoot } from '../../core/index.ts';
import { Act, BatchNormC, Conv2dBN } from './layers.ts';
import { ConvBlock, SSMBlock } from './blocks.ts';
import { SCAN_SIDE } from './ss2d.ts';
import { DepthHead } from './head.ts';

const { Module, ModuleList } = nn;

export interface DepthartConfig {
  readonly dims: readonly [number, number, number, number];
  readonly depths: readonly [number, number, number, number];
  readonly ssmAt: readonly (readonly number[])[];
  readonly dInner: readonly [number, number, number, number];
  readonly dtRank: readonly [number, number, number, number];
  readonly features: number;
  readonly inputSize: number;
  readonly mean: readonly [number, number, number];
  readonly std: readonly [number, number, number];
}

export const DEPTHART_B_448 = Object.freeze({
  dims: [48, 96, 192, 384] as const,
  depths: [4, 3, 10, 5] as const,
  ssmAt: [[3], [2], [5, 9], [4]] as const,
  dInner: [12, 48, 96, 288] as const,
  dtRank: [3, 6, 12, 24] as const,
  features: 48,
  inputSize: 448,
  mean: [0.485, 0.456, 0.406] as const,
  std: [0.229, 0.224, 0.225] as const,
}) satisfies DepthartConfig;

export const DEPTHART_S_448 = Object.freeze({
  dims: [48, 64, 168, 224] as const,
  depths: [3, 3, 9, 6] as const,
  ssmAt: [[2], [2], [4, 8], [5]] as const,
  dInner: [12, 32, 84, 168] as const,
  dtRank: [3, 4, 11, 14] as const,
  features: 48,
  inputSize: 448,
  mean: [0.485, 0.456, 0.406] as const,
  std: [0.229, 0.224, 0.225] as const,
}) satisfies DepthartConfig;

class Downsample extends Module {
  readonly proj: Conv2dBN;
  constructor(cIn: number, cOut: number) {
    super();
    this.proj = new Conv2dBN(cIn, cOut, { kernelSize: 3, stride: 2, padding: 1 });
  }
  forward(x: Value): Value {
    return this.proj.forward(x);
  }
}

type Stage = nn.ModuleList<ConvBlock | SSMBlock>;

export class Backbone extends Module<[Value], readonly Value[]> {
  readonly patch_embed: nn.ModuleList<Conv2dBN | Act>;
  readonly network: nn.ModuleList<Stage | Downsample>;
  readonly norm0: BatchNormC;
  readonly norm2: BatchNormC;
  readonly norm4: BatchNormC;
  readonly norm6: BatchNormC;

  constructor(readonly cfg: DepthartConfig) {
    super();
    const { dims, depths, ssmAt, dInner, dtRank, inputSize } = cfg;
    // The fused GELUs still occupy slots 1 and 3, to match the checkpoint.
    this.patch_embed = new ModuleList<Conv2dBN | Act>([
      new Conv2dBN(3, dims[0] / 2, { kernelSize: 3, stride: 2, padding: 1, activation: 'gelu' }),
      new Act(),
      new Conv2dBN(dims[0] / 2, dims[0], {
        kernelSize: 3,
        stride: 2,
        padding: 1,
        activation: 'gelu',
      }),
      new Act(),
    ]);

    const entries: (Stage | Downsample)[] = [];
    for (let s = 0; s < 4; s++) {
      // patch_embed already halved twice; each earlier stage halves once more.
      const side = inputSize / (4 * 2 ** s);
      const ssm = new Set(ssmAt[s]!);
      entries.push(
        new ModuleList<ConvBlock | SSMBlock>(
          Array.from({ length: depths[s]! }, (_, i) =>
            ssm.has(i)
              ? new SSMBlock({
                  channels: dims[s]!,
                  dInner: dInner[s]!,
                  dtRank: dtRank[s]!,
                  side,
                })
              : new ConvBlock(dims[s]!),
          ),
        ),
      );
      if (s < 3) entries.push(new Downsample(dims[s]!, dims[s + 1]!));
    }
    this.network = new ModuleList<Stage | Downsample>(entries);
    // BatchNorms sitting after a residual add, so there is no conv to fold
    // into.
    this.norm0 = new BatchNormC(dims[0]!);
    this.norm2 = new BatchNormC(dims[1]!);
    this.norm4 = new BatchNormC(dims[2]!);
    this.norm6 = new BatchNormC(dims[3]!);
  }

  forward(x: Value): readonly Value[] {
    // Enters hwc4 here and stays there; the SS2D islands convert locally.
    let h = this.patch_embed.forward(toHwc4(x));
    const taps: Value[] = [];
    const norms = [this.norm0, this.norm2, this.norm4, this.norm6];
    for (const [i, entry] of this.network.items.entries()) {
      if (entry instanceof Downsample) {
        h = entry.forward(h);
        continue;
      }
      for (const block of entry.items) h = block.forward(h);
      taps.push(norms[i / 2]!.forward(h));
    }
    return taps;
  }
}

export class DepthartModel extends Module {
  readonly pretrained: Backbone;
  readonly depth_head: DepthHead;
  constructor(readonly cfg: DepthartConfig = DEPTHART_B_448) {
    super();
    // Every published checkpoint is trained at 448, which pools onto the fixed
    // 14² scan grid. Other sizes would mis-pool, so reject them up front.
    if (cfg.inputSize / 4 / 2 ** 3 !== SCAN_SIDE) {
      throw new Error(
        `DepthART: input ${cfg.inputSize} puts the last stage off the ${SCAN_SIDE} scan grid`,
      );
    }
    this.pretrained = new Backbone(cfg);
    this.depth_head = new DepthHead(cfg.dims, cfg.features);
  }
  forward(img: Value): Value {
    const raw = this.depth_head.forward(this.pretrained.forward(img), this.cfg.inputSize);
    return toChw(mul(raw, -1), 'f32');
  }
}

export async function loadDepthartModel(
  url: string,
  opts: {
    cfg?: DepthartConfig;
    onProgress?: (name: string, doneBytes: number, totalBytes: number) => void;
  } = {},
): Promise<DepthartModel> {
  const model = new DepthartModel(opts.cfg ?? DEPTHART_B_448);
  await model.loadStateDict(url, { onProgress: opts.onProgress });
  return model;
}

export function imageValue(chw: Float32Array, size: number): Value {
  return materialized(tensor3d(3, size, size), uploadF32(defaultRoot(), chw));
}
