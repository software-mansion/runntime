/** YOLO26 attention stages from 2D primitives: per-head channel slice →
 *  reshape to [channels, H·W] → matmul/softmax → reshape back. */

import {
  add,
  cat,
  chunk,
  matmul,
  mul,
  nn,
  reshape,
  slice,
  softmax,
  toChw,
  toHwc4,
  transpose,
} from '../../core/index.ts';
import type { Value } from '../../core/index.ts';
import { ResidualBlock, ConvBlock } from './blocks.ts';

const { Module, ModuleList } = nn;

/** Multi-head self-attention over the spatial grid, plus a depthwise
 *  positional term added to the attention output before the projection. */
export class Attention extends Module {
  readonly qkv: ConvBlock;
  readonly proj: ConvBlock;
  readonly pe: ConvBlock;
  private readonly numHeads: number;
  private readonly headDim: number;
  private readonly keyDim: number;
  private readonly scale: number;
  constructor(dim: number, opts: { numHeads?: number; attnRatio?: number } = {}) {
    super();
    const { numHeads = 8, attnRatio = 0.5 } = opts;
    this.numHeads = numHeads;
    this.headDim = Math.floor(dim / numHeads);
    this.keyDim = Math.floor(this.headDim * attnRatio);
    this.scale = this.keyDim ** -0.5;
    const keyChannelsAllHeads = this.keyDim * numHeads;
    // One conv emits query, key and value for every head, side by side.
    this.qkv = new ConvBlock(dim, dim + keyChannelsAllHeads * 2, { kernelSize: 1, act: false });
    this.proj = new ConvBlock(dim, dim, { kernelSize: 1, act: false });
    this.pe = new ConvBlock(dim, dim, { kernelSize: 3, groups: dim, act: false });
  }
  forward(x: Value): Value {
    const [, height, width] = x.shape.dims! as [number, number, number];
    const numCells = height * width;
    // qkv emits hwc4 but the per-head math below is row-major: convert once
    // here (20² maps, the copies are noise), back to hwc4 for pe/proj.
    const qkv = toChw(this.qkv.forward(x), 'f16');
    const channelsPerHead = 2 * this.keyDim + this.headDim;
    const headOuts: Value[] = [];
    const valueSlices: Value[] = [];
    for (let head = 0; head < this.numHeads; head++) {
      const base = head * channelsPerHead;
      const query = reshape(slice(qkv, 0, base, base + this.keyDim), [this.keyDim, numCells]);
      const key = reshape(slice(qkv, 0, base + this.keyDim, base + 2 * this.keyDim), [
        this.keyDim,
        numCells,
      ]);
      const valueMap = slice(qkv, 0, base + 2 * this.keyDim, base + channelsPerHead);
      const value = reshape(valueMap, [this.headDim, numCells]);
      // weights[i,j] = softmax_j( (query_i·scale) · key_j ) — rows are queries,
      // matching torch's softmax(dim=-1) on (query^T @ key).
      const weights = softmax(matmul(transpose(mul(query, this.scale)), key));
      const headOut = matmul(value, transpose(weights)); // [headDim, numCells]
      headOuts.push(reshape(headOut, [this.headDim, height, width]));
      valueSlices.push(valueMap);
    }
    const attended = cat(headOuts, 0);
    const positional = this.pe.forward(toHwc4(cat(valueSlices, 0)));
    return this.proj.forward(add(toHwc4(attended), positional));
  }
}

/** Attention followed by a two-layer feed-forward, each with its own residual
 *  — a transformer block with the norms left out. */
export class AttnFfnBlock extends Module {
  readonly attn: Attention;
  readonly ffn: InstanceType<typeof ModuleList<ConvBlock>>;
  constructor(channels: number, opts: { numHeads?: number; attnRatio?: number } = {}) {
    super();
    this.attn = new Attention(channels, {
      numHeads: opts.numHeads ?? 4,
      attnRatio: opts.attnRatio ?? 0.5,
    });
    this.ffn = new ModuleList([
      new ConvBlock(channels, channels * 2, { kernelSize: 1 }),
      new ConvBlock(channels * 2, channels, { kernelSize: 1, act: false }),
    ]);
  }
  forward(x: Value): Value {
    const attended = add(x, this.attn.forward(x));
    return add(attended, this.ffn.forward(attended));
  }
}

/** A 1×1 doubles the channels, the halves split, only one pays for the
 *  attention blocks, and a 1×1 fuses them back. Channel count unchanged. */
export class SplitAttnStage extends Module {
  readonly cv1: ConvBlock;
  readonly cv2: ConvBlock;
  readonly m: InstanceType<typeof ModuleList<AttnFfnBlock>>;
  private readonly hidden: number;
  constructor(
    inChannels: number,
    outChannels: number,
    opts: { repeats?: number; expansion?: number } = {},
  ) {
    super();
    if (inChannels !== outChannels) {
      throw new Error(
        `SplitAttnStage: inChannels must equal outChannels, got ${inChannels} vs ${outChannels}`,
      );
    }
    const { repeats = 1, expansion = 0.5 } = opts;
    this.hidden = Math.floor(inChannels * expansion);
    this.cv1 = new ConvBlock(inChannels, 2 * this.hidden, { kernelSize: 1 });
    this.cv2 = new ConvBlock(2 * this.hidden, inChannels, { kernelSize: 1 });
    this.m = new ModuleList(
      Array.from(
        { length: repeats },
        () => new AttnFfnBlock(this.hidden, { numHeads: Math.floor(this.hidden / 64) }),
      ),
    );
  }
  forward(x: Value): Value {
    const [passthrough, attnInput] = chunk(this.cv1.forward(x), 2, 0) as [Value, Value];
    return this.cv2.forward(cat([passthrough, this.m.forward(attnInput)], 0));
  }
}

/** SplitFuseStage where each repeat is a residual block plus an attention
 *  block. Used once at P5, where the grid is small enough to be cheap. */
export class SplitFuseAttnStage extends Module {
  readonly cv1: ConvBlock;
  readonly cv2: ConvBlock;
  readonly m: InstanceType<typeof ModuleList<InstanceType<typeof ModuleList>>>;
  private readonly hidden: number;
  constructor(
    inChannels: number,
    outChannels: number,
    opts: { repeats?: number; expansion?: number; shortcut?: boolean } = {},
  ) {
    super();
    const { repeats = 1, expansion = 0.5, shortcut = true } = opts;
    this.hidden = Math.floor(outChannels * expansion);
    this.cv1 = new ConvBlock(inChannels, 2 * this.hidden, { kernelSize: 1 });
    this.cv2 = new ConvBlock((2 + repeats) * this.hidden, outChannels, { kernelSize: 1 });
    this.m = new ModuleList(
      Array.from(
        { length: repeats },
        () =>
          new ModuleList<InstanceType<typeof Module>>([
            new ResidualBlock(this.hidden, this.hidden, { shortcut }),
            new AttnFfnBlock(this.hidden, { numHeads: Math.max(Math.floor(this.hidden / 64), 1) }),
          ]),
      ),
    );
  }
  forward(x: Value): Value {
    const parts: Value[] = chunk(this.cv1.forward(x), 2, 0);
    for (const block of this.m.items) parts.push(block.forward(parts[parts.length - 1]!));
    return this.cv2.forward(cat(parts, 0));
  }
}
