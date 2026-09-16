/** DepthART's SS2D op: the VMamba-style 2D selective scan. Field names mirror
 *  the checkpoint, and the dataflow was read off the upstream ONNX graph.
 *
 *  There are two splits rather than one, which the shapes alone do not
 *  reveal:
 *
 *      in_proj C→C ─ split(C/2, C/2) ─ dwconv_w [7×1] ┐
 *                                      dwconv_h [1×7] ┴ cat ─ SiLU
 *        └ split(dInner, C−dInner)
 *            gate: local_conv 3×3
 *            ssm:  avgPool ─┬─ upsample, sub  ······ high-pass, kept aside
 *                           └─ project → scan → merge → LayerNorm → resize ↑
 *                  add the high-pass back
 *        └ cat(ssm, gate) → out_proj C→C
 *
 *  The pool is what pins every scan to 14×14 whatever the stage resolution:
 *  the recurrence runs on the pooled map and the detail rides around it. At the
 *  last stage `poolSize` is 1 and the pool and high-pass drop out. */

import {
  add,
  avgPool2d,
  cat,
  layerNorm,
  nn,
  reshape,
  resizeBilinear2d,
  toChw,
  toHwc4,
  slice,
  SSM_DIRS,
  SSM_STATE,
  ssmScanMerge,
  ssmScanProject,
  ssmSelectiveScan,
  sub,
  transpose,
  upsample2d,
  type ConvAct,
} from '../../core/index.ts';
import type { Value } from '../../core/index.ts';
import { derivedTensor, type LazyStateDict } from '../../core/index.ts';
import { Conv2dBN, RepDW } from './layers.ts';

const { Module, Parameter } = nn;

export const SCAN_SIDE = 14;

export interface SS2DConfig {
  readonly channels: number;
  readonly dInner: number;
  readonly dtRank: number;
  readonly side: number;
}

export class SS2D extends Module {
  readonly in_proj: Conv2dBN;
  readonly conv2d: ConvPair;
  readonly local_conv: RepDW;
  readonly out_proj: Conv2dBN;
  readonly x_proj_weight: InstanceType<typeof Parameter>;
  readonly dt_projs_weight: InstanceType<typeof Parameter>;
  readonly dt_projs_bias: InstanceType<typeof Parameter>;
  readonly A_logs: InstanceType<typeof Parameter>;
  readonly Ds: InstanceType<typeof Parameter>;
  readonly out_norm: LayerNormC;

  private readonly poolSize: number;

  constructor(readonly cfg: SS2DConfig) {
    super();
    const { channels, dInner, dtRank, side } = cfg;
    if (side % SCAN_SIDE !== 0) {
      throw new Error(`SS2D: side ${side} is not a multiple of the ${SCAN_SIDE} scan grid`);
    }
    this.poolSize = side / SCAN_SIDE;
    const half = channels / 2;

    this.in_proj = new Conv2dBN(channels, channels);
    // silu distributes over the channel cat, so it rides each half's conv
    // epilogue and costs no dispatch.
    this.conv2d = new ConvPair(half, { activation: 'silu' });
    this.local_conv = new RepDW(channels - dInner, { kernelSize: 3 });
    this.out_proj = new Conv2dBN(channels, channels);
    this.out_norm = new LayerNormC(dInner);

    const f32 = (elems: number) => new Parameter({ elems, dtype: 'f32', dims: [elems] });
    this.x_proj_weight = f32(SSM_DIRS * (dtRank + 2 * SSM_STATE) * dInner);
    this.dt_projs_weight = f32(SSM_DIRS * dInner * dtRank);
    this.dt_projs_bias = f32(SSM_DIRS * dInner);
    this.A_logs = f32(SSM_DIRS * dInner * SSM_STATE);
    this.Ds = f32(SSM_DIRS * dInner);
  }

  override transformStateDict(sd: LazyStateDict, prefix: string): void {
    // prefix is empty when this module is the root.
    const key = prefix ? `${prefix}.A_logs` : 'A_logs';
    const src = sd.tensors.get(key);
    if (!src) throw new Error(`SS2D at '${prefix}': missing 'A_logs' in the state dict`);
    sd.tensors.set(
      key,
      derivedTensor(
        [src.shape.reduce((a, b) => a * b, 1)],
        async () => {
          const logs = await src.f32();
          const out = new Float32Array(logs.length);
          for (let i = 0; i < logs.length; i++) out[i] = -Math.exp(logs[i]!);
          return out;
        },
        src.byteLength,
      ),
    );
  }

  forward(x: Value): Value {
    const { channels, dInner, dtRank, side } = this.cfg;

    const mixed = this.conv2d.forward(this.in_proj.forward(x));

    // Second split: recurrence branch and gate branch.
    const ssmIn = slice(mixed, 0, 0, dInner);
    const gate = this.local_conv.forward(slice(mixed, 0, dInner, channels));

    // Pool to the scan grid, keeping the discarded detail as a high-pass. Only
    // the scan trio and its LayerNorm leave hwc4.
    const pooled =
      this.poolSize === 1
        ? ssmIn
        : avgPool2d(ssmIn, { kernelSize: this.poolSize, stride: this.poolSize });
    const detail =
      this.poolSize === 1 ? undefined : sub(ssmIn, upsample2d(pooled, { scale: this.poolSize }));

    const pooledRows = toChw(pooled, 'f32');
    const proj = ssmScanProject(pooledRows, this.x_proj_weight.value, this.dt_projs_weight.value, {
      rank: dtRank,
    });
    const scanned = ssmSelectiveScan(
      pooledRows,
      proj,
      this.A_logs.value,
      this.Ds.value,
      this.dt_projs_bias.value,
    );
    const merged = ssmScanMerge(scanned, { c: dInner, h: SCAN_SIDE, w: SCAN_SIDE });
    const normed = toHwc4(this.out_norm.forward(merged));

    const restored =
      this.poolSize === 1
        ? normed
        : add(resizeBilinear2d(normed, { outH: side, outW: side, mode: 'halfPixel' }), detail!);

    return this.out_proj.forward(cat([restored, gate], 0));
  }
}

export class ConvPair extends Module {
  readonly dwconv_h: RepDW;
  readonly dwconv_w: RepDW;
  constructor(
    readonly halfChannels: number,
    opts: { activation?: ConvAct } = {},
  ) {
    super();
    this.dwconv_w = new RepDW(halfChannels, { kernelSize: [7, 1], activation: opts.activation });
    this.dwconv_h = new RepDW(halfChannels, { kernelSize: [1, 7], activation: opts.activation });
  }
  forward(x: Value): Value {
    const half = this.halfChannels;
    return cat(
      [
        this.dwconv_w.forward(slice(x, 0, 0, half)),
        this.dwconv_h.forward(slice(x, 0, half, 2 * half)),
      ],
      0,
    );
  }
}

export class LayerNormC extends Module {
  readonly weight: InstanceType<typeof Parameter>;
  readonly bias: InstanceType<typeof Parameter>;
  constructor(readonly channels: number) {
    super();
    this.weight = new Parameter({ elems: channels, dtype: 'f32', dims: [channels] });
    this.bias = new Parameter({ elems: channels, dtype: 'f32', dims: [channels] });
  }
  forward(x: Value): Value {
    const [c, h, w] = x.shape.dims as [number, number, number];
    const rows = transpose(reshape(x, [c, h * w]));
    const normed = layerNorm(rows, this.weight.value, 1e-5, this.bias.value);
    return reshape(transpose(normed), [c, h, w]);
  }
}
