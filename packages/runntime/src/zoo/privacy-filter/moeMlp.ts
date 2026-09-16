/** The privacy filter's MoE MLP block, with topk routing on the GPU.
 *
 *  A forward runs in two eval phases: `selection` computes the routing, and the
 *  caller reads it back and groups it to shape `forward`, which builds the
 *  expert-compute DAG. `forwardPacked` does both in one DAG instead.
 *
 *  The residual is folded in here, so composing callers must not re-add it.
 *  K is assumed equal to H. */

import {
  astype,
  add,
  cat,
  gatherRows,
  matmul,
  matmulGather,
  matrix,
  mul,
  slice,
  swiglu,
  topk,
  nn,
  type Value,
  type ValueMeta,
} from '../../core/index.ts';
import type { Grouping } from './routing.ts';

const RMS_EPS = 1e-5; // legacy rmsnorm kernel epsilon (not the eager 1e-6 default)

export interface MoeDims {
  experts: number;
  hidden: number; // H (== K for the privacy filter)
  weightFormat: 'f32' | 'quantInt8' | 'quantInt4';
  groupSize?: number;
}

export class EagerMoeMlp extends nn.Module<[Value, Grouping, Value[]], Value> {
  readonly norm: InstanceType<typeof nn.RMSNorm>;
  readonly gate: InstanceType<typeof nn.Linear>;
  readonly gluWeight: InstanceType<typeof nn.Parameter>;
  readonly linWeight: InstanceType<typeof nn.Parameter>;
  readonly outWeight: InstanceType<typeof nn.Parameter>;
  readonly gluBias: InstanceType<typeof nn.Parameter>;
  readonly linBias: InstanceType<typeof nn.Parameter>;
  readonly outBias: InstanceType<typeof nn.Parameter>;
  readonly gluScales?: InstanceType<typeof nn.Parameter>;
  readonly linScales?: InstanceType<typeof nn.Parameter>;
  readonly outScales?: InstanceType<typeof nn.Parameter>;

  constructor(readonly dims: MoeDims) {
    super();
    const { experts: e, hidden: h, weightFormat } = dims;
    const quant = weightFormat === 'quantInt8' || weightFormat === 'quantInt4';
    const perU32 = weightFormat === 'quantInt8' ? 4 : weightFormat === 'quantInt4' ? 8 : 0;
    const groupSize = dims.groupSize ?? h;
    if (quant && h % groupSize !== 0)
      throw new Error(`EagerMoeMlp: groupSize ${groupSize} must divide hidden ${h}`);
    const kg = quant ? h / groupSize : 0;
    this.norm = new nn.RMSNorm(h, RMS_EPS);
    this.gate = new nn.Linear(h, e);
    // Logical dims in every format; quantW's elems is a u32 word count.
    const weightMeta = (): ValueMeta =>
      quant
        ? { elems: (e * h * h) / perU32, dtype: 'quantW', dims: [e * h, h] }
        : { elems: e * h * h, dtype: 'f32', dims: [e * h, h] };
    this.gluWeight = new nn.Parameter(weightMeta());
    this.linWeight = new nn.Parameter(weightMeta());
    this.outWeight = new nn.Parameter(weightMeta());
    this.gluBias = new nn.Parameter({ elems: e * h, dtype: 'f32', dims: [e, h] });
    this.linBias = new nn.Parameter({ elems: e * h, dtype: 'f32', dims: [e, h] });
    this.outBias = new nn.Parameter({ elems: e * h, dtype: 'f32', dims: [e, h] });
    if (quant) {
      const scaleMeta: ValueMeta = { elems: e * kg * h, dtype: 'f32', dims: [e * kg, h] };
      this.gluScales = new nn.Parameter(scaleMeta);
      this.linScales = new nn.Parameter(scaleMeta);
      this.outScales = new nn.Parameter(scaleMeta);
    }
  }

  routerLogits(x: Value): Value {
    return this.gate.forward(this.norm.forward(x));
  }

  selection(x: Value): Value {
    return topk(this.routerLogits(x), 4);
  }

  private expertMatmul(x: Value, w: Value, scales: Value | undefined, expert: number): Value {
    const h = this.dims.hidden;
    const f = this.dims.weightFormat;
    if (f === 'f32') return matmul(x, w, { baseRow: expert * h });
    if (f === 'quantInt8' || f === 'quantInt4') {
      const bits = f === 'quantInt8' ? 8 : 4;
      const groupSize = this.dims.groupSize ?? h;
      const kg = h / groupSize;
      return matmul(x, w, {
        scales: scales!,
        baseRow: expert * h,
        scaleBase: expert * kg * h,
        bits,
        groupSize,
      });
    }
    return matmul(x, slice(w, 0, expert * h, (expert + 1) * h));
  }

  private get actDtype(): 'f32' | 'f16' {
    return this.norm.scale.shape.dtype === 'f16' ? 'f16' : 'f32';
  }

  forwardGather(x: Value): Value {
    const f = this.dims.weightFormat;

    const xn = this.norm.forward(x); // [T, H]
    const packed = topk(this.gate.forward(xn), 4); // [T, 8]: 4 ids ‖ 4 weights, on GPU
    let y: Value | undefined;

    if (f === 'f32') {
      for (let s = 0; s < 4; s++) {
        const ids = slice(packed, 1, s, s + 1);
        // topk's output is f32 in every variant; narrow it to the activation
        // dtype.
        const wgt = astype(slice(packed, 1, 4 + s, 5 + s), this.actDtype);
        const glu = matmulGather(xn, this.gluWeight.value, ids, { bias: this.gluBias.value });
        const lin = matmulGather(xn, this.linWeight.value, ids, { bias: this.linBias.value });
        const act = swiglu(glu, lin);
        const out = matmulGather(act, this.outWeight.value, ids, { bias: this.outBias.value });
        const contrib = mul(out, wgt);
        y = y ? add(y, contrib) : contrib;
      }
      return add(x, y!);
    }

    const bits = f === 'quantInt8' ? 8 : 4;
    const h = this.dims.hidden;
    const groupSize = this.dims.groupSize ?? h;
    const gather = (a: Value, w: Value, scales: Value, bias: Value, ids: Value): Value =>
      matmulGather(a, w, ids, { bias, scales, bits, groupSize });
    for (let s = 0; s < 4; s++) {
      const ids = slice(packed, 1, s, s + 1); // [T,1] expert id for slot s
      const wgt = astype(slice(packed, 1, 4 + s, 5 + s), this.actDtype); // [T,1] gate weight
      const glu = gather(xn, this.gluWeight.value, this.gluScales!.value, this.gluBias.value, ids);
      const lin = gather(xn, this.linWeight.value, this.linScales!.value, this.linBias.value, ids);
      const act = swiglu(glu, lin); // [T, H]
      const out = gather(act, this.outWeight.value, this.outScales!.value, this.outBias.value, ids);
      const contrib = mul(out, wgt); // [T,H] × [T,1] col-broadcast
      y = y ? add(y, contrib) : contrib;
    }
    return add(x, y!); // residual folded in
  }

  override forward(x: Value, grouping: Grouping, gateColumns: Value[]): Value {
    if (!grouping)
      throw new Error(
        'EagerMoeMlp.forward needs (x, grouping, gateColumns) — evaluate selection(x) and run parseTopk/groupByExpert first (see routing.ts)',
      );
    if (grouping.groups.length === 0) throw new Error('EagerMoeMlp.forward: empty grouping');
    if (gateColumns.length !== grouping.groups.length)
      throw new Error(
        `EagerMoeMlp.forward: ${gateColumns.length} gate columns for ${grouping.groups.length} expert groups`,
      );
    const xn = this.norm.forward(x);
    const contribs: Value[] = [];
    grouping.groups.forEach((g, gi) => {
      const rows = gatherRows(xn, g.tokens); // [n_g, H]
      const e = g.expert;
      const glu = add(
        this.expertMatmul(rows, this.gluWeight.value, this.gluScales?.value, e),
        slice(this.gluBias.value, 0, e, e + 1),
      );
      const lin = add(
        this.expertMatmul(rows, this.linWeight.value, this.linScales?.value, e),
        slice(this.linBias.value, 0, e, e + 1),
      );
      const act = swiglu(glu, lin); // [n_g, H]
      const o = add(
        this.expertMatmul(act, this.outWeight.value, this.outScales?.value, e),
        slice(this.outBias.value, 0, e, e + 1),
      );
      contribs.push(mul(o, gateColumns[gi]!)); // [n_g,1] col-broadcast
    });
    const pile = cat(contribs, 0); // [4T, H] in group order
    let y: Value | undefined;
    for (let s = 0; s < 4; s++) {
      const slot = gatherRows(pile, grouping.slotPos[s]!); // [T, H]
      y = y ? add(y, slot) : slot;
    }
    return add(x, y!); // residual folded in
  }
}

export function gateColumnShape(n: number) {
  return matrix(n, 1);
}
