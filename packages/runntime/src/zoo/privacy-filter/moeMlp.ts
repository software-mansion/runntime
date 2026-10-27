/** The privacy filter's MoE MLP block. Topk routing stays on the GPU: the
 *  expert matmuls read the selected ids straight from topk's output, so a
 *  forward is one DAG with no readback.
 *
 *  The residual is folded in here, so composing callers must not re-add it.
 *  K is assumed equal to H. */

import {
  astype,
  add,
  matmulGather,
  moeDown,
  moeUp,
  mul,
  slice,
  swiglu,
  topk,
  nn,
  type Value,
  type ValueMeta,
} from '../../core/index.ts';

const RMS_EPS = 1e-5; // legacy rmsnorm kernel epsilon (not the eager 1e-6 default)

export interface MoeDims {
  experts: number;
  hidden: number; // H (== K for the privacy filter)
  weightFormat: 'f32' | 'quantInt8' | 'quantInt4';
  groupSize?: number;
}

export class EagerMoeMlp extends nn.Module {
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

  private get actDtype(): 'f32' | 'f16' {
    return this.norm.scale.shape.dtype === 'f16' ? 'f16' : 'f32';
  }

  override forward(x: Value): Value {
    const f = this.dims.weightFormat;

    const xn = this.norm.forward(x); // [T, H]
    const packed = topk(this.gate.forward(xn), 4); // [T, 8]: 4 ids ‖ 4 weights, on GPU

    if (f === 'f32') {
      let y: Value | undefined;
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
    const weight = (w: nn.Parameter, scales: nn.Parameter, bias: nn.Parameter) => ({
      w: w.value,
      scales: scales.value,
      bias: bias.value,
    });
    const act = moeUp(
      xn,
      packed,
      weight(this.gluWeight, this.gluScales!, this.gluBias),
      weight(this.linWeight, this.linScales!, this.linBias),
      { bits, groupSize },
    );
    // moeDown adds the residual.
    return moeDown(act, packed, weight(this.outWeight, this.outScales!, this.outBias), x, {
      bits,
      groupSize,
    });
  }
}
