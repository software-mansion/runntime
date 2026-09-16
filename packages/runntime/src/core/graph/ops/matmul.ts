/** Matrix multiplies: matmul over every weight storage, the expert-indexed
 *  matmulGather, and the fused gemv+argmax argmaxDot. */

import { pending, Value } from '../value.ts';
import { ACT_CODE, FLOAT_DTYPES, want2d, type SlotAct } from './shared.ts';

/** Options for matmul. `bias` and `addend` are fused epilogue adds, not
 *  available for quantW. `baseRow` views into a taller weight without copying,
 *  so MoE experts stay views into one fused tensor. */
export interface MatmulOpts {
  bias?: Value;
  addend?: Value;
  /** Applied to a·b + bias, before any addend, in the same dispatch. */
  activation?: SlotAct;
  baseRow?: number;
  scales?: Value;
  scaleBase?: number;
  bits?: 8 | 4;
  groupSize?: number;
}

/** Activation codes shared by the matmul and conv kernels. */
/** 2D matrix multiply, a [M,K] × b [K,N] to [M,N]. One name for every weight
 *  storage; b's dtype picks the kernel.
 *
 *  Float b supports the fused bias and addend epilogue. quantW b is
 *  dequantized with `scales`, needs bits and groupSize, and has no epilogue. */
export function matmul(a: Value, b: Value, opts: MatmulOpts = {}): Value {
  if (b.shape.dtype === 'quantW') {
    if (opts.bias || opts.addend || opts.activation) {
      throw new Error(
        'matmul: bias/addend/activation epilogue is not supported for quantW weights',
      );
    }
    const { scales, bits, groupSize } = opts;
    if (!scales || !bits || !groupSize) {
      throw new Error('matmul: quantW weights need scales, bits and groupSize');
    }
    return matmulQuantW(a, b, scales, {
      baseRow: opts.baseRow ?? 0,
      scaleBase: opts.scaleBase ?? 0,
      bits,
      groupSize,
    });
  }
  if (opts.scales || opts.bits || opts.groupSize) {
    throw new Error('matmul: scales/bits/groupSize apply to quantW weights only');
  }
  const ad = a.shape.dims;
  const bd = b.shape.dims;
  if (!ad || ad.length !== 2 || !bd || bd.length !== 2) {
    throw new Error('matmul: both inputs must be 2D matrices (use matrix())');
  }
  // The kernel binds one element type for every buffer it touches.
  if (a.shape.dtype !== b.shape.dtype || (a.shape.dtype !== 'f32' && a.shape.dtype !== 'f16')) {
    throw new Error(
      `matmul: both inputs must be f32 or both f16, got ${a.shape.dtype} and ${b.shape.dtype}`,
    );
  }
  const dtype = a.shape.dtype;
  const [m, k] = ad;
  const [rows, n] = bd;
  // With baseRow, b is a taller fused tensor and the view is K rows of it.
  const baseRow = opts.baseRow ?? 0;
  if (!Number.isInteger(baseRow) || baseRow < 0 || baseRow + k! > rows!) {
    throw new Error(
      `matmul: bad baseRow ${baseRow} — need integer >= 0 with baseRow+K (${k}) <= rows (${rows})`,
    );
  }
  if (opts.baseRow === undefined && k !== rows) {
    throw new Error(`matmul: inner dims disagree — a=[${m},${k}] b=[${rows},${n}]`);
  }
  const { bias, addend } = opts;
  if (bias && (bias.shape.dtype !== dtype || bias.shape.elems !== n)) {
    throw new Error(
      `matmul: bias must be ${dtype} [N] = ${n} elems, got (${bias.shape.elems},${bias.shape.dtype})`,
    );
  }
  if (addend && (addend.shape.dtype !== dtype || addend.shape.elems !== m! * n!)) {
    throw new Error(
      `matmul: addend must be ${dtype} [M,N] = ${m! * n!} elems, got (${addend.shape.elems},${addend.shape.dtype})`,
    );
  }
  const inputs = [a, b, ...(bias ? [bias] : []), ...(addend ? [addend] : [])];
  return pending({ elems: m! * n!, dtype, dims: [m!, n!] }, 'matmul', inputs, undefined, [
    bias ? 1 : 0,
    addend ? 1 : 0,
    opts.activation ? ACT_CODE[opts.activation] : 0,
    baseRow,
    opts.baseRow === undefined ? 0 : 1,
  ]);
}

/** matmul's quantW path. bits and groupSize travel as attrs, so one pipeline
 *  serves every quant format. */
function matmulQuantW(
  a: Value,
  w: Value,
  scales: Value,
  opts: { baseRow: number; scaleBase: number; bits: 8 | 4; groupSize: number },
): Value {
  const [m, k] = want2d('matmul', a, ['f32']);
  const wd = w.shape.dims;
  if (!wd || wd.length !== 2) {
    throw new Error('matmul: quantW weight must be 2D');
  }
  if (scales.shape.dtype !== 'f32') throw new Error('matmul: scales must be f32');
  const [r, n] = [wd[0]!, wd[1]!];
  const { baseRow, scaleBase, bits, groupSize } = opts;
  if (bits !== 8 && bits !== 4) throw new Error(`matmul: bits must be 8 or 4, got ${bits}`);
  const perU32 = bits === 8 ? 4 : 8;
  if (n % perU32 !== 0)
    throw new Error(`matmul: cols ${n} must be divisible by ${perU32} for int${bits} packing`);
  if (w.shape.elems !== (r * n) / perU32) {
    throw new Error(
      `matmul: quantW weight elems ${w.shape.elems} != rows*cols/${perU32} = ${(r * n) / perU32} words`,
    );
  }
  if (groupSize <= 0 || k % groupSize !== 0)
    throw new Error(`matmul: groupSize ${groupSize} must divide K ${k}`);
  if (k % 4 !== 0 || groupSize % 4 !== 0) {
    throw new Error(
      `matmul: K ${k} and groupSize ${groupSize} must be multiples of 4 (blocked-unit tiles)`,
    );
  }
  if (!Number.isInteger(baseRow) || baseRow < 0 || baseRow + k > r) {
    throw new Error(`matmul: bad baseRow ${baseRow} — need >= 0 with baseRow+K (${k}) <= R (${r})`);
  }
  if (baseRow % k !== 0) {
    throw new Error(
      `matmul: baseRow ${baseRow} must be K-aligned (${k}) — blocked-unit tiles are expert-granular`,
    );
  }
  if (!Number.isInteger(scaleBase) || scaleBase < 0)
    throw new Error(`matmul: bad scaleBase ${scaleBase}`);
  return pending(
    { elems: m * n, dtype: 'f32', dims: [m, n] },
    'matmulQuantW',
    [a, w, scales],
    undefined,
    [baseRow, scaleBase, bits, groupSize],
  );
}

/** matmulGather's quantW path: out[i] = a[i] · dequant(W[e_i]) + bias[e_i],
 *  with e_i read per row on the GPU. */
function matmulGatherQuantW(
  a: Value,
  w: Value,
  scales: Value,
  bias: Value,
  expertIdx: Value,
  opts: { bits: 8 | 4; groupSize: number },
): Value {
  const [m, k] = want2d('matmulGather', a, ['f32']);
  const wd = w.shape.dims;
  if (!wd || wd.length !== 2) {
    throw new Error('matmulGather: quantW weight must be 2D');
  }
  const [r, n] = [wd[0]!, wd[1]!];
  const { bits, groupSize } = opts;
  if (bits !== 8 && bits !== 4) throw new Error(`matmulGather: bits must be 8 or 4, got ${bits}`);
  const perU32 = bits === 8 ? 4 : 8;
  if (n % perU32 !== 0) throw new Error(`matmulGather: cols ${n} must be divisible by ${perU32}`);
  if (w.shape.elems !== (r * n) / perU32) {
    throw new Error(`matmulGather: quantW weight elems ${w.shape.elems} != rows*cols/${perU32}`);
  }
  if (r % k !== 0) throw new Error(`matmulGather: w rows ${r} not a multiple of K ${k}`);
  if (groupSize <= 0 || k % groupSize !== 0)
    throw new Error(`matmulGather: groupSize ${groupSize} must divide K ${k}`);
  if (k % 4 !== 0 || groupSize % 4 !== 0) {
    throw new Error(
      `matmulGather: K ${k} and groupSize ${groupSize} must be multiples of 4 (blocked-unit tiles)`,
    );
  }
  if (scales.shape.dtype !== 'f32') throw new Error('matmulGather: scales must be f32');
  if (bias.shape.dtype !== 'f32') throw new Error('matmulGather: bias must be f32');
  if (expertIdx.shape.dtype !== 'f32' || expertIdx.shape.elems !== m) {
    throw new Error(
      `matmulGather: expertIdx must be f32 with ${m} elems, got ${expertIdx.shape.elems}`,
    );
  }
  return pending(
    { elems: m * n, dtype: 'f32', dims: [m, n] },
    'matmulGatherQuantW',
    [a, w, scales, bias, expertIdx],
    undefined,
    [bits, groupSize],
  );
}

/** Indexed matmul plus bias: out[i] = a[i] · W[e_i] + bias[e_i], with e_i
 *  looked up per row, so routing stays on the GPU with no readback. MLX calls
 *  this gather_qmm.
 *
 *  a and the fused w [E·K, N] share one float dtype, or w is quantW, which
 *  additionally needs scales, bits and groupSize. */
export function matmulGather(
  a: Value,
  w: Value,
  expertIdx: Value,
  opts: { bias: Value; scales?: Value; bits?: 8 | 4; groupSize?: number },
): Value {
  if (w.shape.dtype === 'f32' || w.shape.dtype === 'f16') {
    if (opts.scales || opts.bits || opts.groupSize) {
      throw new Error('matmulGather: scales/bits/groupSize apply to quantW weights only');
    }
    return matmulGatherFloat(a, w, opts.bias, expertIdx);
  }
  if (w.shape.dtype === 'quantW') {
    const { scales, bits, groupSize } = opts;
    if (!scales || !bits || !groupSize) {
      throw new Error('matmulGather: quantW weights need scales, bits and groupSize');
    }
    return matmulGatherQuantW(a, w, scales, opts.bias, expertIdx, { bits, groupSize });
  }
  throw new Error(`matmulGather: w must be a float or quantW weight, got ${w.shape.dtype}`);
}

/** matmulGather's float path. expertIdx holds row numbers, so it stays f32
 *  whatever the model is. N must be divisible by 4. */
function matmulGatherFloat(a: Value, w: Value, bias: Value, expertIdx: Value): Value {
  const [m, k, dtype] = want2d('matmulGather', a, FLOAT_DTYPES);
  const [r, n, wDtype] = want2d('matmulGather.w', w, FLOAT_DTYPES);
  if (wDtype !== dtype) {
    throw new Error(`matmulGather: w is ${wDtype} but a is ${dtype}`);
  }
  if (n % 4 !== 0) throw new Error(`matmulGather: n ${n} must be divisible by 4`);
  if (r % k !== 0) throw new Error(`matmulGather: w rows ${r} not a multiple of K ${k}`);
  if (bias.shape.dtype !== dtype) {
    throw new Error(`matmulGather: bias must be ${dtype}, got ${bias.shape.dtype}`);
  }
  if (expertIdx.shape.dtype !== 'f32' || expertIdx.shape.elems !== m) {
    throw new Error(
      `matmulGather: expertIdx must be f32 with ${m} elems, got ${expertIdx.shape.elems}`,
    );
  }
  return pending({ elems: m * n, dtype, dims: [m, n] }, 'matmulGather', [a, w, bias, expertIdx]);
}

/** Fused gemv and argmax for greedy decoding: w [V,N] · x [1,N] to the argmax
 *  row index, lowest on ties. The logits vector never materializes, so the
 *  readback is one float.
 *
 *  x must be one row; teacher-forced paths use matmul. The result is always
 *  f32, since f16 is only exact to 2048 and would decode the wrong token on a
 *  larger vocabulary. */
export function argmaxDot(w: Value, x: Value): Value {
  const [, n, wDtype] = want2d('argmaxDot.w', w, FLOAT_DTYPES);
  const [xr, xn, xDtype] = want2d('argmaxDot.x', x, FLOAT_DTYPES);
  if (xr !== 1) throw new Error(`argmaxDot: x must be a single row, got ${xr}`);
  if (xn !== n) throw new Error(`argmaxDot: x cols ${xn} != w cols ${n}`);
  if (wDtype !== xDtype) {
    throw new Error(`argmaxDot: w dtype ${wDtype} must match x's ${xDtype}`);
  }
  return pending({ elems: 1, dtype: 'f32', dims: [1, 1] }, 'argmaxDot', [w, x]);
}
