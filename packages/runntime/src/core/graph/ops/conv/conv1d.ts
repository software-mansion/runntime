import { pending, Value } from '../../value.ts';
import { ACT_CODE, FLOAT_DTYPES, want2d, type SlotAct } from '../shared.ts';

/** Strided 1-D convolution over time-major [T_in, C_in], with w tap-major
 *  [kernelSize·C_in, C_out].
 *
 *  `padding` takes torch's number, 'valid' or 'same'; padLeft and padRight pad
 *  each side independently instead, and kernelSize−1 on the left is causal.
 *  Passing both is an error, and every pad must stay below kernelSize — an
 *  output frame computed from padding alone is a model bug.
 *
 *  `bias` and `activation` fold into the epilogue as act(conv + bias). */
export function conv1d(
  x: Value,
  w: Value,
  opts: {
    kernelSize: number;
    stride: number;
    padding?: number | 'valid' | 'same';
    padLeft?: number;
    padRight?: number;
    bias?: Value;
    activation?: SlotAct;
  },
): Value {
  const { kernelSize, stride, padding, bias, activation } = opts;
  if (padding !== undefined && (opts.padLeft !== undefined || opts.padRight !== undefined)) {
    throw new Error('conv1d: padding excludes padLeft/padRight — give one form or the other');
  }
  const [tIn, cIn, dtype] = want2d('conv1d', x, FLOAT_DTYPES);
  const [wRows, cOut, wDtype] = want2d('conv1d.w', w, FLOAT_DTYPES);
  if (wDtype !== dtype) {
    throw new Error(`conv1d: weight dtype ${wDtype} must match the input's ${dtype}`);
  }
  if (!Number.isInteger(kernelSize) || kernelSize < 1 || !Number.isInteger(stride) || stride < 1) {
    throw new Error(
      `conv1d: kernelSize ${kernelSize} and stride ${stride} must be positive integers`,
    );
  }
  let padLeft: number;
  let padRight: number;
  if (padding === 'valid') {
    padLeft = 0;
    padRight = 0;
  } else if (padding === 'same') {
    if (stride !== 1) {
      throw new Error(`conv1d: padding 'same' requires stride 1, got ${stride}`);
    }
    padLeft = Math.floor((kernelSize - 1) / 2);
    padRight = kernelSize - 1 - padLeft;
  } else {
    padLeft = padding ?? opts.padLeft ?? 0;
    padRight = padding ?? opts.padRight ?? 0;
  }
  for (const [name, pad] of [
    ['padLeft', padLeft],
    ['padRight', padRight],
  ] as const) {
    if (!Number.isInteger(pad) || pad < 0 || pad >= kernelSize) {
      const given = typeof padding === 'number' ? `padding ${padding}` : `${name} ${pad}`;
      throw new Error(`conv1d: ${given} must be an integer in [0, kernelSize)`);
    }
  }
  if (wRows !== kernelSize * cIn) {
    throw new Error(`conv1d: w rows ${wRows} != kernelSize·C_in = ${kernelSize}·${cIn}`);
  }
  if (tIn + padLeft + padRight < kernelSize) {
    throw new Error(
      `conv1d: padded frames ${tIn}+${padLeft}+${padRight} < kernelSize ${kernelSize} — no output`,
    );
  }
  if (bias && (bias.shape.dtype !== dtype || bias.shape.elems !== cOut)) {
    throw new Error(
      `conv1d: bias must be ${dtype} [C_out] = ${cOut} elems, got (${bias.shape.elems},${bias.shape.dtype})`,
    );
  }
  const act = activation ? ACT_CODE[activation] : 0;
  const tOut = Math.floor((tIn + padLeft + padRight - kernelSize) / stride) + 1;
  return pending(
    { elems: tOut * cOut, dtype, dims: [tOut, cOut] },
    'conv1d',
    [x, w, ...(bias ? [bias] : [])],
    undefined,
    [kernelSize, stride, padLeft, padRight, bias ? 1 : 0, act],
  );
}

/** GLU then a causal depthwise conv over time, in one dispatch. x [T, 2C] is
 *  clamped to [lo, hi] and gated, first half by σ of the second; w tap-major
 *  [kernelSize, C] then mixes each channel's last kernelSize frames.
 *  Returns [T, C]. C must be a multiple of 4. */
export function gluConv1d(x: Value, w: Value, opts: { lo: number; hi: number }): Value {
  const [t, cols, dtype] = want2d('gluConv1d', x, FLOAT_DTYPES);
  const [kernelSize, c, wDtype] = want2d('gluConv1d.w', w, FLOAT_DTYPES);
  if (wDtype !== dtype) {
    throw new Error(`gluConv1d: weight dtype ${wDtype} must match the input's ${dtype}`);
  }
  if (cols !== 2 * c || c % 4 !== 0) {
    throw new Error(`gluConv1d: x cols ${cols} must be 2·C for weight [K, C], C a multiple of 4`);
  }
  const { lo, hi } = opts;
  if (!(lo <= hi)) throw new Error(`gluConv1d: lo (${lo}) must be <= hi (${hi})`);
  return pending({ elems: t * c, dtype, dims: [t, c] }, 'gluConv1d', [x, w], undefined, [
    kernelSize,
    lo,
    hi,
  ]);
}
