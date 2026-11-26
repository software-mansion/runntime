import { RunntimeError } from '../../../error.ts';
import { hwc4Meta, pending, Value } from '../../value.ts';
import { ACT_CODE, type ActName, wantHwc4 } from '../shared.ts';

/** The activations conv2d can fuse into its epilogue. */
export type ConvAct = Extract<ActName, 'silu' | 'gelu' | 'relu'>;

/** 2D convolution over hwc4-stored f16 activations, with a torch-shaped
 *  surface. Input and output both stay hwc4, and groups are 1 or fully
 *  depthwise. `activation` fuses silu, gelu or relu into the kernel.
 *  `winograd` runs a 3×3, stride 1, padding 1 conv as Winograd F(4×4, 3×3),
 *  with the weight packed by packConvWinogradF16. Throws INVALID_ARGUMENT for
 *  an f32 weight. */
export function conv2d(
  x: Value,
  weight: Value,
  bias: Value | undefined,
  opts: {
    kernelSize: number | [number, number];
    stride: number;
    padding: number;
    groups: number;
    activation?: ConvAct;
    winograd?: boolean;
  },
): Value {
  const [cIn, h, w] = wantHwc4('conv2d', x);
  const { kernelSize, stride, padding, groups } = opts;
  const [kH, kW] = typeof kernelSize === 'number' ? [kernelSize, kernelSize] : kernelSize;
  if (!Number.isInteger(kH) || kH < 1 || !Number.isInteger(kW) || kW < 1) {
    throw new Error(`conv2d: kernelSize must be positive integer(s), got ${kH}×${kW}`);
  }
  const dw = groups === cIn;
  if (groups !== 1 && !(dw && weight.shape.dims![0] === cIn)) {
    throw new Error(
      `conv2d: groups=${groups} is neither 1 nor depthwise (cIn=${cIn}, cOut=${weight.shape.dims![0]})`,
    );
  }
  const wd = weight.shape.dims;
  if (weight.shape.dtype === 'f32') {
    throw new RunntimeError(
      'INVALID_ARGUMENT',
      'conv2d: no f32 kernel yet, run the model in f16 with half()',
    );
  }
  if (weight.shape.dtype !== 'f16' || !wd || wd.length !== 2 || weight.shape.layout !== 'hwc4') {
    throw new Error(
      'conv2d: weight must be the hwc4 mat4-tile packing (nn.Conv2d repacks at load) — an unpacked weight would be read as tiles',
    );
  }
  const [cOut, cin_kk] = wd as [number, number];
  if (cin_kk !== (cIn / groups) * kH * kW) {
    throw new Error(
      `conv2d: weight cols ${cin_kk} != C_in/groups·kH·kW = ${cIn / groups}·${kH}·${kW}`,
    );
  }
  if (bias !== undefined && (bias.shape.dtype !== 'f16' || bias.shape.elems !== cOut)) {
    throw new Error(`conv2d: bias must be f16 with ${cOut} elems (call model.half())`);
  }
  if (opts.winograd && (kH !== 3 || kW !== 3 || stride !== 1 || padding !== 1 || groups !== 1)) {
    throw new Error(
      `conv2d: winograd needs a 3×3, stride 1, padding 1, groups 1 conv, got ${kH}×${kW} stride ${stride} padding ${padding} groups ${groups}`,
    );
  }
  const hOut = Math.floor((h + 2 * padding - kH) / stride) + 1;
  const wOut = Math.floor((w + 2 * padding - kW) / stride) + 1;
  const inputs: Value[] = bias !== undefined ? [x, weight, bias] : [x, weight];
  return pending(hwc4Meta(cOut, hOut, wOut), 'conv2dHwc4', inputs, undefined, [
    kH,
    kW,
    stride,
    padding,
    groups,
    bias !== undefined ? 1 : 0,
    ACT_CODE[opts.activation ?? 'none'],
    opts.winograd ? 1 : 0,
  ]);
}
