import { pending, Value } from '../../value.ts';
import { FLOAT_DTYPES, want3d } from '../shared.ts';

/** Transposed 2D convolution, restricted to the no-overlap case kernelSize ===
 *  stride, padding 0 (torch nn.ConvTranspose2d(cIn, cOut, k, k)): x=[C_in,H,W]
 *  → [C_out, H·k, W·k]. With k === s every output pixel gets exactly one
 *  contribution: out[co, k·i+di, k·j+dj] = Σ_ci W[ci,co,di,dj]·x[ci,i,j] + b.
 *  Weight is f16 [C_out·k·k, C_in] — the exporter pre-permutes torch's
 *  [C_in, C_out, kH, kW] so row co·k²+di·k+dj holds the (di,dj) tap of output
 *  channel co (contiguous C_in per row → sequential f16-pair decode).
 *  Generalizing to k ≠ s later = a gather path behind this same signature.
 *  attrs: [k, hasBias] */
export function convTranspose2d(
  x: Value,
  weight: Value,
  bias: Value | undefined,
  opts: { kernelSize: number; stride: number },
): Value {
  const [cIn, h, w] = want3d('convTranspose2d', x, FLOAT_DTYPES);
  const { kernelSize: k, stride } = opts;
  if (!Number.isInteger(k) || k < 1 || !Number.isInteger(stride)) {
    throw new Error(
      `convTranspose2d: kernelSize and stride must be positive integers, got k=${k} stride=${stride}`,
    );
  }
  if (stride !== k) {
    throw new Error(
      `convTranspose2d: only kernelSize === stride (no output overlap) is implemented, got k=${k} stride=${stride}`,
    );
  }
  const wd = weight.shape.dims;
  if (weight.shape.dtype !== 'f16' || !wd || wd.length !== 2) {
    throw new Error('convTranspose2d: weight must be 2D f16 [C_out·k·k, C_in]');
  }
  const [rows, wCin] = wd as [number, number];
  if (wCin !== cIn) {
    throw new Error(`convTranspose2d: weight C_in ${wCin} != input C_in ${cIn}`);
  }
  if (rows % (k * k) !== 0) {
    throw new Error(`convTranspose2d: weight rows ${rows} not a multiple of k²=${k * k}`);
  }
  const cOut = rows / (k * k);
  if (bias !== undefined && (bias.shape.dtype !== x.shape.dtype || bias.shape.elems !== cOut)) {
    throw new Error(`convTranspose2d: bias must be ${x.shape.dtype} with ${cOut} elems`);
  }
  const inputs: Value[] = bias !== undefined ? [x, weight, bias] : [x, weight];
  return pending(
    { elems: cOut * h * k * w * k, dtype: x.shape.dtype, dims: [cOut, h * k, w * k] },
    'convTranspose2d',
    inputs,
    undefined,
    [k, bias !== undefined ? 1 : 0],
  );
}
