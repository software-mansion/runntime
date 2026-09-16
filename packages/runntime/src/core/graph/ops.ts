/** Eager op constructors. Each returns a PENDING Value wired to its inputs and
 *  dispatches nothing. Shape/dtype rules run in the constructors (cheap,
 *  pre-dispatch). This file is the public surface; the implementations live
 *  in ops/ grouped by domain. */

export {
  add,
  asinh,
  clamp,
  gelu,
  mul,
  rsqrt,
  sigmoid,
  silu,
  sub,
  swiglu,
  swigluChunk,
  tanh,
} from './ops/elementwise.ts';
export { argmaxDot, matmul, matmulGather, type MatmulOpts } from './ops/matmul.ts';
export { rope, sdpa, sdpaPacked } from './ops/attention.ts';
export {
  cat,
  chunk,
  gatherRows,
  gatherRowsFrom,
  reshape,
  slice,
  split,
  astype,
  transpose,
  writeRows,
} from './ops/shape.ts';
export { layerNorm, mean, meanSquare, softmax, topk } from './ops/reduce.ts';
export { channelAffine, toChw, toHwc4 } from './ops/hwc4.ts';
export { conv1d } from './ops/conv/conv1d.ts';
export { SSM_DIRS, SSM_STATE, ssmScanMerge, ssmScanProject, ssmSelectiveScan } from './ops/ssm.ts';
export { type ConvAct, conv2d } from './ops/conv/conv2d.ts';
export { ACT_CODE, type ActName, type SlotAct } from './ops/shared.ts';
export { convTranspose2d } from './ops/conv/convTranspose2d.ts';
export { avgPool2d, maxPool2d, pad2d, resizeBilinear2d, upsample2d } from './ops/conv/pool.ts';
