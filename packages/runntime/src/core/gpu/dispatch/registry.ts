/** Op name → KernelSpec: how the GPU executor runs each op a pending node
 *  can name. `satisfies Record<OpName, AnySpec>` makes a missing or extra
 *  entry a type error here. */

import type { OpName } from '../../graph/value.ts';
import type { AnySpec } from './spec.ts';
import {
  addScalarSpec,
  addSpec,
  asinhSpec,
  clampScalarSpec,
  geluSpec,
  mulScalarSpec,
  mulSpec,
  rsqrtSpec,
  sigmoidSpec,
  siluSpec,
  subSpec,
  swigluChunkSpec,
  swigluSpec,
  tanhSpec,
} from './elementwise.ts';
import { layerNormSpec, meanSpec, meanSquareSpec, softmaxSpec, topkSpec } from './reduce.ts';
import { attnSpec, ropeSpec } from './attention.ts';
import {
  avgPool2dHwc4Spec,
  channelAffineHwc4Spec,
  concatChannelsHwc4Spec,
  conv1dSpec,
  conv2dHwc4Spec,
  convTranspose2dSpec,
  maxPool2dHwc4Spec,
  pad2dHwc4Spec,
  resizeBilinearHwc4Spec,
  sliceChannelsHwc4Spec,
  toChwSpec,
  toHwc4Spec,
  upsample2dHwc4Spec,
} from './conv.ts';
import { ssmScanMergeSpec, ssmScanProjectSpec, ssmSelectiveScanSpec } from './ssm.ts';
import {
  argmaxDotSpec,
  matmulGatherQuantWSpec,
  matmulGatherSpec,
  matmulQuantWSpec,
  matmulSpec,
} from './matmul.ts';
import {
  astypeSpec,
  concatChannelsSpec,
  concatColsSpec,
  concatRowsSpec,
  gatherRowsFromSpec,
  gatherRowsSpec,
  reshapeSpec,
  sliceChannelsSpec,
  sliceColsSpec,
  sliceRowsSpec,
  transposeSpec,
  writeRowsSpec,
} from './shape.ts';

export const specs = {
  // elementwise
  add: addSpec,
  mul: mulSpec,
  sub: subSpec,
  swiglu: swigluSpec,
  swigluChunk: swigluChunkSpec,
  rsqrt: rsqrtSpec,
  addScalar: addScalarSpec,
  sigmoid: sigmoidSpec,
  mulScalar: mulScalarSpec,
  clampScalar: clampScalarSpec,
  tanh: tanhSpec,
  gelu: geluSpec,
  silu: siluSpec,
  asinh: asinhSpec,
  // matmul
  matmul: matmulSpec,
  matmulQuantW: matmulQuantWSpec,
  matmulGatherQuantW: matmulGatherQuantWSpec,
  matmulGather: matmulGatherSpec,
  argmaxDot: argmaxDotSpec,
  // attention
  attn: attnSpec,
  rope: ropeSpec,
  // conv
  conv1d: conv1dSpec,
  convTranspose2d: convTranspose2dSpec,
  conv2dHwc4: conv2dHwc4Spec,
  maxPool2dHwc4: maxPool2dHwc4Spec,
  upsample2dHwc4: upsample2dHwc4Spec,
  avgPool2dHwc4: avgPool2dHwc4Spec,
  pad2dHwc4: pad2dHwc4Spec,
  resizeBilinearHwc4: resizeBilinearHwc4Spec,
  channelAffineHwc4: channelAffineHwc4Spec,
  concatChannelsHwc4: concatChannelsHwc4Spec,
  sliceChannelsHwc4: sliceChannelsHwc4Spec,
  toHwc4: toHwc4Spec,
  toChw: toChwSpec,
  // ssm
  ssmScanProject: ssmScanProjectSpec,
  ssmSelectiveScan: ssmSelectiveScanSpec,
  ssmScanMerge: ssmScanMergeSpec,
  // reduce
  softmax: softmaxSpec,
  mean: meanSpec,
  meanSquare: meanSquareSpec,
  layerNorm: layerNormSpec,
  topk: topkSpec,
  // shape
  astype: astypeSpec,
  transpose: transposeSpec,
  sliceCols: sliceColsSpec,
  concatCols: concatColsSpec,
  sliceRows: sliceRowsSpec,
  concatRows: concatRowsSpec,
  reshape: reshapeSpec,
  sliceChannels: sliceChannelsSpec,
  concatChannels: concatChannelsSpec,
  writeRows: writeRowsSpec,
  gatherRowsFrom: gatherRowsFromSpec,
  gatherRows: gatherRowsSpec,
} as const satisfies Record<OpName, AnySpec>;

export function specFor(op: OpName): AnySpec {
  return specs[op];
}
