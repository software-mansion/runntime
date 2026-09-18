// --- Weights & IO ---
export {
  bf16ToF16Bits,
  bf16ToF32,
  convToTapMajor,
  f16BitsToF32,
  f32ArrayToF16Bits,
  f16ToF32,
  f32ToF16Bits,
  packConvHwc4F16,
  packDwHwc4F16,
  permuteConvTransposeF16,
  transposeF32,
} from './weights/convert.ts';
export {
  createStorageFor,
  readbackF16,
  readbackF32,
  tensor,
  uploadF16,
  uploadF32,
  uploadU32,
  writeF32,
} from './gpu/buffers.ts';
// --- Shared kernel plumbing ---
export { WORKGROUP_SIZE } from './kernels/common.ts';
export { ACC_F16, ACC_F32, getMatmulAccum, setMatmulAccum } from './kernels/matmul/matmul.ts';
export type { F16Buffer, F32Buffer, KernelHandle, U32Buffer, V4Buffer } from './kernels/common.ts';
// --- Eager runtime ---
import './graph/methods.ts'; // attaches the torch-style method forms to Value
export { Value, c4of, elemBytes, hwc4Meta, materialized, matrix, tensor3d } from './graph/value.ts';
export type { EagerDtype, GpuBufferRef, ValueMeta } from './graph/value.ts';
export {
  type ConvAct,
  add,
  argmaxDot,
  asinh,
  cat,
  chunk,
  clamp,
  conv1d,
  conv2d,
  convTranspose2d,
  gatherRows,
  gatherRowsFrom,
  gelu,
  layerNorm,
  matmul,
  matmulGather,
  maxPool2d,
  mean,
  mul,
  reshape,
  rope,
  rsqrt,
  sdpa,
  sdpaPacked,
  sigmoid,
  silu,
  slice,
  softmax,
  split,
  sub,
  swiglu,
  swigluChunk,
  tanh,
  toChw,
  toHwc4,
  avgPool2d,
  channelAffine,
  pad2d,
  resizeBilinear2d,
  ssmScanMerge,
  SSM_DIRS,
  SSM_STATE,
  ssmScanProject,
  ssmSelectiveScan,
  topk,
  astype,
  transpose,
  upsample2d,
  writeRows,
} from './graph/ops.ts';
export { groupNorm, linear, rmsNorm } from './graph/compose.ts';
export * as nn from './nn/index.ts';
export {
  defaultExecutor,
  evalValues,
  gpuExecutor,
  initRunntime,
  toArray,
  type CaptureExecutor,
  type Executor,
  type FusedReadback,
  type KernelTiming,
  type Readback,
  type Releaser,
  type RunntimeExecutor,
} from './gpu/eval.ts';
export {
  createReplayCache,
  type ReplayCache,
  type ReplayCacheConfig,
  type ReplayInputShapes,
} from './gpu/replayCache.ts';
export type { GpuOpTime, GpuPerfSink, GpuSubmitTiming } from './gpu/perf.ts';
export { defaultRoot, resetRunntime, supportsF16 } from './gpu/context.ts';
export { inGpuErrorScopes, warmUp } from './gpu/errorScopes.ts';
export {
  RUNNTIME_ERROR_CODES,
  RunntimeError,
  isRunntimeError,
  type RunntimeErrorCode,
} from './error.ts';
export { createResourceScope, type Disposable, type ResourceScope } from './lifetime.ts';
export { BufferPool } from './gpu/bufferPool.ts';
export { packQuantColSlice } from './weights/quantPack.ts';
export type { WeightCache } from './weights/cache.ts';
export {
  bufferSource,
  chunkedSource,
  httpRangeSource,
  parseSafetensorsHeader,
  derivedTensor,
  eagerTensor,
  fromSafetensors,
  memoryStateDict,
} from './weights/safetensors.ts';
export { cachedRangeSource } from './weights/cachedSource.ts';
export { createOpfsCache } from './weights/opfsCache.ts';
export { foldBnIntoConv } from './weights/foldBn.ts';
export type {
  RangeSource,
  SafeDtype,
  SafetensorsIndex,
  SafeTensorInfo,
  LazyStateDict,
  LazyTensor,
  MemoryStateDict,
} from './weights/safetensors.ts';
export { gpuUpload } from './nn/loadStateDict.ts';
export type { LoadStateDictOpts, UploadFn } from './nn/loadStateDict.ts';
