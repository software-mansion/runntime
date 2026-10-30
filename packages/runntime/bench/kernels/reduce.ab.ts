/** The row reductions (softmax, layerNorm, mean, topk, logitStats) and the
 *  norms composed from them (rmsNorm, groupNorm). softmax and rmsNorm pick
 *  vec4 when the row width (rmsNorm: the group) is a multiple of 4, scalar
 *  otherwise, each with and without subgroups. The rest pick no route; the
 *  cases vary dtype, row count and row width. */
import { abSuite, type AbCase, type Version } from '../ab.ts';
import type { Value } from '../../src/core/index.ts';
import { randMat } from './inputs.ts';

type Float = 'f32' | 'f16';

function rows(
  name: string,
  m: number,
  n: number,
  dtype: Float,
  op: (version: Version, x: Value) => () => Value,
): AbCase {
  return {
    name,
    needs: dtype === 'f16' ? ['shader-f16'] : [],
    build: (version) => op(version, randMat(version, m, n, dtype)),
  };
}

const softmax = (name: string, m: number, n: number, dtype: Float) =>
  rows(name, m, n, dtype, (version, x) => () => version.lib.softmax(x));

const layerNorm = (name: string, m: number, n: number, dtype: Float, bias: boolean) =>
  rows(name, m, n, dtype, (version, x) => {
    const w = randMat(version, 1, n, dtype);
    const b = bias ? randMat(version, 1, n, dtype) : undefined;
    return () => version.lib.layerNorm(x, w, 1e-5, b);
  });

const rmsNorm = (name: string, m: number, n: number, dtype: Float) =>
  rows(name, m, n, dtype, (version, x) => {
    const w = randMat(version, 1, n, dtype);
    return () => version.lib.rmsNorm(x, w, 1e-5);
  });

/** Echo's sampler: `blocks` codebooks of 1025 logits per row, then 2 padding
 *  columns. The last logit of each codebook is the mask, which the argmax
 *  skips. */
const logitStats = (name: string, m: number, blocks: number, guided: boolean): AbCase =>
  rows(name, m, blocks * 1025 + 2, 'f16', (version, x) => {
    const guide = guided ? randMat(version, m, blocks * 1025 + 2, 'f16') : undefined;
    const call = () =>
      version.lib.logitStats(x, { blocks, block: 1025, argmaxCols: 1024, guide, guideWeight: 1.5 });
    // Built once here, so a base without logitStats skips the case.
    call();
    return call;
  });

/** The same route on a device without subgroups. */
function noSubgroups(testCase: AbCase): AbCase {
  return { ...testCase, name: `${testCase.name}, no subgroups`, noSubgroups: true };
}

abSuite('reduce', [
  softmax('softmax f16, YOLO26n PSA [400,400]', 400, 400, 'f16'),
  softmax('softmax, MiniLM scores [1536,128]', 1536, 128, 'f32'),
  softmax('softmax, one wide row [1,32768]', 1, 32768, 'f32'),
  softmax('softmax, odd row count [37,1000]', 37, 1000, 'f32'),
  // scalar: cols % 4 != 0
  softmax('softmax scalar, odd sizes [37,1001]', 37, 1001, 'f32'),
  softmax('softmax scalar f16, odd sizes [37,1001]', 37, 1001, 'f16'),
  noSubgroups(softmax('softmax, MiniLM scores [1536,128]', 1536, 128, 'f32')),
  noSubgroups(softmax('softmax scalar, odd sizes [37,1001]', 37, 1001, 'f32')),

  layerNorm('layerNorm + bias, MiniLM [128,384]', 128, 384, 'f32', true),
  layerNorm('layerNorm f16 + bias, MiniLM [512,384]', 512, 384, 'f16', true),
  layerNorm('layerNorm, streaming CMVN [500,80]', 500, 80, 'f32', false),
  layerNorm('layerNorm, wide rows [16,4096]', 16, 4096, 'f32', true),

  rows(
    'mean f16, MobileNetV4 global pool [960,49]',
    960,
    49,
    'f16',
    (version, x) => () => version.lib.mean(x),
  ),
  rows('mean, one long row [1,65536]', 1, 65536, 'f32', (version, x) => () => version.lib.mean(x)),

  rows(
    'topk 4, privacy-filter router [512,128]',
    512,
    128,
    'f32',
    (version, x) => () => version.lib.topk(x, 4),
  ),
  rows('topk 4 f16, router [64,128]', 64, 128, 'f16', (version, x) => () => version.lib.topk(x, 4)),

  logitStats('logitStats f16, Echo coarse pass [335,2052], 2 × 1025', 335, 2, false),
  logitStats('logitStats f16 + guide, Echo fine pass [335,6152], 6 × 1025', 335, 6, true),

  rmsNorm('rmsNorm, privacy-filter [512,640]', 512, 640, 'f32'),
  rmsNorm('rmsNorm f16, privacy-filter [512,640]', 512, 640, 'f16'),
  // scalar: cols % 4 != 0
  rmsNorm('rmsNorm scalar, odd width [512,642]', 512, 642, 'f32'),
  rmsNorm('rmsNorm scalar f16, odd width [512,642]', 512, 642, 'f16'),
  noSubgroups(rmsNorm('rmsNorm, privacy-filter [512,640]', 512, 640, 'f32')),
  noSubgroups(rmsNorm('rmsNorm scalar, odd width [512,642]', 512, 642, 'f32')),

  rows('groupNorm 1 group, Moonshine stem [500,288]', 500, 288, 'f32', (version, x) => {
    const w = randMat(version, 1, 288);
    const b = randMat(version, 1, 288);
    return () => version.lib.groupNorm(x, 1, w, b);
  }),
]);
