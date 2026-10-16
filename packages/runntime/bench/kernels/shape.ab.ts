/** The copy kernels behind transpose, cat, slice, astype and the gathers.
 *  Inputs are uploads, so no copy is elided into a producer. */
import { abSuite, type AbCase, type Version } from '../ab.ts';
import type { Value } from '../../src/core/index.ts';
import { chw, ids, mat, randHwc4, randMat } from './inputs.ts';

function on(
  name: string,
  make: (version: Version) => Value[],
  op: (version: Version, xs: Value[]) => Value,
  f16 = false,
): AbCase {
  return {
    name,
    needs: f16 ? ['shader-f16'] : [],
    build: (version) => {
      const xs = make(version);
      return () => op(version, xs);
    },
  };
}

abSuite('shape', [
  on(
    'transpose [1500,512]',
    (version) => [randMat(version, 1500, 512)],
    (version, [x]) => version.lib.transpose(x!),
  ),
  on(
    'transpose f16 [384,1536]',
    (version) => [randMat(version, 384, 1536, 'f16')],
    (version, [x]) => version.lib.transpose(x!),
    true,
  ),

  on(
    'cat rows [200,288] + [200,288]',
    (version) => [randMat(version, 200, 288), randMat(version, 200, 288)],
    (version, xs) => version.lib.cat(xs, 0),
  ),
  on(
    'cat cols [512,320] + [512,320]',
    (version) => [randMat(version, 512, 320), randMat(version, 512, 320)],
    (version, xs) => version.lib.cat(xs, 1),
  ),
  on(
    'cat channels CHW [48,56,56] + [48,56,56]',
    (version) => [
      version.upload(version.rand(48 * 56 * 56), chw(48, 56, 56)),
      version.upload(version.rand(48 * 56 * 56), chw(48, 56, 56)),
    ],
    (version, xs) => version.lib.cat(xs, 0),
  ),
  on(
    'cat channels hwc4 [64,40,40] + [128,40,40]',
    (version) => [randHwc4(version, 64, 40, 40), randHwc4(version, 128, 40, 40)],
    (version, xs) => version.lib.cat(xs, 0),
    true,
  ),

  on(
    'slice rows [512,640] → 128..384',
    (version) => [randMat(version, 512, 640)],
    (version, [x]) => version.lib.slice(x!, 0, 128, 384),
  ),
  on(
    'slice cols [512,1152] → 384..768',
    (version) => [randMat(version, 512, 1152)],
    (version, [x]) => version.lib.slice(x!, 1, 384, 768),
  ),
  on(
    'slice channels CHW [96,28,28] → 32..64',
    (version) => [version.upload(version.rand(96 * 28 * 28), chw(96, 28, 28))],
    (version, [x]) => version.lib.slice(x!, 0, 32, 64),
  ),
  on(
    'slice channels hwc4 [128,40,40] → 64..128',
    (version) => [randHwc4(version, 128, 40, 40)],
    (version, [x]) => version.lib.slice(x!, 0, 64, 128),
    true,
  ),

  on(
    'astype f32 → f16 [512,640]',
    (version) => [randMat(version, 512, 640)],
    (version, [x]) => version.lib.astype(x!, 'f16'),
    true,
  ),
  on(
    'astype f16 → f32 [512,640]',
    (version) => [randMat(version, 512, 640, 'f16')],
    (version, [x]) => version.lib.astype(x!, 'f32'),
    true,
  ),

  on(
    'gatherRows, 64 of [4096,384]',
    (version) => [randMat(version, 4096, 384)],
    (version, [x]) => version.lib.gatherRows(x!, Array.from(ids(64, 4096))),
  ),
  on(
    'gatherRowsFrom, Moonshine-tiny embedding [32768,288], 200 ids',
    (version) => [randMat(version, 32768, 288), version.upload(ids(200, 32768), mat(200, 1))],
    (version, [x, idx]) => version.lib.gatherRowsFrom(x!, idx!),
  ),
  on(
    'gatherRowsFrom f16, one id from [32768,640]',
    (version) => [randMat(version, 32768, 640, 'f16'), version.upload(ids(1, 32768), mat(1, 1))],
    (version, [x, idx]) => version.lib.gatherRowsFrom(x!, idx!),
    true,
  ),
]);
