/** The copy kernels behind transpose, cat, slice, astype and the gathers.
 *  Inputs are uploads, so no copy is elided into a producer. */
import { abSuite, type AbCase, type Side } from '../ab.ts';
import type { Value } from '../../src/core/index.ts';
import { chw, ids, mat, randHwc4, randMat } from './inputs.ts';

function on(
  name: string,
  make: (s: Side) => Value[],
  op: (s: Side, xs: Value[]) => Value,
  f16 = false,
): AbCase {
  return {
    name,
    needs: f16 ? ['shader-f16'] : [],
    build: (s) => {
      const xs = make(s);
      return () => op(s, xs);
    },
  };
}

abSuite('shape', [
  on(
    'transpose [1500,512]',
    (s) => [randMat(s, 1500, 512)],
    (s, [x]) => s.lib.transpose(x!),
  ),
  on(
    'transpose f16 [384,1536]',
    (s) => [randMat(s, 384, 1536, 'f16')],
    (s, [x]) => s.lib.transpose(x!),
    true,
  ),

  on(
    'cat rows [200,288] + [200,288]',
    (s) => [randMat(s, 200, 288), randMat(s, 200, 288)],
    (s, xs) => s.lib.cat(xs, 0),
  ),
  on(
    'cat cols [512,320] + [512,320]',
    (s) => [randMat(s, 512, 320), randMat(s, 512, 320)],
    (s, xs) => s.lib.cat(xs, 1),
  ),
  on(
    'cat channels CHW [48,56,56] + [48,56,56]',
    (s) => [
      s.up(s.rand(48 * 56 * 56), chw(48, 56, 56)),
      s.up(s.rand(48 * 56 * 56), chw(48, 56, 56)),
    ],
    (s, xs) => s.lib.cat(xs, 0),
  ),
  on(
    'cat channels hwc4 [64,40,40] + [128,40,40]',
    (s) => [randHwc4(s, 64, 40, 40), randHwc4(s, 128, 40, 40)],
    (s, xs) => s.lib.cat(xs, 0),
    true,
  ),

  on(
    'slice rows [512,640] → 128..384',
    (s) => [randMat(s, 512, 640)],
    (s, [x]) => s.lib.slice(x!, 0, 128, 384),
  ),
  on(
    'slice cols [512,1152] → 384..768',
    (s) => [randMat(s, 512, 1152)],
    (s, [x]) => s.lib.slice(x!, 1, 384, 768),
  ),
  on(
    'slice channels CHW [96,28,28] → 32..64',
    (s) => [s.up(s.rand(96 * 28 * 28), chw(96, 28, 28))],
    (s, [x]) => s.lib.slice(x!, 0, 32, 64),
  ),
  on(
    'slice channels hwc4 [128,40,40] → 64..128',
    (s) => [randHwc4(s, 128, 40, 40)],
    (s, [x]) => s.lib.slice(x!, 0, 64, 128),
    true,
  ),

  on(
    'astype f32 → f16 [512,640]',
    (s) => [randMat(s, 512, 640)],
    (s, [x]) => s.lib.astype(x!, 'f16'),
    true,
  ),
  on(
    'astype f16 → f32 [512,640]',
    (s) => [randMat(s, 512, 640, 'f16')],
    (s, [x]) => s.lib.astype(x!, 'f32'),
    true,
  ),

  on(
    'gatherRows, 64 of [4096,384]',
    (s) => [randMat(s, 4096, 384)],
    (s, [x]) => s.lib.gatherRows(x!, Array.from(ids(64, 4096))),
  ),
  on(
    'gatherRowsFrom, Moonshine-tiny embedding [32768,288], 200 ids',
    (s) => [randMat(s, 32768, 288), s.up(ids(200, 32768), mat(200, 1))],
    (s, [x, idx]) => s.lib.gatherRowsFrom(x!, idx!),
  ),
  on(
    'gatherRowsFrom f16, one id from [32768,640]',
    (s) => [randMat(s, 32768, 640, 'f16'), s.up(ids(1, 32768), mat(1, 1))],
    (s, [x, idx]) => s.lib.gatherRowsFrom(x!, idx!),
    true,
  ),
]);
