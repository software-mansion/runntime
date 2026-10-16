/** The elementwise kernels: add, sub and mul in all four broadcast modes,
 *  the scalar and unary kernels, swiglu, and hwc4 inputs. */
import { abSuite, type AbCase, type Version } from '../ab.ts';
import type { Value } from '../../src/core/index.ts';
import { mat, randHwc4, randMat } from './inputs.ts';

type Float = 'f32' | 'f16';

function binary(
  name: string,
  op: 'add' | 'sub' | 'mul',
  m: number,
  n: number,
  b: [number, number],
  dtype: Float = 'f32',
): AbCase {
  return {
    name,
    needs: dtype === 'f16' ? ['shader-f16'] : [],
    build: (version) => {
      const x = randMat(version, m, n, dtype);
      const y = randMat(version, b[0], b[1], dtype);
      return () => version.lib[op](x, y);
    },
  };
}

function unary(
  name: string,
  m: number,
  n: number,
  dtype: Float,
  op: (version: Version, x: Value) => Value,
): AbCase {
  return {
    name,
    needs: dtype === 'f16' ? ['shader-f16'] : [],
    build: (version) => {
      const x = randMat(version, m, n, dtype);
      return () => op(version, x);
    },
  };
}

function hwc4(name: string, op: (version: Version, x: Value) => Value): AbCase {
  return {
    name,
    needs: ['shader-f16'],
    build: (version) => {
      const x = randHwc4(version, 64, 80, 80);
      return () => op(version, x);
    },
  };
}

/** Past 65535 × 64 elements the dispatch splits into a 2D grid. */
const BIG = 2 ** 23;

abSuite('elementwise', [
  binary('add, same shape, privacy-filter residual [512,640]', 'add', 512, 640, [512, 640]),
  binary('add, row bias [512,640] + [1,640]', 'add', 512, 640, [1, 640]),
  binary('add, column [512,640] + [512,1]', 'add', 512, 640, [512, 1]),
  binary('add, single value [512,640] + [1,1]', 'add', 512, 640, [1, 1]),
  binary('add f16, row bias [512,640] + [1,640]', 'add', 512, 640, [1, 640], 'f16'),
  binary('sub, column [512,640] − [512,1]', 'sub', 512, 640, [512, 1]),
  binary('mul, row [512,640] · [1,640]', 'mul', 512, 640, [1, 640]),
  binary('mul f16, same shape [512,640]', 'mul', 512, 640, [512, 640], 'f16'),
  binary('add, 2D grid, 2^23 elements', 'add', 1, BIG, [1, BIG]),

  unary('addScalar [512,640]', 512, 640, 'f32', (version, x) => version.lib.add(x, 0.5)),
  unary('mulScalar f16 [512,640]', 512, 640, 'f16', (version, x) => version.lib.mul(x, 0.5)),
  unary('clamp [512,640]', 512, 640, 'f32', (version, x) => version.lib.clamp(x, -0.5, 0.5)),
  unary('gelu, MiniLM fc1 [128,1536]', 128, 1536, 'f32', (version, x) => version.lib.gelu(x)),
  unary('gelu f16 [128,1536]', 128, 1536, 'f16', (version, x) => version.lib.gelu(x)),
  unary('silu, streaming [500,320]', 500, 320, 'f32', (version, x) => version.lib.silu(x)),
  unary('sigmoid [512,640]', 512, 640, 'f32', (version, x) => version.lib.sigmoid(x)),
  unary('tanh [512,640]', 512, 640, 'f32', (version, x) => version.lib.tanh(x)),
  unary('asinh, streaming frontend [500,80]', 500, 80, 'f32', (version, x) => version.lib.asinh(x)),
  {
    name: 'rsqrt [512,640]',
    build: (version) => {
      const x = version.upload(
        version.rand(512 * 640).map((v) => Math.abs(v) + 0.5),
        mat(512, 640),
      );
      return () => version.lib.rsqrt(x);
    },
  },
  unary('silu, 2D grid, 2^23 elements', 1, BIG, 'f32', (version, x) => version.lib.silu(x)),

  {
    name: 'swiglu, privacy-filter [512,640]',
    build: (version) => {
      const glu = randMat(version, 512, 640);
      const lin = randMat(version, 512, 640);
      return () => version.lib.swiglu(glu, lin);
    },
  },
  {
    name: 'swiglu f16 [512,640]',
    needs: ['shader-f16'],
    build: (version) => {
      const glu = randMat(version, 512, 640, 'f16');
      const lin = randMat(version, 512, 640, 'f16');
      return () => version.lib.swiglu(glu, lin);
    },
  },
  unary('swigluChunk, Moonshine-tiny fc1 [1,2304]', 1, 2304, 'f32', (version, x) =>
    version.lib.swigluChunk(x),
  ),
  unary('swigluChunk f16, prefill [200,2304]', 200, 2304, 'f16', (version, x) =>
    version.lib.swigluChunk(x),
  ),

  hwc4('silu on hwc4, YOLO26n [64,80,80]', (version, x) => version.lib.silu(x)),
  hwc4('mulScalar on hwc4 [64,80,80]', (version, x) => version.lib.mul(x, 0.5)),
  {
    name: 'add hwc4 + hwc4, YOLO26n residual [64,80,80]',
    needs: ['shader-f16'],
    build: (version) => {
      const x = randHwc4(version, 64, 80, 80);
      const y = randHwc4(version, 64, 80, 80);
      return () => version.lib.add(x, y);
    },
  },
]);
