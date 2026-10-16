/** Every conv2d kind gpu/dispatch/conv.ts can pick (1×1, split-K 1×1, tiled,
 *  generic, depthwise), the fused epilogues, conv1d, convTranspose2d and the
 *  other hwc4 image kernels. Shapes are YOLO26n at 640, MobileNetV4-S at 224,
 *  DepthART and Moonshine. */
import { abSuite, type AbCase, type Version } from '../ab.ts';
import type { Value } from '../../src/core/index.ts';
import { chw, convWeight, randHwc4, randMat } from './inputs.ts';

interface Conv {
  cIn: number;
  cOut: number;
  hw: number;
  k: number;
  stride?: number;
  depthwise?: boolean;
  act?: 'silu' | 'gelu' | 'relu';
  /** Adds a residual of the output's shape, which fuses into the conv. */
  residual?: boolean;
}

function conv2d(name: string, o: Conv): AbCase {
  const { cIn, cOut, hw, k, stride = 1 } = o;
  const groups = o.depthwise ? cIn : 1;
  const padding = (k - 1) >> 1;
  return {
    name,
    needs: ['shader-f16'],
    build: (version) => {
      const x = randHwc4(version, cIn, hw, hw);
      const w = convWeight(version, cOut, cIn, k, groups);
      const bias = version.upload(version.rand(cOut), { elems: cOut, dtype: 'f16', dims: [cOut] });
      const hOut = Math.floor((hw + 2 * padding - k) / stride) + 1;
      const res = o.residual ? randHwc4(version, cOut, hOut, hOut) : undefined;
      return () => {
        const y = version.lib.conv2d(x, w, bias, {
          kernelSize: k,
          stride,
          padding,
          groups,
          activation: o.act,
        });
        return res ? version.lib.add(res, y) : y;
      };
    },
  };
}

function image(
  name: string,
  c: number,
  hw: number,
  op: (version: Version, x: Value) => Value,
): AbCase {
  return {
    name,
    needs: ['shader-f16'],
    build: (version) => {
      const x = randHwc4(version, c, hw, hw);
      return () => op(version, x);
    },
  };
}

function conv1d(
  name: string,
  tIn: number,
  cIn: number,
  cOut: number,
  k: number,
  stride: number,
  act?: 'tanh' | 'gelu',
): AbCase {
  return {
    name,
    build: (version) => {
      const x = randMat(version, tIn, cIn);
      const w = version.upload(
        version.lib.convToTapMajor(version.rand(cOut * cIn * k), cOut, cIn, k),
        {
          elems: k * cIn * cOut,
          dtype: 'f32',
          dims: [k * cIn, cOut],
        },
      );
      return () => version.lib.conv1d(x, w, { kernelSize: k, stride, activation: act });
    },
  };
}

function convTranspose2d(name: string, cIn: number, cOut: number, hw: number): AbCase {
  const k = 2;
  return {
    name,
    needs: ['shader-f16'],
    build: (version) => {
      const x = version.upload(version.rand(cIn * hw * hw), chw(cIn, hw, hw, 'f16'));
      const bits = version.lib.f32ArrayToF16Bits(version.rand(cIn * cOut * k * k)).data;
      const w = version.freeAfterCase(
        version.lib.materialized(
          { elems: cOut * k * k * cIn, dtype: 'f16', dims: [cOut * k * k, cIn] },
          version.lib.uploadF16(
            version.root,
            version.lib.permuteConvTransposeF16(bits, cIn, cOut, k),
          ),
        ),
      );
      const bias = version.upload(version.rand(cOut), { elems: cOut, dtype: 'f16', dims: [cOut] });
      return () => version.lib.convTranspose2d(x, w, bias, { kernelSize: k, stride: k });
    },
  };
}

abSuite('conv', [
  // kind 1: 1×1, hw % 4 = 0 and enough tiles. 8 pixels per thread when hw % 8 = 0.
  conv2d('1×1 (8 px/thread), YOLO26n SplitFuse 32→32 @160²', {
    cIn: 32,
    cOut: 32,
    hw: 160,
    k: 1,
    act: 'silu',
  }),
  conv2d('1×1 (4 px/thread), 64→64 @38²', { cIn: 64, cOut: 64, hw: 38, k: 1 }),
  // kind 3: split-K 1×1, small maps.
  conv2d('1×1 split-K, MobileNetV4 conv_head 960→1280 @1²', {
    cIn: 960,
    cOut: 1280,
    hw: 1,
    k: 1,
    act: 'relu',
  }),
  conv2d('1×1 split-K, YOLO26n cls 80→80 @20²', { cIn: 80, cOut: 80, hw: 20, k: 1 }),
  // kind 4: tiled, 8 or 16 output channels per block.
  conv2d('tiled (8 ch/block), YOLO26n layer1 16→32 k3 s2 @320²', {
    cIn: 16,
    cOut: 32,
    hw: 320,
    k: 3,
    stride: 2,
    act: 'silu',
  }),
  conv2d('tiled (16 ch/block), YOLO26n layer3 64→64 k3 s2 @160²', {
    cIn: 64,
    cOut: 64,
    hw: 160,
    k: 3,
    stride: 2,
    act: 'silu',
  }),
  conv2d('tiled (16 ch/block), YOLO26n layer7 128→256 k3 s2 @40²', {
    cIn: 128,
    cOut: 256,
    hw: 40,
    k: 3,
    stride: 2,
  }),
  // 1×1 with stride 2 can't use the 1×1 kernels, so it lands here.
  conv2d('tiled (16 ch/block), 1×1 stride 2, 64→128 @80²', {
    cIn: 64,
    cOut: 128,
    hw: 80,
    k: 1,
    stride: 2,
  }),
  // kind 0: generic.
  conv2d('generic, YOLO26n layer0 3→16 k3 s2 @640²', {
    cIn: 3,
    cOut: 16,
    hw: 640,
    k: 3,
    stride: 2,
    act: 'silu',
  }),
  conv2d('generic, YOLO26n box head 64→16 k3 @80²', { cIn: 64, cOut: 16, hw: 80, k: 3 }),
  // kind 2: depthwise, 2 columns per thread on wide maps or k ≥ 7.
  conv2d('depthwise (2 cols), YOLO26n head 64ch k3 @80²', {
    cIn: 64,
    cOut: 64,
    hw: 80,
    k: 3,
    depthwise: true,
    act: 'silu',
  }),
  conv2d('depthwise (1 col), MobileNetV4 576ch k5 @14²', {
    cIn: 576,
    cOut: 576,
    hw: 14,
    k: 5,
    depthwise: true,
  }),
  conv2d('depthwise (2 cols), k7 96ch @28²', { cIn: 96, cOut: 96, hw: 28, k: 7, depthwise: true }),
  // Fused residual add (hasAdd).
  conv2d('tiled + fused residual, 64→64 k3 @80²', {
    cIn: 64,
    cOut: 64,
    hw: 80,
    k: 3,
    residual: true,
  }),
  conv2d('1×1 + fused residual, 64→64 @80²', { cIn: 64, cOut: 64, hw: 80, k: 1, residual: true }),
  {
    name: 'two convs into a cat (copy elided), YOLO26n neck 64+64 @40²',
    needs: ['shader-f16'],
    build: (version) => {
      const x = randHwc4(version, 64, 40, 40);
      const w1 = convWeight(version, 64, 64, 1);
      const w2 = convWeight(version, 64, 64, 3);
      const opts = { stride: 1, groups: 1 };
      return () =>
        version.lib.cat(
          [
            version.lib.conv2d(x, w1, undefined, { ...opts, kernelSize: 1, padding: 0 }),
            version.lib.conv2d(x, w2, undefined, { ...opts, kernelSize: 3, padding: 1 }),
          ],
          0,
        );
    },
  },

  conv1d('conv1d + tanh, Moonshine-tiny stem 1→288 k127 s64', 16000, 1, 288, 127, 64, 'tanh'),
  conv1d('conv1d + gelu, Moonshine-tiny conv2 288→576 k7 s3', 249, 288, 576, 7, 3, 'gelu'),
  conv1d('conv1d, cOut not a multiple of 4, 32→6 k3', 500, 32, 6, 3, 1),

  convTranspose2d('convTranspose2d (paired), YOLO26n-seg proto 64→64 @80²', 64, 64, 80),
  convTranspose2d('convTranspose2d (unpaired), cIn 3 @64²', 3, 8, 64),

  image('maxPool2d k5, YOLO26n SPPF 128ch @20²', 128, 20, (version, x) =>
    version.lib.maxPool2d(x, { kernelSize: 5, stride: 1, padding: 2 }),
  ),
  image('avgPool2d k8, DepthART stage0 12ch @112²', 12, 112, (version, x) =>
    version.lib.avgPool2d(x, { kernelSize: 8 }),
  ),
  image('upsample2d ×2, YOLO26n neck 256ch @20²', 256, 20, (version, x) =>
    version.lib.upsample2d(x, { scale: 2 }),
  ),
  image('resizeBilinear2d alignCorners, DepthART head 48ch @112²→224²', 48, 112, (version, x) =>
    version.lib.resizeBilinear2d(x, { outH: 224, outW: 224 }),
  ),
  image('resizeBilinear2d halfPixel, 3ch @112²→448²', 3, 112, (version, x) =>
    version.lib.resizeBilinear2d(x, { outH: 448, outW: 448, mode: 'halfPixel' }),
  ),
  image('toChw f32, MobileNetV4 960ch @7²', 960, 7, (version, x) => version.lib.toChw(x, 'f32')),
  image('toChw f16, YOLO26n PSA 128ch @20²', 128, 20, (version, x) => version.lib.toChw(x, 'f16')),
  image('pad2d, 64ch @80²', 64, 80, (version, x) => version.lib.pad2d(x, { padH: 1, padW: 1 })),
  {
    name: 'channelAffine + shift, 96ch @56²',
    needs: ['shader-f16'],
    build: (version) => {
      const x = randHwc4(version, 96, 56, 56);
      const scale = randMat(version, 1, 96);
      const shift = randMat(version, 1, 96);
      return () => version.lib.channelAffine(x, scale, shift);
    },
  },
  {
    name: 'toHwc4 from f32, YOLO26n input 3ch @640²',
    needs: ['shader-f16'],
    build: (version) => {
      const x = version.upload(version.rand(3 * 640 * 640), chw(3, 640, 640));
      return () => version.lib.toHwc4(x);
    },
  },
  {
    name: 'toHwc4 from f16, 96ch @56²',
    needs: ['shader-f16'],
    build: (version) => {
      const x = version.upload(version.rand(96 * 56 * 56), chw(96, 56, 56, 'f16'));
      return () => version.lib.toHwc4(x);
    },
  },
]);
