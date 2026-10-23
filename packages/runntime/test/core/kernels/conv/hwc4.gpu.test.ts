import { describe, expect, it, onTestFinished, type TestContext } from 'vitest';
import { toArray } from '../../../../src/core/gpu/eval.ts';
import {
  add,
  avgPool2d,
  cat,
  channelAffine,
  type ConvAct,
  conv2d,
  maxPool2d,
  pad2d,
  resizeBilinear2d,
  slice,
  softmax,
  toChw,
  toHwc4,
  upsample2d,
} from '../../../../src/core/graph/ops.ts';
import { c4of, hwc4Meta, tensor3d, type Value } from '../../../../src/core/graph/value.ts';
import { Conv2d } from '../../../../src/core/nn/layers.ts';
import { memoryStateDict } from '../../../../src/core/weights/safetensors.ts';
import { expectClose, gpu, randomF32, requireFeature, roundF16, upload } from '../../../gpu.ts';
import { ACTIVATION_REF } from '../matmul/checkMatmul.ts';

const { root, ex } = await gpu();

/** Every hwc4 kernel stores f16. */
const needF16 = (ctx: TestContext) => requireFeature(ctx, root, 'shader-f16');

/** Random [C, H, W] values, rounded to the f16 that toHwc4 stores. */
const image = (c: number, h: number, w: number, seed: number) =>
  roundF16(randomF32(c * h * w, seed));

/** CHW f32 data into hwc4 storage, through toHwc4. */
const hwc4 = (x: Float32Array, c: number, h: number, w: number) =>
  toHwc4(upload(root, x, tensor3d(c, h, w)));

/** Back to CHW f32, the padding lanes dropped. */
const read = (y: Value) => toArray(toChw(y, 'f32'), ex);

const EXACT = { atol: 0, rtol: 0 };

interface Conv {
  cIn: number;
  cOut: number;
  h: number;
  w: number;
  kH: number;
  kW: number;
  stride: number;
  padding: number;
  /** 1, or cIn for depthwise. */
  groups: number;
  bias?: boolean;
  activation?: ConvAct;
  /** add(residual, conv), which the evaluator folds into the conv's epilogue. */
  residual?: boolean;
  winograd?: boolean;
}

const outSize = (c: Conv) => ({
  hOut: Math.floor((c.h + 2 * c.padding - c.kH) / c.stride) + 1,
  wOut: Math.floor((c.w + 2 * c.padding - c.kW) / c.stride) + 1,
});

/** torch's conv2d over x [C_in, H, W] with weight [C_out, C_in / groups, kH,
 *  kW], then act(conv + bias) + residual, in f64. */
function conv2dRef(
  x: Float32Array,
  wt: Float32Array,
  c: Conv,
  bias?: Float32Array,
  residual?: Float32Array,
) {
  const { hOut, wOut } = outSize(c);
  const cInG = c.cIn / c.groups;
  const out = new Float64Array(c.cOut * hOut * wOut);
  for (let co = 0; co < c.cOut; co++) {
    const ci0 = c.groups === 1 ? 0 : co;
    for (let oy = 0; oy < hOut; oy++) {
      for (let ox = 0; ox < wOut; ox++) {
        let sum = bias ? bias[co]! : 0;
        for (let g = 0; g < cInG; g++) {
          for (let ky = 0; ky < c.kH; ky++) {
            const iy = oy * c.stride + ky - c.padding;
            if (iy < 0 || iy >= c.h) continue;
            for (let kx = 0; kx < c.kW; kx++) {
              const ix = ox * c.stride + kx - c.padding;
              if (ix < 0 || ix >= c.w) continue;
              const wv = wt[((co * cInG + g) * c.kH + ky) * c.kW + kx]!;
              sum += wv * x[((ci0 + g) * c.h + iy) * c.w + ix]!;
            }
          }
        }
        if (c.activation) sum = ACTIVATION_REF[c.activation](sum);
        const i = (co * hOut + oy) * wOut + ox;
        out[i] = sum + (residual ? residual[i]! : 0);
      }
    }
  }
  return out;
}

/** An nn.Conv2d, which packs the weight for the route it takes, with random
 *  f16 weights scaled by 1/sqrt(K) so every output stays near 1, where f16
 *  sums are good to ~1e-3. */
async function makeConv2d(c: Conv, seed: number) {
  const k = (c.cIn / c.groups) * c.kH * c.kW;
  const wt = roundF16(randomF32(c.cOut * k, seed).map((v) => v / Math.sqrt(k)));
  const bias = c.bias ? roundF16(randomF32(c.cOut, seed + 1)) : undefined;
  const conv = new Conv2d(c.cIn, c.cOut, {
    kernelSize: [c.kH, c.kW],
    stride: c.stride,
    padding: c.padding,
    groups: c.groups,
    bias: c.bias ?? false,
    activation: c.activation,
    winograd: c.winograd,
  });
  await conv.loadStateDict(
    memoryStateDict({
      weight: { data: wt, shape: [c.cOut, c.cIn / c.groups, c.kH, c.kW] },
      ...(bias ? { bias } : {}),
    }),
  );
  onTestFinished(() => conv.dispose());
  return { conv, wt, bias };
}

/** Every route but depthwise sums in f16, which drifts with the square root of
 *  K. The Winograd transforms scale values by up to 100 before the f16 planes. */
function convTol(c: Conv) {
  if (c.winograd) return { atol: 3e-2, rtol: 2e-2 };
  return { atol: 2e-3 + 1e-3 * Math.sqrt((c.cIn / c.groups) * c.kH * c.kW), rtol: 1e-2 };
}

async function checkConv2d(c: Conv): Promise<void> {
  const x = image(c.cIn, c.h, c.w, 1);
  const { conv, wt, bias } = await makeConv2d(c, 2);
  const { hOut, wOut } = outSize(c);
  const residual = c.residual ? image(c.cOut, hOut, wOut, 4) : undefined;
  const y = conv.forward(hwc4(x, c.cIn, c.h, c.w));
  const got = await read(residual ? add(hwc4(residual, c.cOut, hOut, wOut), y) : y);
  expectClose(got, conv2dRef(x, wt, c, bias, residual), convTol(c));
}

/** max over the taps inside x, as torch's max_pool2d pads with -inf. */
function maxPoolRef(
  x: Float32Array,
  c: number,
  h: number,
  w: number,
  k: number,
  stride: number,
  padding: number,
) {
  const hOut = Math.floor((h + 2 * padding - k) / stride) + 1;
  const wOut = Math.floor((w + 2 * padding - k) / stride) + 1;
  const out = new Float64Array(c * hOut * wOut);
  for (let ch = 0; ch < c; ch++) {
    for (let oy = 0; oy < hOut; oy++) {
      for (let ox = 0; ox < wOut; ox++) {
        let best = -Infinity;
        for (let ky = 0; ky < k; ky++) {
          for (let kx = 0; kx < k; kx++) {
            const iy = oy * stride + ky - padding;
            const ix = ox * stride + kx - padding;
            if (iy >= 0 && iy < h && ix >= 0 && ix < w)
              best = Math.max(best, x[(ch * h + iy) * w + ix]!);
          }
        }
        out[(ch * hOut + oy) * wOut + ox] = best;
      }
    }
  }
  return out;
}

describe('hwc4 kernels', () => {
  describe('conv2d', () => {
    // Routes are picked in gpu/dispatch/conv.ts: winograd when opted in on the
    // module; a 1×1, stride 1, padding 0, groups 1 conv takes 1x1 or 1x1s by
    // conv1x1Kind; any other groups-1 conv takes tiled when convTiledBlocks
    // says so and direct otherwise; groups = C_in takes depthwise.
    const ROUTES: { name: string; c: Conv }[] = [
      // direct: CO4 <= 8 orders threads block-first; columns per thread are
      // the most of 4, 2, 1 that keep 16384 threads.
      {
        name: '3×3, odd channels (direct, 1 column)',
        c: { cIn: 3, cOut: 5, h: 8, w: 7, kH: 3, kW: 3, stride: 1, padding: 1, groups: 1 },
      },
      {
        name: 'stride 2, no padding (direct, 1 column)',
        c: { cIn: 4, cOut: 8, h: 9, w: 9, kH: 3, kW: 3, stride: 2, padding: 0, groups: 1 },
      },
      {
        name: '3×1 kernel (direct, 1 column)',
        c: { cIn: 6, cOut: 4, h: 5, w: 6, kH: 3, kW: 1, stride: 1, padding: 1, groups: 1 },
      },
      {
        name: '1×3 kernel (direct, 1 column)',
        c: { cIn: 12, cOut: 8, h: 9, w: 11, kH: 1, kW: 3, stride: 1, padding: 0, groups: 1 },
      },
      // direct: a 1×1 with stride 2 is not a 1x1 route
      {
        name: '1×1 stride 2 (direct, 1 column)',
        c: { cIn: 4, cOut: 4, h: 6, w: 6, kH: 1, kW: 1, stride: 2, padding: 0, groups: 1 },
      },
      {
        name: '1×1 out pixel (direct, 1 column)',
        c: { cIn: 2, cOut: 3, h: 3, w: 3, kH: 3, kW: 3, stride: 1, padding: 0, groups: 1 },
      },
      {
        name: 'narrow, stride 2 (direct, 1 column)',
        c: { cIn: 8, cOut: 16, h: 32, w: 32, kH: 3, kW: 3, stride: 2, padding: 1, groups: 1 },
      },
      {
        name: 'narrow (direct, 4 columns)',
        c: {
          cIn: 8,
          cOut: 16,
          h: 128,
          w: 128,
          kH: 3,
          kW: 3,
          stride: 1,
          padding: 1,
          groups: 1,
          bias: true,
          activation: 'relu',
        },
      },
      {
        name: 'wide (direct, 2 columns)',
        c: { cIn: 4, cOut: 40, h: 64, w: 64, kH: 3, kW: 3, stride: 1, padding: 1, groups: 1 },
      },
      {
        name: 'wide, stride 2 (direct, 1 column)',
        c: { cIn: 32, cOut: 64, h: 28, w: 28, kH: 3, kW: 3, stride: 2, padding: 1, groups: 1 },
      },
      // tiled: >= 24 workgroups of 64 pixels, and C_out >= 64, or C_out >= 32
      // on >= 6400 pixels, or C_out >= 24 on >= 3136 pixels with K >= 64
      {
        name: 'C_out 64 (tiled, 16 blocks)',
        c: {
          cIn: 16,
          cOut: 64,
          h: 48,
          w: 48,
          kH: 3,
          kW: 3,
          stride: 1,
          padding: 1,
          groups: 1,
          bias: true,
        },
      },
      // tiled: C_out 66 leaves a second, partial row of channel blocks
      {
        name: 'C_out 66 (tiled, 16 blocks)',
        c: { cIn: 5, cOut: 66, h: 41, w: 39, kH: 3, kW: 3, stride: 1, padding: 1, groups: 1 },
      },
      {
        name: 'C_out 32 on 80×80 (tiled, 8 blocks)',
        c: { cIn: 8, cOut: 32, h: 80, w: 80, kH: 3, kW: 3, stride: 1, padding: 1, groups: 1 },
      },
      {
        name: 'C_out 24, deep K (tiled, 8 blocks)',
        c: { cIn: 8, cOut: 24, h: 112, w: 112, kH: 3, kW: 3, stride: 2, padding: 1, groups: 1 },
      },
      // 1x1: H·W % 4 == 0 and (H·W / 4)·CO4 >= 2048. H·W % 8 == 0 runs 8
      // pixels per thread, else 4. Under 8192 threads K unrolls by the most of
      // 4, 2 dividing CI4.
      {
        name: '8 pixels per thread, K unrolled by 4 (1x1)',
        c: { cIn: 32, cOut: 32, h: 32, w: 32, kH: 1, kW: 1, stride: 1, padding: 0, groups: 1 },
      },
      {
        name: '4 pixels per thread, K unrolled by 2 (1x1)',
        c: { cIn: 24, cOut: 40, h: 30, w: 30, kH: 1, kW: 1, stride: 1, padding: 0, groups: 1 },
      },
      {
        name: '8192 threads: K not unrolled, odd CI4 (1x1)',
        c: { cIn: 20, cOut: 64, h: 64, w: 64, kH: 1, kW: 1, stride: 1, padding: 0, groups: 1 },
      },
      {
        name: 'padded channels (1x1)',
        c: {
          cIn: 6,
          cOut: 10,
          h: 56,
          w: 56,
          kH: 1,
          kW: 1,
          stride: 1,
          padding: 0,
          groups: 1,
          bias: true,
          activation: 'silu',
        },
      },
      // 1x1s: few threads, or H·W % 4 != 0
      {
        name: 'small map (1x1s, 2 lanes)',
        c: { cIn: 4, cOut: 8, h: 4, w: 4, kH: 1, kW: 1, stride: 1, padding: 0, groups: 1 },
      },
      {
        name: 'H·W % 4 != 0, odd channels (1x1s)',
        c: { cIn: 7, cOut: 5, h: 3, w: 5, kH: 1, kW: 1, stride: 1, padding: 0, groups: 1 },
      },
      {
        name: '7×7 map (1x1s, 2 lanes)',
        c: {
          cIn: 64,
          cOut: 48,
          h: 7,
          w: 7,
          kH: 1,
          kW: 1,
          stride: 1,
          padding: 0,
          groups: 1,
          bias: true,
        },
      },
      {
        name: 'C_in 256 (1x1s, 8 lanes)',
        c: { cIn: 256, cOut: 16, h: 5, w: 5, kH: 1, kW: 1, stride: 1, padding: 0, groups: 1 },
      },
      {
        name: 'C_in 512 (1x1s, 16 lanes)',
        c: { cIn: 512, cOut: 8, h: 3, w: 3, kH: 1, kW: 1, stride: 1, padding: 0, groups: 1 },
      },
      // depthwise: one column per thread on maps up to 14 wide with kW < 7
      {
        name: '3×3 on a narrow map (depthwise, 1 column)',
        c: { cIn: 8, cOut: 8, h: 10, w: 10, kH: 3, kW: 3, stride: 1, padding: 1, groups: 8 },
      },
      // depthwise: two columns past that, the last run partial when W_out is odd
      {
        name: '3×3 on 21 columns (depthwise, 2 columns)',
        c: { cIn: 6, cOut: 6, h: 9, w: 21, kH: 3, kW: 3, stride: 1, padding: 1, groups: 6 },
      },
      {
        name: '7×7 on a narrow map (depthwise, 2 columns)',
        c: { cIn: 4, cOut: 4, h: 10, w: 10, kH: 7, kW: 7, stride: 1, padding: 3, groups: 4 },
      },
      {
        name: 'stride 2 (depthwise, 2 columns)',
        c: { cIn: 12, cOut: 12, h: 31, w: 31, kH: 5, kW: 5, stride: 2, padding: 2, groups: 12 },
      },
      {
        name: '1×1 (depthwise, 1 column)',
        c: { cIn: 5, cOut: 5, h: 4, w: 6, kH: 1, kW: 1, stride: 1, padding: 0, groups: 5 },
      },
      // winograd: 3×3, stride 1, padding 1, opted in on the module
      {
        name: 'whole tiles (winograd)',
        c: {
          cIn: 16,
          cOut: 16,
          h: 16,
          w: 16,
          kH: 3,
          kW: 3,
          stride: 1,
          padding: 1,
          groups: 1,
          winograd: true,
        },
      },
      {
        name: 'partial tiles, padded channels (winograd)',
        c: {
          cIn: 6,
          cOut: 10,
          h: 14,
          w: 13,
          kH: 3,
          kW: 3,
          stride: 1,
          padding: 1,
          groups: 1,
          winograd: true,
        },
      },
      {
        name: 'a map smaller than a tile (winograd)',
        c: {
          cIn: 8,
          cOut: 8,
          h: 3,
          w: 2,
          kH: 3,
          kW: 3,
          stride: 1,
          padding: 1,
          groups: 1,
          winograd: true,
        },
      },
      {
        name: 'bias and relu (winograd)',
        c: {
          cIn: 32,
          cOut: 48,
          h: 64,
          w: 64,
          kH: 3,
          kW: 3,
          stride: 1,
          padding: 1,
          groups: 1,
          bias: true,
          activation: 'relu',
          winograd: true,
        },
      },
    ];
    it.for(ROUTES)('$name', async ({ c }, ctx) => {
      needF16(ctx);
      await checkConv2d(c);
    });

    // Every epilogue on every route: act(conv + bias) + residual.
    const EPILOGUES: { name: string; opts: Partial<Conv> }[] = [
      { name: 'bias', opts: { bias: true } },
      { name: 'bias and silu', opts: { bias: true, activation: 'silu' } },
      { name: 'bias and gelu', opts: { bias: true, activation: 'gelu' } },
      { name: 'bias and relu', opts: { bias: true, activation: 'relu' } },
      { name: 'a residual', opts: { residual: true } },
      {
        name: 'bias, silu and a residual',
        opts: { bias: true, activation: 'silu', residual: true },
      },
    ];
    const SHAPES: { route: string; c: Conv }[] = [
      {
        route: 'direct',
        c: { cIn: 3, cOut: 6, h: 7, w: 6, kH: 3, kW: 3, stride: 1, padding: 1, groups: 1 },
      },
      {
        route: 'tiled',
        c: { cIn: 5, cOut: 66, h: 41, w: 39, kH: 3, kW: 3, stride: 1, padding: 1, groups: 1 },
      },
      {
        route: '1x1',
        c: { cIn: 16, cOut: 34, h: 32, w: 32, kH: 1, kW: 1, stride: 1, padding: 0, groups: 1 },
      },
      {
        route: '1x1s',
        c: { cIn: 7, cOut: 5, h: 3, w: 5, kH: 1, kW: 1, stride: 1, padding: 0, groups: 1 },
      },
      {
        route: 'depthwise',
        c: { cIn: 6, cOut: 6, h: 9, w: 21, kH: 3, kW: 3, stride: 1, padding: 1, groups: 6 },
      },
      {
        route: 'winograd',
        c: {
          cIn: 12,
          cOut: 12,
          h: 18,
          w: 18,
          kH: 3,
          kW: 3,
          stride: 1,
          padding: 1,
          groups: 1,
          winograd: true,
        },
      },
    ];
    it.for(SHAPES.flatMap((s) => EPILOGUES.map((e) => ({ ...s, ...e }))))(
      '$route with $name',
      async ({ c, opts }, ctx) => {
        needF16(ctx);
        await checkConv2d({ ...c, ...opts });
      },
    );

    // A conv feeding a concat writes straight into the concat's buffer, at a
    // channel-block offset.
    it.for<{ name: string; c: Conv }>([
      // 1x1: (H·W / 4)·CO4 = 2048
      {
        name: '1x1',
        c: { cIn: 8, cOut: 32, h: 32, w: 32, kH: 1, kW: 1, stride: 1, padding: 0, groups: 1 },
      },
      // 1x1s: H·W % 4 != 0
      {
        name: '1x1s',
        c: { cIn: 8, cOut: 8, h: 3, w: 5, kH: 1, kW: 1, stride: 1, padding: 0, groups: 1 },
      },
      {
        name: 'direct',
        c: { cIn: 4, cOut: 8, h: 8, w: 7, kH: 3, kW: 3, stride: 1, padding: 1, groups: 1 },
      },
      {
        name: 'tiled',
        c: { cIn: 16, cOut: 64, h: 48, w: 48, kH: 3, kW: 3, stride: 1, padding: 1, groups: 1 },
      },
      {
        name: 'depthwise',
        c: { cIn: 8, cOut: 8, h: 10, w: 10, kH: 3, kW: 3, stride: 1, padding: 1, groups: 8 },
      },
      {
        name: 'winograd',
        c: {
          cIn: 8,
          cOut: 8,
          h: 10,
          w: 10,
          kH: 3,
          kW: 3,
          stride: 1,
          padding: 1,
          groups: 1,
          winograd: true,
        },
      },
    ])('$name into the second half of a concat', async ({ c }, ctx) => {
      needF16(ctx);
      const x = image(c.cIn, c.h, c.w, 1);
      const { conv, wt } = await makeConv2d(c, 2);
      const xs = hwc4(x, c.cIn, c.h, c.w);
      const got = await read(cat([xs, conv.forward(xs)], 0));
      expectClose(got, [...x, ...conv2dRef(x, wt, c)], convTol(c));
    });

    // The residual add folds into the conv's epilogue, so it writes at the
    // concat's offset too.
    it('a conv with a residual into the second half of a concat (direct)', async (ctx) => {
      needF16(ctx);
      const c = { cIn: 8, cOut: 8, h: 8, w: 7, kH: 3, kW: 3, stride: 1, padding: 1, groups: 1 };
      const x = image(c.cIn, c.h, c.w, 1);
      const { conv, wt } = await makeConv2d(c, 2);
      const xs = hwc4(x, c.cIn, c.h, c.w);
      const got = await read(cat([xs, add(xs, conv.forward(xs))], 0));
      expectClose(got, [...x, ...conv2dRef(x, wt, c, undefined, x)], convTol(c));
    });

    // Known bug: the 1x1s kernel reads its pixel block from workgroup_id.x
    // alone, but its 1-D workgroup count passes 65535 on a large map with
    // H·W % 4 != 0 and the executor splits it into a 2-D grid. The second
    // grid row repeats the first, so the pixels past 65535·4 are never
    // written. Turn into a plain `it` with the fix.
    // 1x1s: H·W % 4 != 0
    it.fails('a 513×513 map (1x1s)', async (ctx) => {
      needF16(ctx);
      await checkConv2d({
        cIn: 4,
        cOut: 4,
        h: 513,
        w: 513,
        kH: 1,
        kW: 1,
        stride: 1,
        padding: 0,
        groups: 1,
      });
    });

    it('rejects an empty input with an INVALID_ARGUMENT', () => {
      expect(() => upload(root, new Float32Array(0), tensor3d(0, 4, 4))).toThrow(
        expect.objectContaining({ code: 'INVALID_ARGUMENT' }),
      );
    });

    const weightOf = (cOut: number, cols: number, layout?: 'hwc4') =>
      upload(root, new Float32Array(c4of(cOut) * 16 * cols), {
        elems: c4of(cOut) * 16 * cols,
        dtype: 'f16',
        dims: [cOut, cols],
        layout,
      });
    const geometry = { kernelSize: 3, stride: 1, padding: 1, groups: 1 };
    it('refuses a row-major input', (ctx) => {
      needF16(ctx);
      const x = upload(root, new Float32Array(4 * 9), tensor3d(4, 3, 3));
      expect(() => conv2d(x, weightOf(4, 36, 'hwc4'), undefined, geometry)).toThrow(
        /input must be hwc4-stored/,
      );
    });

    it('refuses an unpacked weight', (ctx) => {
      needF16(ctx);
      const x = hwc4(new Float32Array(4 * 9), 4, 3, 3);
      expect(() => conv2d(x, weightOf(4, 36), undefined, geometry)).toThrow(
        /weight must be the hwc4 mat4-tile packing/,
      );
    });

    it('refuses groups that are neither 1 nor depthwise', (ctx) => {
      needF16(ctx);
      const x = hwc4(new Float32Array(4 * 9), 4, 3, 3);
      expect(() =>
        conv2d(x, weightOf(4, 18, 'hwc4'), undefined, { ...geometry, groups: 2 }),
      ).toThrow(/groups=2 is neither 1 nor depthwise/);
    });

    it('refuses an f32 bias', (ctx) => {
      needF16(ctx);
      const x = hwc4(new Float32Array(4 * 9), 4, 3, 3);
      const bias = upload(root, new Float32Array(4), { elems: 4, dtype: 'f32' });
      expect(() => conv2d(x, weightOf(4, 36, 'hwc4'), bias, geometry)).toThrow(
        /bias must be f16 with 4 elems/,
      );
    });
  });

  describe('pools, pad, upsample and resize', () => {
    it.for([
      { name: '2×2 stride 2', c: 4, h: 8, w: 8, k: 2, stride: 2, padding: 0 },
      {
        name: '3×3 stride 2, padding 1, odd channels',
        c: 5,
        h: 9,
        w: 7,
        k: 3,
        stride: 2,
        padding: 1,
      },
      { name: 'SPPF 5×5 stride 1, padding 2', c: 8, h: 10, w: 10, k: 5, stride: 1, padding: 2 },
      { name: 'a window as large as the input', c: 3, h: 4, w: 4, k: 4, stride: 1, padding: 0 },
    ])('maxPool2d $name', async ({ c, h, w, k, stride, padding }, ctx) => {
      needF16(ctx);
      const x = image(c, h, w, 1);
      const got = await read(maxPool2d(hwc4(x, c, h, w), { kernelSize: k, stride, padding }));
      expectClose(got, maxPoolRef(x, c, h, w, k, stride, padding), EXACT);
    });

    it.for([
      { name: '2×2', c: 4, h: 8, w: 6, k: 2, stride: 2 },
      { name: '3×3 stride 1, odd channels', c: 7, h: 5, w: 6, k: 3, stride: 1 },
      { name: '8×8, the widest window', c: 4, h: 16, w: 16, k: 8, stride: 8 },
    ])('avgPool2d $name', async ({ c, h, w, k, stride }, ctx) => {
      needF16(ctx);
      const x = image(c, h, w, 1);
      const hOut = Math.floor((h - k) / stride) + 1;
      const wOut = Math.floor((w - k) / stride) + 1;
      const want = new Float64Array(c * hOut * wOut);
      for (let ch = 0; ch < c; ch++) {
        for (let oy = 0; oy < hOut; oy++) {
          for (let ox = 0; ox < wOut; ox++) {
            let sum = 0;
            for (let ky = 0; ky < k; ky++)
              for (let kx = 0; kx < k; kx++)
                sum += x[(ch * h + oy * stride + ky) * w + ox * stride + kx]!;
            want[(ch * hOut + oy) * wOut + ox] = sum / (k * k);
          }
        }
      }
      const got = await read(avgPool2d(hwc4(x, c, h, w), { kernelSize: k, stride }));
      // The mean is rounded to f16 once.
      expectClose(got, want, { atol: 1e-3, rtol: 1e-3 });
    });

    it('pad2d pads each axis with zeros', async (ctx) => {
      needF16(ctx);
      const [c, h, w, padH, padW] = [5, 3, 4, 1, 2];
      const x = image(c, h, w, 1);
      const [hOut, wOut] = [h + 2 * padH, w + 2 * padW];
      const want = new Float64Array(c * hOut * wOut);
      for (let ch = 0; ch < c; ch++) {
        for (let y = 0; y < h; y++)
          for (let xx = 0; xx < w; xx++)
            want[(ch * hOut + y + padH) * wOut + xx + padW] = x[(ch * h + y) * w + xx]!;
      }
      expectClose(await read(pad2d(hwc4(x, c, h, w), { padH, padW })), want, EXACT);
    });

    it.for([2, 3])('upsample2d by %i repeats each pixel', async (scale, ctx) => {
      needF16(ctx);
      const [c, h, w] = [6, 3, 5];
      const x = image(c, h, w, 1);
      const [hOut, wOut] = [h * scale, w * scale];
      const want = new Float64Array(c * hOut * wOut);
      for (let ch = 0; ch < c; ch++) {
        for (let y = 0; y < hOut; y++) {
          for (let xx = 0; xx < wOut; xx++)
            want[(ch * hOut + y) * wOut + xx] =
              x[(ch * h + Math.floor(y / scale)) * w + Math.floor(xx / scale)]!;
        }
      }
      expectClose(await read(upsample2d(hwc4(x, c, h, w), { scale })), want, EXACT);
    });

    /** torch's interpolate(mode='bilinear'): the source coordinate is
     *  dst·(in − 1)/(out − 1) with align_corners, else max((dst + 0.5)·in/out
     *  − 0.5, 0); the two neighbours clamp to the last row and column. */
    function resizeRef(
      x: Float32Array,
      c: number,
      h: number,
      w: number,
      outH: number,
      outW: number,
      alignCorners: boolean,
    ) {
      const src = (dst: number, n: number, out: number) =>
        alignCorners
          ? out > 1
            ? (dst * (n - 1)) / (out - 1)
            : 0
          : Math.max(((dst + 0.5) * n) / out - 0.5, 0);
      const res = new Float64Array(c * outH * outW);
      for (let ch = 0; ch < c; ch++) {
        for (let oy = 0; oy < outH; oy++) {
          const sy = src(oy, h, outH);
          const y0 = Math.min(Math.floor(sy), h - 1);
          const y1 = Math.min(y0 + 1, h - 1);
          const fy = sy - y0;
          for (let ox = 0; ox < outW; ox++) {
            const sx = src(ox, w, outW);
            const x0 = Math.min(Math.floor(sx), w - 1);
            const x1 = Math.min(x0 + 1, w - 1);
            const fx = sx - x0;
            const at = (y: number, xx: number) => x[(ch * h + y) * w + xx]!;
            const top = at(y0, x0) * (1 - fx) + at(y0, x1) * fx;
            const bot = at(y1, x0) * (1 - fx) + at(y1, x1) * fx;
            res[(ch * outH + oy) * outW + ox] = top * (1 - fy) + bot * fy;
          }
        }
      }
      return res;
    }

    it.for(
      (['alignCorners', 'halfPixel'] as const).flatMap((mode) =>
        [
          { name: 'up 2×', c: 4, h: 5, w: 6, outH: 10, outW: 12 },
          { name: 'up to an odd size', c: 5, h: 4, w: 4, outH: 13, outW: 7 },
          { name: 'down', c: 3, h: 16, w: 12, outH: 5, outW: 7 },
          { name: 'to one pixel', c: 4, h: 3, w: 5, outH: 1, outW: 1 },
          { name: 'from one pixel', c: 4, h: 1, w: 1, outH: 3, outW: 2 },
        ].map((s) => ({ mode, ...s })),
      ),
    )('resizeBilinear2d $mode, $name', async ({ mode, c, h, w, outH, outW }, ctx) => {
      needF16(ctx);
      const x = image(c, h, w, 1);
      const got = await read(resizeBilinear2d(hwc4(x, c, h, w), { outH, outW, mode }));
      // f32 weights, one f16 rounding of the result.
      expectClose(got, resizeRef(x, c, h, w, outH, outW, mode === 'alignCorners'), {
        atol: 1e-3,
        rtol: 1e-3,
      });
    });

    it.each([
      {
        name: 'maxPool2d padding past k/2',
        op: (x: Value) => maxPool2d(x, { kernelSize: 2, stride: 1, padding: 2 }),
        err: /torch requires padding <= k\/2/,
      },
      {
        name: 'avgPool2d a window past the input',
        op: (x: Value) => avgPool2d(x, { kernelSize: 5 }),
        err: /window k=5 larger than input 4×4/,
      },
      {
        name: 'pad2d a negative pad',
        op: (x: Value) => pad2d(x, { padH: -1, padW: 0 }),
        err: /pads must be non-negative integers/,
      },
      {
        name: 'upsample2d a fractional scale',
        op: (x: Value) => upsample2d(x, { scale: 1.5 }),
        err: /scale must be a positive integer, got 1.5/,
      },
      {
        name: 'resizeBilinear2d an empty size',
        op: (x: Value) => resizeBilinear2d(x, { outH: 0, outW: 3 }),
        err: /bad output size 0×3/,
      },
    ])('$name is refused', ({ op, err }) => {
      const x = upload(root, new Float32Array(16 * 4), hwc4Meta(4, 4, 4));
      expect(() => op(x)).toThrow(err);
    });
  });

  describe('layout and channel ops', () => {
    it.for(
      (['f32', 'f16'] as const).flatMap((from) =>
        (['f32', 'f16'] as const).flatMap((to) =>
          [
            { c: 1, h: 1, w: 1 },
            { c: 5, h: 3, w: 7 },
            { c: 8, h: 4, w: 4 },
          ].map((s) => ({ from, to, ...s })),
        ),
      ),
    )(
      'toHwc4 from $from and toChw to $to round-trip [$c, $h, $w]',
      async ({ from, to, c, h, w }, ctx) => {
        needF16(ctx);
        const x = image(c, h, w, 1);
        const stored = toHwc4(upload(root, x, { ...tensor3d(c, h, w), dtype: from }));
        expect(stored.shape).toEqual(hwc4Meta(c, h, w));
        expectClose(await toArray(toChw(stored, to), ex), x, EXACT);
      },
    );

    it.for([false, true])('channelAffine, shift %s', async (withShift, ctx) => {
      needF16(ctx);
      const [c, h, w] = [6, 3, 5];
      const x = image(c, h, w, 1);
      const scale = randomF32(c, 2);
      const shift = randomF32(c, 3);
      const y = channelAffine(
        hwc4(x, c, h, w),
        upload(root, scale, { elems: c, dtype: 'f32' }),
        withShift ? upload(root, shift, { elems: c, dtype: 'f32' }) : undefined,
      );
      const want = Float64Array.from(x, (v, i) => {
        const ch = Math.floor(i / (h * w));
        return v * scale[ch]! + (withShift ? shift[ch]! : 0);
      });
      expectClose(await read(y), want, { atol: 1e-3, rtol: 1e-3 });
    });

    it('slice keeps channel blocks [4, 12) of 14', async (ctx) => {
      needF16(ctx);
      const x = image(14, 3, 3, 1);
      expectClose(await read(slice(hwc4(x, 14, 3, 3), 0, 4, 12)), x.subarray(36, 108), EXACT);
    });

    // concatChannelsHwc4 copies inputs it cannot elide; an input whose only
    // consumer is the concat and whose producer is conv2d, maxPool2d,
    // upsample2d, a channel slice or another concat writes straight into the
    // concat's buffer at its channel offset (graph/evalCore.ts).
    const conv1x1 = async (x: Value, cIn: number, cOut: number, seed: number) => {
      const c: Conv = { cIn, cOut, h: 4, w: 4, kH: 1, kW: 1, stride: 1, padding: 0, groups: 1 };
      const { conv, wt } = await makeConv2d(c, seed);
      return { y: conv.forward(x), wt, c };
    };

    it('cat of two uploads, both copied', async (ctx) => {
      needF16(ctx);
      const a = image(4, 4, 4, 1);
      const b = image(8, 4, 4, 2);
      const got = await read(cat([hwc4(a, 4, 4, 4), hwc4(b, 8, 4, 4)], 0));
      expectClose(got, [...a, ...b], EXACT);
    });

    it('cat of two convs and a pool, each written in place', async (ctx) => {
      needF16(ctx);
      const x = image(4, 4, 4, 1);
      const xs = hwc4(x, 4, 4, 4);
      const first = await conv1x1(xs, 4, 8, 2);
      const second = await conv1x1(xs, 4, 4, 3);
      const pooled = maxPool2d(hwc4(x, 4, 4, 4), { kernelSize: 3, stride: 1, padding: 1 });
      const got = await read(cat([first.y, second.y, pooled], 0));
      const want = [
        ...conv2dRef(x, first.wt, first.c),
        ...conv2dRef(x, second.wt, second.c),
        ...maxPoolRef(x, 4, 4, 4, 3, 1, 1),
      ];
      expectClose(got, want, { atol: 5e-3, rtol: 1e-2 });
    });

    it('cat of an upsample and a slice, written in place, and an upload, copied', async (ctx) => {
      needF16(ctx);
      const small = image(4, 2, 2, 1);
      const wide = image(8, 4, 4, 2);
      const extra = image(4, 4, 4, 3);
      const got = await read(
        cat(
          [
            upsample2d(hwc4(small, 4, 2, 2), { scale: 2 }),
            slice(hwc4(wide, 8, 4, 4), 0, 4, 8),
            hwc4(extra, 4, 4, 4),
          ],
          0,
        ),
      );
      const up = Array.from({ length: 4 * 16 }, (_, i) => {
        const [ch, y, xx] = [Math.floor(i / 16), Math.floor((i % 16) / 4), i % 4];
        return small[(ch * 2 + Math.floor(y / 2)) * 2 + Math.floor(xx / 2)]!;
      });
      expectClose(got, [...up, ...wide.subarray(64, 128), ...extra], EXACT);
    });

    it('refuses to toHwc4 an hwc4 input', (ctx) => {
      needF16(ctx);
      const x = upload(root, new Float32Array(16), hwc4Meta(4, 2, 2));
      expect(() => toHwc4(x)).toThrow(/already stored 'hwc4'/);
    });

    it('refuses to toChw a row-major input', () => {
      const x = upload(root, new Float32Array(16), tensor3d(4, 2, 2));
      expect(() => toChw(x, 'f32')).toThrow(/input must be hwc4-stored/);
    });

    it('refuses a channel slice that is not whole blocks of 4', (ctx) => {
      needF16(ctx);
      const x = upload(root, new Float32Array(32), hwc4Meta(8, 2, 2));
      expect(() => slice(x, 0, 2, 6)).toThrow(/hwc4 channel ranges must align to 4/);
    });

    it('refuses to cat channel counts that are not multiples of 4', (ctx) => {
      needF16(ctx);
      const a = upload(root, new Float32Array(16), hwc4Meta(3, 2, 2));
      const b = upload(root, new Float32Array(16), hwc4Meta(4, 2, 2));
      expect(() => cat([a, b], 0)).toThrow(
        /hwc4 channel counts must be multiples of 4, got 3 \+\+ 4/,
      );
    });

    it('refuses a channelAffine scale of the wrong length', (ctx) => {
      needF16(ctx);
      const x = upload(root, new Float32Array(16), hwc4Meta(4, 2, 2));
      const scale = upload(root, new Float32Array(3), { elems: 3, dtype: 'f32' });
      expect(() => channelAffine(x, scale)).toThrow(/scale must be f32 with 4 elems, got 3 f32/);
    });

    // Row-major ops would read hwc4 bytes as garbage, so they refuse it.
    it('a row-major op refuses an hwc4 input', (ctx) => {
      needF16(ctx);
      const x = upload(root, new Float32Array(16), { ...hwc4Meta(4, 2, 2), dims: [4, 4] });
      expect(() => softmax(x)).toThrow(
        /this op reads row-major storage; convert with toChw\(\) first/,
      );
    });
  });
});
