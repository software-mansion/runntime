import { describe, it, onTestFinished } from 'vitest';
import { toArray } from '../../../src/core/gpu/eval.ts';
import type { SlotAct } from '../../../src/core/graph/ops.ts';
import { matrix } from '../../../src/core/graph/value.ts';
import { Conv1d, ConvTranspose1d, Linear } from '../../../src/core/nn/layers.ts';
import {
  bufferSource,
  fromSafetensors,
  memoryStateDict,
  type LazyStateDict,
} from '../../../src/core/weights/safetensors.ts';
import { expectClose, gpu, randomF32, requireFeature, roundF16, upload } from '../../gpu.ts';
import { ACTIVATION_REF } from '../kernels/matmul/checkMatmul.ts';
import { quantFile } from '../weights/checkpoint.ts';

const { root, ex } = await gpu();

interface Case {
  m: number;
  inF: number;
  outF: number;
  groupSize: number;
  bias: boolean;
  addend?: boolean;
  activation?: SlotAct;
  /** Takes the tiled kernel, which stages the dequantized weight in f16
   *  workgroup memory for an f16 input. */
  tiled?: boolean;
}

/** A Linear loaded from a quantized file matches a CPU loop of
 *  act(x·W + bias) + addend in f64, with W dequantized from the same ints as
 *  int · f16 scale. */
async function checkQuantizedLinear(c: Case, bits: 4 | 8, dtype: 'f32' | 'f16') {
  const { m, inF, outF, groupSize } = c;
  const half = 2 ** (bits - 1);
  // torch [out, in] ints over the full signed range, positive scales
  const ints = Array.from(randomF32(outF * inF, 1), (v) =>
    Math.min(half - 1, Math.floor(v * half)),
  );
  const scales = roundF16(
    randomF32((outF * inF) / groupSize, 2).map((v) => (Math.abs(v) + 0.1) / half),
  );
  // Rounded to f16 for both dtypes, so the f16 upload stores them exactly.
  const bias = roundF16(randomF32(outF, 3));
  const xs = roundF16(randomF32(m * inF, 4));
  const addend = roundF16(randomF32(m * outF, 5));

  const layer = new Linear(inF, outF, { bias: c.bias, activation: c.activation });
  if (dtype === 'f16') layer.half(root);
  onTestFinished(() => layer.dispose());
  const file = quantFile(bits, groupSize, {
    weight: { ints, scales: [...scales], shape: [outF, inF] },
  });
  const sd = await fromSafetensors(bufferSource(file), { chunkBytes: 0 });
  if (c.bias) sd.tensors.set('bias', memoryStateDict({ bias }).tensors.get('bias')!);
  await layer.loadStateDict(sd, { root });

  const x = upload(root, xs, { ...matrix(m, inF), dtype });
  const call = c.addend ? { addend: upload(root, addend, { ...matrix(m, outF), dtype }) } : {};
  const got = await toArray(layer.forward(x, call), ex);

  const dequant = Float32Array.from(ints, (q, i) => q * scales[Math.floor(i / groupSize)]!);
  const weight = dtype === 'f16' && c.tiled ? roundF16(dequant) : dequant;
  const want = new Float64Array(m * outF);
  for (let i = 0; i < m; i++) {
    for (let o = 0; o < outF; o++) {
      let sum = 0;
      for (let p = 0; p < inF; p++) sum += xs[i * inF + p]! * weight[o * inF + p]!;
      if (c.bias) sum += bias[o]!;
      if (c.activation) sum = ACTIVATION_REF[c.activation](sum);
      if (c.addend) sum += addend[i * outF + o]!;
      want[i * outF + o] = sum;
    }
  }
  // The kernel sums in f32, so its error grows with inF; f16 also rounds the
  // output to half precision, 2^-11 relative.
  const tol = dtype === 'f16' ? { atol: 1e-5 * inF, rtol: 1e-3 } : { atol: 1e-6 * inF, rtol: 1e-5 };
  expectClose(got, want, tol);
}

describe('Linear with a quantized weight', () => {
  // A quantized Linear takes matmulQuantW below 48·1024 output values (m ·
  // outF) and the tiled kernel from there, for either dtype and any epilogue.
  const CASES: (Case & { name: string; dtypes: ('f32' | 'f16')[] })[] = [
    // matmulQuantW: m · outF < 48·1024
    { name: '1 row', m: 1, inF: 64, outF: 48, groupSize: 16, bias: false, dtypes: ['f32', 'f16'] },
    { name: '8 rows', m: 8, inF: 64, outF: 48, groupSize: 16, bias: false, dtypes: ['f32'] },
    { name: 'bias, 7 rows', m: 7, inF: 64, outF: 48, groupSize: 16, bias: true, dtypes: ['f32'] },
    {
      name: 'activation, 3 rows',
      m: 3,
      inF: 64,
      outF: 48,
      groupSize: 16,
      bias: false,
      activation: 'gelu',
      dtypes: ['f32', 'f16'],
    },
    {
      name: 'addend, 2 rows',
      m: 2,
      inF: 64,
      outF: 48,
      groupSize: 16,
      bias: false,
      addend: true,
      dtypes: ['f32', 'f16'],
    },
    {
      name: 'bias, 5 rows',
      m: 5,
      inF: 64,
      outF: 48,
      groupSize: 16,
      bias: true,
      dtypes: ['f16'],
    },
    {
      name: 'bias, gelu and addend, 37 rows',
      m: 37,
      inF: 256,
      outF: 136,
      groupSize: 32,
      bias: true,
      addend: true,
      activation: 'gelu',
      dtypes: ['f32', 'f16'],
    },
    {
      name: 'one group per row, 767 rows',
      m: 767,
      inF: 128,
      outF: 64,
      groupSize: 128,
      bias: false,
      dtypes: ['f32', 'f16'],
    },
    // tiled: m · outF >= 48·1024
    {
      name: 'one group per row, 768 rows',
      m: 768,
      inF: 128,
      outF: 64,
      groupSize: 128,
      bias: false,
      tiled: true,
      dtypes: ['f32', 'f16'],
    },
    {
      name: 'partial tiles with bias, gelu and addend, 400 rows',
      m: 400,
      inF: 256,
      outF: 136,
      groupSize: 32,
      bias: true,
      addend: true,
      activation: 'gelu',
      tiled: true,
      dtypes: ['f32', 'f16'],
    },
  ];
  it.for(
    CASES.flatMap(({ dtypes, ...c }) =>
      dtypes.flatMap((dtype) => ([4, 8] as const).map((bits) => ({ ...c, dtype, bits }))),
    ),
  )('$dtype int$bits $name: [$m, $inF] → $outF, groupSize $groupSize', async (c, ctx) => {
    if (c.dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    await checkQuantizedLinear(c, c.bits, c.dtype);
  });
});

interface ConvCase {
  name: string;
  kind: 'conv' | 'transpose';
  tIn: number;
  cIn: number;
  cOut: number;
  kernel: number;
  stride: number;
  padding: number;
  /** Values of a torch row per scale. */
  groupSize: number;
  bias: boolean;
  activation?: 'gelu';
}

/** A conv loaded from a quantized file of its torch weight, [out, in·kernel]
 *  or a transposed conv's [in, out·kernel], matches the same conv loaded
 *  with the dequantized floats, int · f16 scale. */
async function checkQuantizedConv(c: ConvCase, bits: 4 | 8, dtype: 'f32' | 'f16') {
  const { tIn, cIn, cOut, kernel, stride, padding, groupSize } = c;
  const half = 2 ** (bits - 1);
  const [rows, cols] = c.kind === 'conv' ? [cOut, cIn * kernel] : [cIn, cOut * kernel];
  const ints = Array.from(randomF32(rows * cols, 1), (v) =>
    Math.min(half - 1, Math.floor(v * half)),
  );
  const scales = roundF16(
    randomF32((rows * cols) / groupSize, 2).map((v) => (Math.abs(v) + 0.1) / half),
  );
  const bias = roundF16(randomF32(cOut, 3));
  const xs = roundF16(randomF32(tIn * cIn, 4));
  const dequant = Float32Array.from(ints, (q, i) => q * scales[Math.floor(i / groupSize)]!);
  const torchWeight = {
    data: dequant,
    shape: c.kind === 'conv' ? [cOut, cIn, kernel] : [cIn, cOut, kernel],
  };
  const make = () =>
    c.kind === 'conv'
      ? new Conv1d(cIn, cOut, kernel, { stride, padding, bias: c.bias, activation: c.activation })
      : new ConvTranspose1d(cIn, cOut, kernel, { stride, padding, bias: c.bias });
  const run = async (layer: Conv1d | ConvTranspose1d, sd: LazyStateDict) => {
    if (dtype === 'f16') layer.half(root);
    onTestFinished(() => layer.dispose());
    await layer.loadStateDict(sd, { root });
    return toArray(layer.forward(upload(root, xs, { ...matrix(tIn, cIn), dtype })), ex);
  };
  const file = quantFile(bits, groupSize, {
    weight: { ints, scales: [...scales], shape: [rows, cols] },
  });
  const quantSd = await fromSafetensors(bufferSource(file), { chunkBytes: 0 });
  if (c.bias) quantSd.tensors.set('bias', memoryStateDict({ bias }).tensors.get('bias')!);
  const got = await run(make(), quantSd);
  const want = await run(
    make(),
    memoryStateDict({ weight: torchWeight, ...(c.bias ? { bias } : {}) }),
  );
  // Both sum in f32 in another order; f16 rounds the output to half precision.
  const tol = dtype === 'f16' ? { atol: 2e-3, rtol: 2e-3 } : { atol: 1e-5, rtol: 1e-5 };
  expectClose(got, want, tol);
}

describe('Conv1d and ConvTranspose1d with a quantized weight', () => {
  const CASES: ConvCase[] = [
    {
      name: 'conv, padded, groups that end inside a channel, bias and gelu',
      kind: 'conv',
      tIn: 40,
      cIn: 16,
      cOut: 24,
      kernel: 7,
      stride: 1,
      padding: 3,
      groupSize: 16,
      bias: true,
      activation: 'gelu',
    },
    {
      name: 'conv 1×1, stride 2',
      kind: 'conv',
      tIn: 33,
      cIn: 32,
      cOut: 16,
      kernel: 1,
      stride: 2,
      padding: 0,
      groupSize: 32,
      bias: false,
    },
    // 1 and 3 output channels pad up to the packing width and are cut after.
    {
      name: 'conv to 1 channel, bias',
      kind: 'conv',
      tIn: 30,
      cIn: 32,
      cOut: 1,
      kernel: 7,
      stride: 1,
      padding: 3,
      groupSize: 56,
      bias: true,
    },
    {
      name: 'conv to 3 channels',
      kind: 'conv',
      tIn: 20,
      cIn: 8,
      cOut: 3,
      kernel: 3,
      stride: 1,
      padding: 1,
      groupSize: 6,
      bias: true,
    },
    {
      name: 'transposed, stride 4, bias',
      kind: 'transpose',
      tIn: 12,
      cIn: 16,
      cOut: 8,
      kernel: 8,
      stride: 4,
      padding: 2,
      groupSize: 16,
      bias: true,
    },
    {
      name: 'transposed, stride 2',
      kind: 'transpose',
      tIn: 9,
      cIn: 32,
      cOut: 8,
      kernel: 4,
      stride: 2,
      padding: 1,
      groupSize: 32,
      bias: false,
    },
  ];
  it.for(
    (['f32', 'f16'] as const).flatMap((dtype) =>
      CASES.flatMap((c) => ([8, 4] as const).map((bits) => ({ ...c, dtype, bits }))),
    ),
  )('$dtype int$bits $name', async (c, ctx) => {
    if (c.dtype === 'f16') requireFeature(ctx, root, 'shader-f16');
    await checkQuantizedConv(c, c.bits, c.dtype);
  });
});
