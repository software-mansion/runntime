import { describe, it, onTestFinished } from 'vitest';
import { toArray } from '../../../src/core/gpu/eval.ts';
import type { SlotAct } from '../../../src/core/graph/ops.ts';
import { matrix } from '../../../src/core/graph/value.ts';
import { Linear } from '../../../src/core/nn/layers.ts';
import {
  bufferSource,
  fromSafetensors,
  memoryStateDict,
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

  // The f16 kernel stages the dequantized weight in f16 workgroup memory.
  const dequant = Float32Array.from(ints, (q, i) => q * scales[Math.floor(i / groupSize)]!);
  const weight = dtype === 'f16' ? roundF16(dequant) : dequant;
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
  // A quantized Linear takes the plain quantW kernel only for an f32 input
  // under 8 rows with no epilogue; anything else takes the tiled kernel.
  const CASES: (Case & { name: string; dtypes: ('f32' | 'f16')[] })[] = [
    // plain: f32, m < 8, no bias
    { name: 'plain, 1 row', m: 1, inF: 64, outF: 48, groupSize: 16, bias: false, dtypes: ['f32'] },
    { name: 'plain, 7 rows', m: 7, inF: 64, outF: 48, groupSize: 16, bias: false, dtypes: ['f32'] },
    // tiled: m >= 8
    { name: 'tiled, 8 rows', m: 8, inF: 64, outF: 48, groupSize: 16, bias: false, dtypes: ['f32'] },
    // tiled: a bias, an activation or an addend is an epilogue, whatever m
    {
      name: 'tiled by its bias, 7 rows',
      m: 7,
      inF: 64,
      outF: 48,
      groupSize: 16,
      bias: true,
      dtypes: ['f32'],
    },
    {
      name: 'tiled by its activation, 3 rows',
      m: 3,
      inF: 64,
      outF: 48,
      groupSize: 16,
      bias: false,
      activation: 'gelu',
      dtypes: ['f32', 'f16'],
    },
    {
      name: 'tiled by its addend, 2 rows',
      m: 2,
      inF: 64,
      outF: 48,
      groupSize: 16,
      bias: false,
      addend: true,
      dtypes: ['f32', 'f16'],
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
    // tiled: an f16 input, whatever m
    {
      name: 'f16 input, 1 row',
      m: 1,
      inF: 64,
      outF: 48,
      groupSize: 16,
      bias: false,
      dtypes: ['f16'],
    },
    {
      name: 'f16 input with bias, 5 rows',
      m: 5,
      inF: 64,
      outF: 48,
      groupSize: 16,
      bias: true,
      dtypes: ['f16'],
    },
    {
      name: 'partial tiles with bias, 37 rows',
      m: 37,
      inF: 256,
      outF: 136,
      groupSize: 32,
      bias: true,
      dtypes: ['f32', 'f16'],
    },
    {
      name: 'one group per row, 37 rows',
      m: 37,
      inF: 128,
      outF: 64,
      groupSize: 128,
      bias: false,
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
