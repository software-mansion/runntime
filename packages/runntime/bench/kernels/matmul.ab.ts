/** Every route gpu/dispatch/matmul.ts can pick, plus the epilogue variants,
 *  quantized weights, matmulGather and argmaxDot. Shapes come from the zoo. */
import { abSuite, type AbCase, type Version } from '../ab.ts';
import { ids, mat, quantWeight, randMat } from './inputs.ts';

type Float = 'f32' | 'f16';
type Act = 'gelu' | 'silu' | 'tanh' | 'relu';

/** [m, k] × [k, n] with an optional bias [n], addend [m, n] and activation. */
function mm(
  name: string,
  m: number,
  k: number,
  n: number,
  opts: { dtype?: Float; bias?: boolean; addend?: boolean; act?: Act; baseRow?: number } = {},
): AbCase {
  const { dtype = 'f32' } = opts;
  return {
    name,
    needs: dtype === 'f16' ? ['shader-f16'] : [],
    build: (version: Version) => {
      const a = randMat(version, m, k, dtype);
      const b = randMat(version, k + (opts.baseRow ?? 0), n, dtype);
      const bias = opts.bias ? randMat(version, 1, n, dtype) : undefined;
      const addend = opts.addend ? randMat(version, m, n, dtype) : undefined;
      return () =>
        version.lib.matmul(a, b, { bias, addend, activation: opts.act, baseRow: opts.baseRow });
    },
  };
}

/** The same routes on a device without subgroups, where smallm does not exist. */
function noSg(testCase: AbCase): AbCase {
  return { ...testCase, name: `${testCase.name}, no subgroups`, noSubgroups: true };
}

function quant(name: string, t: number, e: number, k: number, o: number, bits: 8 | 4): AbCase {
  const groupSize = 32;
  return {
    name,
    build: (version) => {
      const a = randMat(version, t, k);
      const { w, scales } = quantWeight(version, e, k, o, bits, groupSize);
      // The last expert, so the base row and scale offsets are nonzero.
      const baseRow = (e - 1) * k;
      const scaleBase = (e - 1) * (k / groupSize) * o;
      return () => version.lib.matmul(a, w, { scales, bits, groupSize, baseRow, scaleBase });
    },
  };
}

function gather(name: string, t: number, e: number, k: number, n: number): AbCase {
  return {
    name,
    build: (version) => {
      const a = randMat(version, t, k);
      const w = randMat(version, e * k, n);
      const bias = randMat(version, e, n);
      const expertIdx = version.upload(ids(t, e), mat(t, 1));
      return () => version.lib.matmulGather(a, w, expertIdx, { bias });
    },
  };
}

function argmaxDot(name: string, v: number, n: number, dtype: Float = 'f32'): AbCase {
  return {
    name,
    needs: dtype === 'f16' ? ['shader-f16'] : [],
    build: (version) => {
      const w = randMat(version, v, n, dtype);
      const x = randMat(version, 1, n, dtype);
      return () => version.lib.argmaxDot(w, x);
    },
  };
}

abSuite('matmul', [
  // gemvVec4: m = 1, no activation, n a multiple of 4.
  mm('gemvVec4, Moonshine-tiny decode [1,288]×[288,864]', 1, 288, 864),
  mm('gemvVec4 f16 + bias, MobileNetV4 classifier [1,1280]×[1280,1000]', 1, 1280, 1000, {
    dtype: 'f16',
    bias: true,
  }),
  // gemv: the same with n not a multiple of 4.
  mm('gemv + addend, odd N [1,288]×[288,33]', 1, 288, 33, { addend: true }),

  // smallm: subgroups, 4 ≤ m ≤ 32, m and n multiples of 4. When tiledVec4 also
  // fits, only for n < 1024 (TILED_MIN_N_FEW_ROWS).
  mm('smallm + addend, MiniLM fc2 bucket 32 [32,1536]×[1536,384]', 32, 1536, 384, {
    bias: true,
    addend: true,
  }),
  mm('smallm, K not a multiple of 32 [4,100]×[100,64]', 4, 100, 64),

  // tiledVec4: m ≥ 8, k ≥ 64, k and n multiples of 4, f32 or f16.
  mm('tiledVec4, MiniLM qkv bucket 16, n ≥ 1024 so not smallm [16,384]×[384,1152]', 16, 384, 1152, {
    bias: true,
  }),
  mm(
    'tiledVec4 + gelu, MiniLM fc1 bucket 16, n ≥ 1024 so not smallm [16,384]×[384,1536]',
    16,
    384,
    1536,
    {
      bias: true,
      act: 'gelu',
    },
  ),
  mm(
    'tiledVec4 f16, MiniLM qkv bucket 32, n ≥ 1024 so not smallm [32,384]×[384,1152]',
    32,
    384,
    1152,
    {
      dtype: 'f16',
    },
  ),
  mm('tiledVec4, MiniLM qkv bucket 128 [128,384]×[384,1152]', 128, 384, 1152, { bias: true }),
  mm('tiledVec4 + silu, m not a multiple of the tile [130,200]×[200,300]', 130, 200, 300, {
    act: 'silu',
  }),
  mm('tiledVec4 + addend, MiniLM fc2 bucket 512 [512,1536]×[1536,384]', 512, 1536, 384, {
    bias: true,
    addend: true,
  }),
  mm('tiledVec4, B past 2^24 elements [9,2560]×[2560,9728]', 9, 2560, 9728),
  mm('tiledVec4 f16, MiniLM qkv bucket 64 [64,384]×[384,1152]', 64, 384, 1152, {
    dtype: 'f16',
    bias: true,
  }),
  mm('tiledVec4 f16, YOLO26n PSA [64,400]×[400,400]', 64, 400, 400, { dtype: 'f16' }),
  mm('tiledVec4 f16 + gelu + addend [256,384]×[384,1536]', 256, 384, 1536, {
    dtype: 'f16',
    bias: true,
    act: 'gelu',
    addend: true,
  }),

  // tiled: f32, m ≥ 8, when tiledVec4 does not fit.
  mm('tiled, privacy-filter unembed, N=33 [64,640]×[640,33]', 64, 640, 33),

  // plain: everything else.
  mm('plain, m = 2 [2,384]×[384,384]', 2, 384, 384),
  mm('plain, m = 1 with tanh [1,288]×[288,288]', 1, 288, 288, { act: 'tanh' }),
  mm('plain f16, YOLO26n PSA scores, K=32 [400,32]×[32,400]', 400, 32, 400, { dtype: 'f16' }),
  mm('plain f16, N not a multiple of 4 [64,640]×[640,33]', 64, 640, 33, { dtype: 'f16' }),
  mm('plain f16 + relu, m = 6 [6,128]×[128,128]', 6, 128, 128, { dtype: 'f16', act: 'relu' }),
  mm('plain, expert view baseRow, privacy-filter [32,640]×[640,640]', 32, 640, 640, {
    baseRow: 3 * 640,
  }),

  // Without subgroups, smallm shapes fall to tiledVec4 or plain.
  noSg(mm('tiledVec4, MiniLM qkv bucket 16 [16,384]×[384,1152]', 16, 384, 1152, { bias: true })),
  noSg(mm('plain, m = 4 [4,384]×[384,1152]', 4, 384, 1152)),

  // Quantized weights, privacy-filter MoE expert.
  quant('quantW int8, privacy-filter expert [32,640]×[640,640]', 32, 16, 640, 640, 8),
  quant('quantW int4, privacy-filter expert [32,640]×[640,640]', 32, 16, 640, 640, 4),
  quant('quantW int8, decode [1,640]×[640,640]', 1, 16, 640, 640, 8),

  // matmulGather: one expert per row.
  gather('matmulGather f32, privacy-filter [64,640], 16 experts', 64, 16, 640, 640),

  // argmaxDot: greedy next token.
  argmaxDot('argmaxDot, Moonshine-tiny vocab [32768,288]', 32768, 288),
  argmaxDot('argmaxDot f16, streaming-medium [32768,640]', 32768, 640, 'f16'),
]);
