/** Every route gpu/dispatch/matmul.ts can pick, plus the epilogue variants,
 *  quantized weights, matmulGather and argmaxDot. Shapes come from the zoo. */
import { abSuite, type AbCase, type Side } from '../ab.ts';
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
    build: (s: Side) => {
      const a = randMat(s, m, k, dtype);
      const b = randMat(s, k + (opts.baseRow ?? 0), n, dtype);
      const bias = opts.bias ? randMat(s, 1, n, dtype) : undefined;
      const addend = opts.addend ? randMat(s, m, n, dtype) : undefined;
      return () =>
        s.lib.matmul(a, b, { bias, addend, activation: opts.act, baseRow: opts.baseRow });
    },
  };
}

/** The same routes on a device without subgroups, where smallm does not exist. */
function noSg(c: AbCase): AbCase {
  return { ...c, name: `${c.name}, no subgroups`, noSubgroups: true };
}

function quant(name: string, t: number, e: number, k: number, o: number, bits: 8 | 4): AbCase {
  const groupSize = 32;
  return {
    name,
    build: (s) => {
      const a = randMat(s, t, k);
      const { w, scales } = quantWeight(s, e, k, o, bits, groupSize);
      // The last expert, so the base row and scale offsets are nonzero.
      const baseRow = (e - 1) * k;
      const scaleBase = (e - 1) * (k / groupSize) * o;
      return () => s.lib.matmul(a, w, { scales, bits, groupSize, baseRow, scaleBase });
    },
  };
}

function gather(name: string, t: number, e: number, k: number, n: number, bits?: 8 | 4): AbCase {
  const groupSize = 32;
  return {
    name,
    build: (s) => {
      const a = randMat(s, t, k);
      const bias = randMat(s, e, n);
      const expertIdx = s.up(ids(t, e), mat(t, 1));
      if (bits === undefined) {
        const w = randMat(s, e * k, n);
        return () => s.lib.matmulGather(a, w, expertIdx, { bias });
      }
      const { w, scales } = quantWeight(s, e, k, n, bits, groupSize);
      return () => s.lib.matmulGather(a, w, expertIdx, { bias, scales, bits, groupSize });
    },
  };
}

function argmaxDot(name: string, v: number, n: number, dtype: Float = 'f32'): AbCase {
  return {
    name,
    needs: dtype === 'f16' ? ['shader-f16'] : [],
    build: (s) => {
      const w = randMat(s, v, n, dtype);
      const x = randMat(s, 1, n, dtype);
      return () => s.lib.argmaxDot(w, x);
    },
  };
}

abSuite('matmul', [
  // gemv: m = 1, no activation.
  mm('gemv, Moonshine-tiny decode [1,288]×[288,864]', 1, 288, 864),
  mm('gemv f16 + bias, MobileNetV4 classifier [1,1280]×[1280,1000]', 1, 1280, 1000, {
    dtype: 'f16',
    bias: true,
  }),
  mm('gemv + addend, odd N [1,288]×[288,33]', 1, 288, 33, { addend: true }),

  // smallm: subgroups, 4 ≤ m ≤ 32, m and n multiples of 4.
  mm('smallm, MiniLM qkv bucket 16 [16,384]×[384,1152]', 16, 384, 1152, { bias: true }),
  mm('smallm + gelu, MiniLM fc1 bucket 16 [16,384]×[384,1536]', 16, 384, 1536, {
    bias: true,
    act: 'gelu',
  }),
  mm('smallm + addend, MiniLM fc2 bucket 32 [32,1536]×[1536,384]', 32, 1536, 384, {
    bias: true,
    addend: true,
  }),
  mm('smallm, K not a multiple of 32 [4,100]×[100,64]', 4, 100, 64),
  mm('smallm f16, MiniLM qkv bucket 32 [32,384]×[384,1152]', 32, 384, 1152, { dtype: 'f16' }),

  // tiled: f32, m ≥ 8 and not smallm.
  mm('tiled, MiniLM qkv bucket 128 [128,384]×[384,1152]', 128, 384, 1152, { bias: true }),
  mm('tiled + silu, odd sizes [130,200]×[200,300]', 130, 200, 300, { act: 'silu' }),
  mm('tiled, privacy-filter unembed, N=33 [64,640]×[640,33]', 64, 640, 33),
  mm('tiled + addend, MiniLM fc2 bucket 512 [512,1536]×[1536,384]', 512, 1536, 384, {
    bias: true,
    addend: true,
  }),

  mm('tiled, B past 2^24 elements [9,2560]×[2560,9728]', 9, 2560, 9728),

  // tiledF16: f16, m ≥ 8, k ≥ 64, k and n multiples of 4.
  mm('tiledF16, MiniLM qkv bucket 64 [64,384]×[384,1152]', 64, 384, 1152, {
    dtype: 'f16',
    bias: true,
  }),
  mm('tiledF16, YOLO26n PSA [64,400]×[400,400]', 64, 400, 400, { dtype: 'f16' }),
  mm('tiledF16 + gelu + addend [256,384]×[384,1536]', 256, 384, 1536, {
    dtype: 'f16',
    bias: true,
    act: 'gelu',
    addend: true,
  }),

  // plain: everything else.
  mm('plain, m = 2 [2,384]×[384,384]', 2, 384, 384),
  mm('plain, m = 1 with tanh [1,288]×[288,288]', 1, 288, 288, { act: 'tanh' }),
  mm('plain f16, YOLO26n PSA scores, K=32 [400,32]×[32,400]', 400, 32, 400, { dtype: 'f16' }),
  mm('plain f16, N not a multiple of 4 [64,640]×[640,33]', 64, 640, 33, { dtype: 'f16' }),
  mm('plain f16 + relu, m = 6 [6,128]×[128,128]', 6, 128, 128, { dtype: 'f16', act: 'relu' }),
  mm('plain, expert view baseRow, privacy-filter [32,640]×[640,640]', 32, 640, 640, {
    baseRow: 3 * 640,
  }),

  // Without subgroups, smallm shapes fall to tiled or plain.
  noSg(mm('tiled, MiniLM qkv bucket 16 [16,384]×[384,1152]', 16, 384, 1152, { bias: true })),
  noSg(mm('plain, m = 4 [4,384]×[384,1152]', 4, 384, 1152)),

  // Quantized weights, privacy-filter MoE expert.
  quant('quantW int8, privacy-filter expert [32,640]×[640,640]', 32, 16, 640, 640, 8),
  quant('quantW int4, privacy-filter expert [32,640]×[640,640]', 32, 16, 640, 640, 4),
  quant('quantW int8, decode [1,640]×[640,640]', 1, 16, 640, 640, 8),

  // matmulGather: one expert per row.
  gather('matmulGather f32, privacy-filter [64,640], 16 experts', 64, 16, 640, 640),
  gather('matmulGather int8, privacy-filter [64,640], 128 experts', 64, 128, 640, 640, 8),
  gather('matmulGather int4, privacy-filter [64,640], 128 experts', 64, 128, 640, 640, 4),

  // argmaxDot: greedy next token.
  argmaxDot('argmaxDot, Moonshine-tiny vocab [32768,288]', 32768, 288),
  argmaxDot('argmaxDot f16, streaming-medium [32768,640]', 32768, 640, 'f16'),
]);
