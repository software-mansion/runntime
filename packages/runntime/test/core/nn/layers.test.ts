import { describe, expect, it, vi } from 'vitest';
import { pending, type ValueMeta } from '../../../src/core/graph/value.ts';
import {
  BatchNorm2d,
  Conv1d,
  Conv2d,
  ConvTranspose2d,
  Embedding,
  GroupNorm,
  LayerNorm,
  Linear,
  MultiHeadAttention,
  RMSNorm,
} from '../../../src/core/nn/layers.ts';
import { CpuParameter, Module, type AnyModule } from '../../../src/core/nn/module.ts';
import { memoryStateDict } from '../../../src/core/weights/safetensors.ts';
import { hwc4Input, input, meta } from '../graph/values.ts';

/** Binds every parameter to a stand-in of its declared shape, so forward()
 *  can build its graph without real weights. */
function fakeLoad<M extends AnyModule>(m: M): M {
  for (const [, p] of m.namedParameters()) p.bind(pending(p.shape, 'reshape', []));
  return m;
}

/** name → declared shape, for comparing whole layers at once. */
const shapes = (m: AnyModule) =>
  Object.fromEntries([...m.namedParameters()].map(([n, p]) => [n, p.shape]));

describe('Linear', () => {
  it('stores the weight as [in, out] and the bias as [out]', () => {
    expect(shapes(new Linear(3, 4))).toEqual({ weight: meta([3, 4]), bias: meta([4]) });
  });

  it('can go without a bias', () => {
    expect(Object.keys(shapes(new Linear(3, 4, { bias: false })))).toEqual(['weight']);
  });

  it('maps [T, in] to [T, out]', () => {
    expect(fakeLoad(new Linear(3, 4)).forward(input([5, 3])).shape).toEqual(meta([5, 4]));
  });

  it('transposes a torch [out, in] weight at load', async () => {
    const sd = memoryStateDict({
      'fc.weight': { data: new Float32Array([1, 2, 3, 4, 5, 6]), shape: [2, 3] },
    });
    new Linear(3, 2).transformStateDict(sd, 'fc');
    const w = sd.tensors.get('fc.weight')!;
    expect(w.shape).toEqual([3, 2]);
    expect(await w.f32()).toEqual(new Float32Array([1, 4, 2, 5, 3, 6]));
    expect(w.byteLength).toBe(24); // still counts the download
  });

  it('rejects a weight already in [in, out] layout', () => {
    const sd = memoryStateDict({ 'fc.weight': { data: new Float32Array(6), shape: [3, 2] } });
    expect(() => new Linear(3, 2).transformStateDict(sd, 'fc')).toThrow(
      expect.objectContaining({ code: 'CHECKPOINT_MISMATCH' }),
    );
  });

  it('leaves a missing weight for the strict key check to report', () => {
    const sd = memoryStateDict({});
    new Linear(3, 2).transformStateDict(sd, 'fc');
    expect(sd.tensors.size).toBe(0);
  });

  it('uses the bare name at the root', () => {
    const sd = memoryStateDict({ weight: { data: new Float32Array(6), shape: [2, 3] } });
    new Linear(3, 2).transformStateDict(sd, '');
    expect(sd.tensors.get('weight')!.shape).toEqual([3, 2]);
  });
});

describe('Conv1d', () => {
  it('stores the weight tap-major as [kernelSize·in, out]', () => {
    expect(shapes(new Conv1d(4, 6, 3))).toEqual({ weight: meta([12, 6]), bias: meta([6]) });
  });

  it('maps [T, in] to [T_out, out]', () => {
    const conv = fakeLoad(new Conv1d(4, 6, 3, { padding: 'same' }));
    expect(conv.forward(input([10, 4])).shape).toEqual(meta([10, 6]));
  });

  it('reorders a torch [out, in, k] weight at load', async () => {
    const sd = memoryStateDict({
      'c.weight': {
        data: new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]),
        shape: [2, 3, 2],
      },
    });
    new Conv1d(3, 2, 2).transformStateDict(sd, 'c');
    const w = sd.tensors.get('c.weight')!;
    // row tap·in + i, column out
    expect(w.shape).toEqual([6, 2]);
    expect(await w.f32()).toEqual(new Float32Array([1, 7, 3, 9, 5, 11, 2, 8, 4, 10, 6, 12]));
  });

  it('rejects a weight in another layout', () => {
    const sd = memoryStateDict({ 'c.weight': { data: new Float32Array(4), shape: [2, 2, 1] } });
    expect(() => new Conv1d(1, 2, 2).transformStateDict(sd, 'c')).toThrow(
      expect.objectContaining({ code: 'CHECKPOINT_MISMATCH' }),
    );
  });
});

describe('Conv2d', () => {
  it('declares an f16 hwc4 weight that half() and float() leave alone', () => {
    const conv = new Conv2d(8, 16, { kernelSize: 3 });
    expect(conv.weight.shape).toEqual({
      elems: 16 * 72,
      dtype: 'f16',
      dims: [16, 8 * 3 * 3],
      layout: 'hwc4',
    });
    expect(conv.weight.pinned).toBe(true);
    expect(conv.bias!.shape).toEqual(meta([16], 'f16'));
  });

  it('sizes a depthwise weight per channel', () => {
    expect(new Conv2d(8, 8, { kernelSize: 3, groups: 8 }).weight.shape.dims).toEqual([8, 9]);
  });

  it('takes a non-square kernel', () => {
    expect(new Conv2d(4, 2, { kernelSize: [1, 3] }).weight.shape.dims).toEqual([2, 12]);
  });

  it('rejects groups that are neither 1 nor depthwise', () => {
    expect(() => new Conv2d(8, 16, { kernelSize: 3, groups: 8 })).toThrow(
      /groups=8 is neither 1 nor depthwise \(cIn=8, cOut=16\)/,
    );
  });

  it('packs f16 weights into 4×4 tiles at load', () => {
    // [cOut=3, cIn=2, 1×1]: one tile laid out [in][out], zero-padded
    const packed = new Conv2d(2, 3, { kernelSize: 1 }).weight.preprocess!(
      new Uint16Array([1, 2, 3, 4, 5, 6]),
    );
    expect(packed).toEqual(new Uint16Array([1, 3, 5, 0, 2, 4, 6, 0, 0, 0, 0, 0, 0, 0, 0, 0]));
  });

  it('packs depthwise weights by groups of 4 channels', () => {
    const packed = new Conv2d(5, 5, { kernelSize: 1, groups: 5 }).weight.preprocess!(
      new Uint16Array([1, 2, 3, 4, 5]),
    );
    expect(packed).toEqual(new Uint16Array([1, 2, 3, 4, 5, 0, 0, 0]));
  });

  it('refuses f32 weights at load', () => {
    const conv = new Conv2d(2, 2, { kernelSize: 1 });
    expect(() => conv.weight.preprocess!(new Float32Array(4))).toThrow(/must arrive as f16 bits/);
  });

  it('maps an hwc4 map to an hwc4 map', () => {
    const conv = fakeLoad(new Conv2d(8, 16, { kernelSize: 3, stride: 2, padding: 1 }));
    expect(conv.forward(hwc4Input(8, 32, 32)).shape).toEqual(hwc4Input(16, 16, 16).shape);
  });
});

describe('ConvTranspose2d', () => {
  it('declares an f16 [out·k·k, in] weight and an f32 bias', () => {
    const conv = new ConvTranspose2d(4, 3, { kernelSize: 2, stride: 2 });
    expect(shapes(conv)).toEqual({ weight: meta([12, 4], 'f16'), bias: meta([3]) });
    expect(conv.weight.pinned).toBe(true);
  });

  it('reorders f16 weights at load', () => {
    const p = new ConvTranspose2d(2, 3, { kernelSize: 1, stride: 1 }).weight;
    // [in=2, out=3] to [out, in]
    expect(p.preprocess!(new Uint16Array([1, 2, 3, 4, 5, 6]))).toEqual(
      new Uint16Array([1, 4, 2, 5, 3, 6]),
    );
  });

  it('scales each side by k', () => {
    const conv = fakeLoad(new ConvTranspose2d(4, 3, { kernelSize: 2, stride: 2 }));
    expect(conv.forward(input([4, 5, 6])).shape).toEqual(meta([3, 10, 12]));
  });
});

describe('BatchNorm2d', () => {
  /** Records every upload and hands back a stand-in of the slot's shape. */
  const recorder = () =>
    vi.fn((_data: Float32Array | Uint32Array | Uint16Array, shape: ValueMeta) =>
      pending(shape, 'reshape', []),
    );
  /** bf16 bits, which memoryStateDict decodes like a file tensor. */
  const bf16 = (values: number[]) =>
    Uint16Array.from(values, (v) => new Uint32Array(new Float32Array([v]).buffer)[0]! >>> 16);

  // Two 1×1 out channels over two in channels, eps = 1, worked by hand:
  //   scale   = gamma / sqrt(var + eps) = [4 / 2, 4 / 2] = [2, 2]
  //   weight' = weight · scale          = [[1, 2], [3, 4]] · 2 = [[2, 4], [6, 8]]
  //   bias'   = beta + (bias − mean) · scale = [0 + 0 · 2, 1 + 2 · 2] = [0, 5]
  const EPS = 1;
  const conv = (prefix = '') => ({
    [`${prefix}conv.weight`]: { data: bf16([1, 2, 3, 4]), shape: [2, 2, 1, 1] },
    [`${prefix}conv.bias`]: bf16([1, 2]),
  });
  const bn = (prefix = '') => ({
    [`${prefix}bn.weight`]: bf16([4, 4]),
    [`${prefix}bn.bias`]: bf16([0, 1]),
    [`${prefix}bn.running_mean`]: bf16([1, 0]),
    [`${prefix}bn.running_var`]: bf16([3, 3]),
  });
  // The folded weight, packed by Conv2d into one [in][out] hwc4 tile, as f16
  // bits: 2 → 0x4000, 4 → 0x4400, 6 → 0x4600, 8 → 0x4800, 5 → 0x4500.
  const FOLDED_WEIGHT = new Uint16Array(16);
  FOLDED_WEIGHT.set([0x4000, 0x4600, 0, 0, 0x4400, 0x4800]);
  const FOLDED_BIAS = new Uint16Array([0, 0x4500]);

  class ConvBn extends Module {
    conv = new Conv2d(2, 2, { kernelSize: 1, bias: true });
    bn = new BatchNorm2d(2, { eps: EPS });
  }

  it('folds into the conv declared right before it and loads only conv weights', async () => {
    const m = new ConvBn();
    const upload = recorder();
    await m.loadStateDict({ ...conv(), ...bn() }, { upload });
    expect(upload.mock.calls.map(([data]) => data)).toEqual([FOLDED_WEIGHT, FOLDED_BIAS]);
    expect(m.conv.weight.loaded).toBe(true);
    expect(m.conv.bias!.loaded).toBe(true);
  });

  it('finds its conv inside a nested block', async () => {
    class Net extends Module {
      block = new ConvBn();
    }
    const net = new Net();
    const upload = recorder();
    await net.loadStateDict({ ...conv('block.'), ...bn('block.') }, { upload });
    expect(upload.mock.calls[0]![0]).toEqual(FOLDED_WEIGHT);
    expect(net.block.conv.weight.loaded).toBe(true);
  });

  it('folds into a named conv with foldInto', async () => {
    class Apart extends Module {
      stem = new Conv2d(2, 2, { kernelSize: 1 });
      other = new Linear(2, 2);
      bn = new BatchNorm2d(2, { eps: EPS, foldInto: 'stem' });
    }
    const m = new Apart();
    const upload = recorder();
    await m.loadStateDict(
      {
        'stem.weight': { data: bf16([1, 2, 3, 4]), shape: [2, 2, 1, 1] },
        'other.weight': { data: new Float32Array(4), shape: [2, 2] },
        'other.bias': new Float32Array(2),
        ...bn(),
      },
      { upload },
    );
    expect(upload).toHaveBeenCalledWith(FOLDED_WEIGHT, m.stem.weight.shape);
    expect(m.stem.bias!.loaded).toBe(true);
  });

  it.each([
    {
      name: 'something other than a conv',
      Parent: class extends Module {
        fc = new Linear(2, 2);
        bn = new BatchNorm2d(2);
      },
    },
    {
      name: 'a conv with another channel count',
      Parent: class extends Module {
        conv = new Conv2d(2, 4, { kernelSize: 1 });
        bn = new BatchNorm2d(2);
      },
    },
    {
      name: 'a non-conv after a matching conv',
      Parent: class extends Module {
        conv = new Conv2d(2, 2, { kernelSize: 1 });
        fc = new Linear(2, 2);
        bn = new BatchNorm2d(2);
      },
    },
    {
      name: 'nothing',
      Parent: class extends Module {
        bn = new BatchNorm2d(2);
      },
    },
  ])('refuses to guess when right before it is $name', async ({ Parent }) => {
    await expect(new Parent().loadStateDict(bn(), { upload: recorder() })).rejects.toThrow(
      /cannot infer its conv — needs a Conv2d with 2 out-channels/,
    );
  });

  it('has no parameters and passes activations through', () => {
    const x = hwc4Input(2, 4, 4);
    expect([...new BatchNorm2d(2).namedParameters()]).toEqual([]);
    expect(new BatchNorm2d(2).forward(x)).toBe(x);
  });
});

describe('normalization layers', () => {
  it('LayerNorm has a weight, and a bias only when asked', () => {
    expect(Object.keys(shapes(new LayerNorm(8)))).toEqual(['weight']);
    expect(shapes(new LayerNorm(8, 1e-5, { bias: true }))).toEqual({
      weight: meta([8]),
      bias: meta([8]),
    });
  });

  it('GroupNorm has a weight and bias per channel, and needs one group', () => {
    expect(shapes(new GroupNorm(1, 8))).toEqual({ weight: meta([8]), bias: meta([8]) });
    expect(() => new GroupNorm(2, 8)).toThrow(/only numGroups = 1 is supported, got 2/);
  });

  it('RMSNorm has one scale per feature', () => {
    expect(shapes(new RMSNorm(8))).toEqual({ scale: meta([8]) });
  });

  it.each([
    ['LayerNorm', () => new LayerNorm(8, 1e-5, { bias: true })],
    ['GroupNorm', () => new GroupNorm(1, 8)],
    ['RMSNorm', () => new RMSNorm(8)],
  ] as const)('%s keeps the [T, N] shape', (_, make) => {
    expect(fakeLoad(make()).forward(input([5, 8])).shape).toEqual(meta([5, 8]));
  });
});

describe('Embedding', () => {
  it('stores a [vocab, dim] table on the GPU by default', () => {
    const e = new Embedding(100, 16);
    expect(shapes(e)).toEqual({ weight: meta([100, 16]) });
    expect(e.weight).not.toBeInstanceOf(CpuParameter);
  });

  it('looks up rows by token id', () => {
    const e = fakeLoad(new Embedding(100, 16));
    expect(e.forward([5, 0, 5]).shape).toEqual(meta([3, 16]));
    expect(e.forward(input([7])).shape).toEqual(meta([7, 16]));
  });

  describe('on the CPU', () => {
    // vocab 3, dim 2: row i holds [i, i + 0.5]
    const table = new Float32Array([0, 0.5, 1, 1.5, 2, 2.5]);
    const cpu = () => {
      const e = new Embedding(3, 2, { device: 'cpu' });
      expect(e.weight).toBeInstanceOf(CpuParameter);
      return e;
    };

    it('gathers rows from an f32 table', () => {
      const e = cpu();
      (e.weight as CpuParameter).bindData(table);
      expect(e.gather([2, 0, 2])).toEqual(new Float32Array([2, 2.5, 0, 0.5, 2, 2.5]));
    });

    it('expands rows from a bf16 table to f32', () => {
      const e = cpu();
      const bf16 = Uint16Array.from(
        table,
        (v) => new Uint32Array(new Float32Array([v]).buffer)[0]! >>> 16,
      );
      (e.weight as CpuParameter).bindData(bf16);
      expect(e.gather([1])).toEqual(new Float32Array([1, 1.5]));
    });

    it.each([3, -1, 0.5])('rejects token id %s', (id) => {
      const e = cpu();
      (e.weight as CpuParameter).bindData(table);
      expect(() => e.gather([id])).toThrow(/out of range \(vocab 3\)/);
    });

    it('cannot take token ids that live on the GPU', () => {
      expect(() => cpu().forward(input([2]))).toThrow(/GPU-resident ids need a gpu table/);
    });
  });

  it('gather() refuses a GPU table', () => {
    expect(() => new Embedding(3, 2).gather([0])).toThrow(/cpu-resident tables only/);
  });
});

describe('MultiHeadAttention', () => {
  const opts = { qHeads: 4, kvHeads: 2, headDim: 8, windowLeft: Infinity, windowRight: 0 };

  it('has no parameters without sinks', () => {
    expect(shapes(new MultiHeadAttention(opts))).toEqual({});
  });

  it('declares one f32 sink per query head, pinned', () => {
    const attn = new MultiHeadAttention({ ...opts, sinks: true });
    expect(shapes(attn)).toEqual({ sinks: meta([4]) });
    expect(attn.sinks!.pinned).toBe(true);
  });

  it('attends from q, k, v or from one packed qkv', () => {
    const attn = fakeLoad(new MultiHeadAttention({ ...opts, sinks: true }));
    const kv = input([5, 16]);
    expect(attn.forward(input([5, 32]), kv, kv).shape).toEqual(meta([5, 32]));
    expect(attn.forwardPacked(input([5, 64])).shape).toEqual(meta([5, 32]));
  });
});
