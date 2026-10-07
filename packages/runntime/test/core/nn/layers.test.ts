import { describe, expect, it } from 'vitest';
import { pending } from '../../../src/core/graph/value.ts';
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
      'c.weight': { data: new Float32Array([1, 2, 3, 4]), shape: [2, 1, 2] },
    });
    new Conv1d(1, 2, 2).transformStateDict(sd, 'c');
    const w = sd.tensors.get('c.weight')!;
    expect(w.shape).toEqual([2, 2]);
    expect(await w.f32()).toEqual(new Float32Array([1, 3, 2, 4]));
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
    // [cOut=2, cIn=2, 1×1]: one tile laid out [in][out], zero-padded
    const packed = new Conv2d(2, 2, { kernelSize: 1 }).weight.preprocess!(
      new Uint16Array([1, 2, 3, 4]),
    );
    expect([...packed.slice(0, 8)]).toEqual([1, 3, 0, 0, 2, 4, 0, 0]);
    expect(packed).toHaveLength(16);
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

  it('reorders f16 weights at load and leaves other data alone', () => {
    const p = new ConvTranspose2d(2, 2, { kernelSize: 1, stride: 1 }).weight;
    // [in=2, out=2] to [out, in]
    expect(p.preprocess!(new Uint16Array([1, 2, 3, 4]))).toEqual(new Uint16Array([1, 3, 2, 4]));
    const f32 = new Float32Array(4);
    expect(p.preprocess!(f32)).toBe(f32);
  });

  it('scales each side by k', () => {
    const conv = fakeLoad(new ConvTranspose2d(4, 3, { kernelSize: 2, stride: 2 }));
    expect(conv.forward(input([4, 5, 6])).shape).toEqual(meta([3, 10, 12]));
  });
});

describe('BatchNorm2d', () => {
  const bn = (n: number) => ({
    [`bn.weight`]: new Float32Array(n).fill(1),
    [`bn.bias`]: new Float32Array(n),
    [`bn.running_mean`]: new Float32Array(n),
    [`bn.running_var`]: new Float32Array(n).fill(1),
  });

  class ConvBn extends Module {
    conv = new Conv2d(2, 2, { kernelSize: 1, bias: true });
    bn = new BatchNorm2d(2);
  }

  it('folds into the conv declared right before it, leaving only conv keys', () => {
    const m = new ConvBn();
    const sd = memoryStateDict({
      'conv.weight': { data: new Uint16Array(4), shape: [2, 2, 1, 1] },
      ...bn(2),
    });
    m.bn.transformStateDict(sd, 'bn', m);
    expect([...sd.tensors.keys()].sort()).toEqual(['conv.bias', 'conv.weight']);
  });

  it('finds its conv inside a nested block', () => {
    class Net extends Module {
      block = new ConvBn();
    }
    const net = new Net();
    const sd = memoryStateDict({
      'block.conv.weight': { data: new Uint16Array(4), shape: [2, 2, 1, 1] },
      ...Object.fromEntries(Object.entries(bn(2)).map(([k, v]) => [`block.${k}`, v])),
    });
    net.block.bn.transformStateDict(sd, 'block.bn', net.block);
    expect([...sd.tensors.keys()].sort()).toEqual(['block.conv.bias', 'block.conv.weight']);
  });

  it('folds into a named conv with foldInto', () => {
    class Apart extends Module {
      stem = new Conv2d(2, 2, { kernelSize: 1 });
      other = new Linear(2, 2);
      bn = new BatchNorm2d(2, { foldInto: 'stem' });
    }
    const m = new Apart();
    const sd = memoryStateDict({
      'stem.weight': { data: new Uint16Array(4), shape: [2, 2, 1, 1] },
      ...bn(2),
    });
    m.bn.transformStateDict(sd, 'bn', m);
    expect(sd.tensors.has('stem.bias')).toBe(true);
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
      name: 'nothing',
      Parent: class extends Module {
        bn = new BatchNorm2d(2);
      },
    },
  ])('refuses to guess when right before it is $name', ({ Parent }) => {
    const m = new Parent();
    expect(() => m.bn.transformStateDict(memoryStateDict(bn(2)), 'bn', m)).toThrow(
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
    expect(shapes(new Embedding(100, 16))).toEqual({ weight: meta([100, 16]) });
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
