import type { TgpuRoot } from 'typegpu';
import { describe, expect, it, vi } from 'vitest';
import { pending, type ValueMeta } from '../../../src/core/graph/value.ts';
import { Conv1d, ConvTranspose1d, Linear } from '../../../src/core/nn/layers.ts';
import { gpuUpload, loadLazyStateDict } from '../../../src/core/nn/loadStateDict.ts';
import {
  Buffer,
  CpuParameter,
  Module,
  ModuleList,
  Parameter,
  type AnyModule,
} from '../../../src/core/nn/module.ts';
import {
  bufferSource,
  derive,
  fromSafetensors,
  memoryStateDict,
  quantTensor,
  type LazyStateDict,
  type LazyTensor,
  type MemoryStateDict,
} from '../../../src/core/weights/safetensors.ts';
import { f32ToF16Bits } from '../../../src/core/weights/convert.ts';
import { concatRows, scaleWeight } from '../../../src/core/weights/transforms.ts';
import { meta } from '../graph/values.ts';
import { safetensorsFile } from '../weights/checkpoint.ts';

/** Records every upload and hands back a stand-in of the slot's shape. */
const recorder = () =>
  vi.fn((_data: Float32Array | Uint32Array | Uint16Array, shape: ValueMeta) =>
    pending(shape, 'reshape', []),
  );

type Entries = Record<string, MemoryStateDict[string] | LazyTensor>;

/** An in-memory state dict that also takes ready LazyTensors, for derived
 *  entries memoryStateDict cannot express. */
function dict(entries: Entries): LazyStateDict {
  const sd = memoryStateDict({});
  for (const [name, e] of Object.entries(entries)) {
    sd.tensors.set(name, 'kind' in e ? e : memoryStateDict({ [name]: e }).tensors.get(name)!);
  }
  return sd;
}

/** The tensor `name` of `sd`. */
const float = (sd: LazyStateDict, name: string) => sd.tensors.get(name)!;

const load = (model: AnyModule, entries: Entries, upload = recorder()) =>
  loadLazyStateDict(model, dict(entries), { upload });

/** f16 bits of values exact in f16: 1 → 0x3c00, 2 → 0x4000, −1.5 → 0xbe00. */
const F16 = { 1: 0x3c00, 2: 0x4000, '-1.5': 0xbe00 } as const;
/** bf16 bits: the top half of the f32 pattern. */
const bf16 = (values: number[]) =>
  Uint16Array.from(values, (v) => new Uint32Array(new Float32Array([v]).buffer)[0]! >>> 16);

class Pair extends Module {
  a = new Parameter(meta([2]));
  b = new Parameter(meta([2]));
}

describe('transform hooks', () => {
  it('runs the root with an empty prefix, then every child with its path and parent', async () => {
    const calls: [string, AnyModule | undefined][] = [];
    class Leaf extends Module {
      override transformStateDict(_sd: LazyStateDict, prefix: string, parent?: AnyModule) {
        calls.push([prefix, parent]);
      }
    }
    class Block extends Module {
      leaf = new Leaf();
      override transformStateDict(_sd: LazyStateDict, prefix: string, parent?: AnyModule) {
        calls.push([prefix, parent]);
      }
    }
    class Net extends Module {
      block = new Block();
      override transformStateDict(_sd: LazyStateDict, prefix: string, parent?: AnyModule) {
        calls.push([prefix, parent]);
      }
    }
    const net = new Net();
    await load(net, {});
    expect(calls).toEqual([
      ['', undefined],
      ['block', net],
      ['block.leaf', net.block],
    ]);
  });

  it('lets a hook add the keys a parameter is bound from', async () => {
    class Split extends Module {
      w = new Parameter(meta([2]));
      override transformStateDict(sd: LazyStateDict) {
        const fused = float(sd, 'fused');
        sd.tensors.set(
          'w',
          derive([2], [fused], ([values]) => values.subarray(0, 2)),
        );
        sd.tensors.delete('fused');
      }
    }
    const m = new Split();
    const upload = recorder();
    await load(m, { fused: new Float32Array([1, 2, 3, 4]) }, upload);
    expect(upload.mock.calls[0]![0]).toEqual(new Float32Array([1, 2]));
    expect(m.w.loaded).toBe(true);
  });

  it('reads a derived tensor once and leaves it holding nothing', async () => {
    const compute = vi.fn(async () => new Float32Array([1, 2]));
    const w = derive([2], [], compute);
    class One extends Module {
      w = new Parameter(meta([2]));
    }
    await load(new One(), { w });
    expect(compute).toHaveBeenCalledOnce();
    // A second read computes again: the loader's copy was the only one.
    await w.f32();
    expect(compute).toHaveBeenCalledTimes(2);
  });
});

describe('key check', () => {
  const unexpected: Entries = {
    a: new Float32Array(2),
    b: new Float32Array(2),
    c: new Float32Array(2),
  };

  it.each<{ name: string; sd: Entries; strict?: boolean; message: RegExp }>([
    { name: 'missing', sd: { a: new Float32Array(2) }, message: /missing: b\./ },
    { name: 'unexpected, when strict', sd: unexpected, strict: true, message: /unexpected: c\./ },
    {
      name: 'both',
      sd: { a: new Float32Array(2), c: new Float32Array(2) },
      message: /missing: b\. unexpected: c\./,
    },
  ])('reports $name keys before uploading anything', async ({ sd, strict, message }) => {
    const upload = recorder();
    await expect(loadLazyStateDict(new Pair(), dict(sd), { upload, strict })).rejects.toMatchObject(
      { code: 'CHECKPOINT_MISMATCH', message },
    );
    expect(upload).not.toHaveBeenCalled();
  });

  it('warns about unexpected keys and binds the rest when not strict', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const m = new Pair();
    const upload = recorder();
    const progress: [number, number][] = [];
    await loadLazyStateDict(m, dict(unexpected), {
      upload,
      onProgress: (_, done, total) => progress.push([done, total]),
    });
    expect(warn).toHaveBeenCalledWith(
      'loadStateDict: state dict has tensors the model does not read: c.',
    );
    expect(upload).toHaveBeenCalledTimes(2);
    expect(m.a.loaded && m.b.loaded).toBe(true);
    // The unbound tensor's bytes are never read, so they are not in the total.
    expect(progress).toEqual([
      [8, 16],
      [16, 16],
    ]);
    warn.mockRestore();
  });

  it('lists eight names and counts the rest', async () => {
    const extra = Object.fromEntries(
      Array.from({ length: 10 }, (_, i) => [`x${i}`, new Float32Array(1)]),
    );
    await expect(
      loadLazyStateDict(
        new Pair(),
        dict({ a: new Float32Array(2), b: new Float32Array(2), ...extra }),
        { upload: recorder(), strict: true },
      ),
    ).rejects.toThrow(/unexpected: x0, x1, x2, x3, x4, x5, x6, x7 … \(\+2\)\./);
  });

  it('rejects a tensor whose element count differs from the slot', async () => {
    await expect(
      load(new Pair(), { a: new Float32Array(2), b: new Float32Array(3) }),
    ).rejects.toMatchObject({
      code: 'CHECKPOINT_MISMATCH',
      message: /'b' checkpoint shape \[3\] \(3 elems\) != model \[2\] \(2 elems\)/,
    });
  });
});

describe('binding', () => {
  it('uploads each parameter at its slot shape and binds the result', async () => {
    const m = new Pair();
    const upload = recorder();
    await load(m, { a: new Float32Array([1, 2]), b: new Float32Array([3, 4]) }, upload);
    expect(upload).toHaveBeenCalledTimes(2);
    expect(upload).toHaveBeenCalledWith(new Float32Array([1, 2]), meta([2]));
    expect(m.a.value).toBe(upload.mock.results[0]!.value);
    expect(m.b.value).toBe(upload.mock.results[1]!.value);
  });

  it('binds in state-dict order, derived keys last', async () => {
    class Three extends Module {
      a = new Parameter(meta([1]));
      b = new Parameter(meta([1]));
      c = new Parameter(meta([1]));
      override transformStateDict(sd: LazyStateDict) {
        sd.tensors.set(
          'a',
          derive([1], [], () => Promise.resolve(new Float32Array(1))),
        );
      }
    }
    const names: string[] = [];
    await loadLazyStateDict(
      new Three(),
      memoryStateDict({ c: new Float32Array(1), b: new Float32Array(1) }),
      { upload: recorder(), onProgress: (name) => names.push(name) },
    );
    expect(names).toEqual(['c', 'b', 'a']);
  });

  it('binds a tensor where the file holds its sources, wherever a hook put its key', async () => {
    class Net extends Module {
      first = new Parameter(meta([1]));
      moved = new Parameter(meta([1]));
      summed = new Parameter(meta([1]));
      replaced = new Parameter(meta([1]));
      last = new Parameter(meta([1]));
      made = new Parameter(meta([1]));
      override transformStateDict(sd: LazyStateDict) {
        const take = (name: string) => {
          const t = float(sd, name);
          sd.tensors.delete(name);
          return t;
        };
        const one = () => new Float32Array(1);
        // One from no file tensor, one replaced under its name by one from
        // no file tensor, one built from two others, and a rename.
        sd.tensors.set('made', derive([1], [], one));
        sd.tensors.set('replaced', (take('replaced'), derive([1], [], one)));
        sd.tensors.set('summed', derive([1], [take('b'), take('a')], one));
        sd.tensors.set('moved', take('old_name'));
      }
    }
    const names: string[] = [];
    const file = Object.fromEntries(
      ['first', 'old_name', 'a', 'replaced', 'b', 'last'].map((name) => [
        name,
        new Float32Array(1),
      ]),
    );
    await loadLazyStateDict(new Net(), memoryStateDict(file), {
      upload: recorder(),
      strict: true,
      onProgress: (name) => names.push(name),
    });
    expect(names).toEqual(['first', 'moved', 'summed', 'last', 'made', 'replaced']);
  });

  // A runner may rename or fuse tensors before the load, as privacy-filter
  // does for the official checkpoint: the file order must survive it.
  class Renamed extends Module {
    b = new Parameter(meta([1, 1]));
    c = new Parameter(meta([1, 1]));
    renamed = new Parameter(meta([1, 1]));
  }
  class Stacked extends Module {
    b = new Parameter(meta([1, 1]));
    qkv = new Parameter(meta([3, 1]));
  }
  it.each<{
    name: string;
    file: string[];
    model: AnyModule;
    prepare: (sd: LazyStateDict) => void;
    want: string[];
  }>([
    {
      name: 'renamed',
      file: ['a', 'b', 'c'],
      model: new Renamed(),
      prepare: (sd) => {
        const a = sd.tensors.get('a')!;
        sd.tensors.delete('a');
        sd.tensors.set('renamed', a);
      },
      want: ['renamed', 'b', 'c'],
    },
    {
      name: 'stacked from three',
      file: ['q', 'k', 'v', 'b'],
      model: new Stacked(),
      prepare: (sd) => {
        const parts = ['q', 'k', 'v'].map((name) => sd.tensors.get(name)!);
        for (const name of ['q', 'k', 'v']) sd.tensors.delete(name);
        sd.tensors.set('qkv', concatRows(parts));
      },
      want: ['qkv', 'b'],
    },
  ])('binds a tensor $name before the load where the file holds it', async (c) => {
    const sd = memoryStateDict(
      Object.fromEntries(
        c.file.map((name) => [name, { data: new Float32Array(1), shape: [1, 1] }]),
      ),
    );
    c.prepare(sd);
    const names: string[] = [];
    await loadLazyStateDict(c.model, sd, {
      upload: recorder(),
      strict: true,
      onProgress: (name) => names.push(name),
    });
    expect(names).toEqual(c.want);
  });

  it('binds a tensor where the largest of its sources is, not a small one far from it', async () => {
    class Folded extends Module {
      early = new Parameter(meta([1]));
      weight = new Parameter(meta([4]));
      late = new Parameter(meta([1]));
      override transformStateDict(sd: LazyStateDict) {
        const norm = float(sd, 'norm');
        sd.tensors.delete('norm');
        sd.tensors.set(
          'weight',
          derive([4], [norm, float(sd, 'weight')], ([, w]) => w),
        );
      }
    }
    const names: string[] = [];
    const file = {
      norm: new Float32Array(1),
      early: new Float32Array(1),
      weight: new Float32Array(4),
      late: new Float32Array(1),
    };
    await loadLazyStateDict(new Folded(), memoryStateDict(file), {
      upload: recorder(),
      onProgress: (name) => names.push(name),
    });
    expect(names).toEqual(['early', 'weight', 'late']);
  });

  it('binds two tensors with the same sources in state-dict order', async () => {
    class Folded extends Module {
      bias = new Parameter(meta([1]));
      weight = new Parameter(meta([1]));
      other = new Parameter(meta([1]));
      override transformStateDict(sd: LazyStateDict) {
        const weight = float(sd, 'weight');
        const norm = float(sd, 'norm');
        sd.tensors.delete('norm');
        sd.tensors.set(
          'weight',
          derive([1], [weight, norm], ([w]) => w),
        );
        sd.tensors.set(
          'bias',
          derive([1], [weight, norm], ([w]) => w, 0),
        );
      }
    }
    const names: string[] = [];
    const file = {
      weight: new Float32Array(1),
      norm: new Float32Array(1),
      other: new Float32Array(1),
    };
    await loadLazyStateDict(new Folded(), memoryStateDict(file), {
      upload: recorder(),
      onProgress: (name) => names.push(name),
    });
    expect(names).toEqual(['weight', 'bias', 'other']);
  });

  it('reports cumulative file bytes, with derived keys adding none', async () => {
    const progress: [number, number][] = [];
    await loadLazyStateDict(
      new Pair(),
      dict({
        a: new Float32Array(2),
        b: derive([2], [], () => Promise.resolve(new Float32Array(2))),
      }),
      { upload: recorder(), onProgress: (_, done, total) => progress.push([done, total]) },
    );
    expect(progress).toEqual([
      [8, 8],
      [8, 8],
    ]);
  });

  it('applies preprocess before upload', async () => {
    class Pre extends Module {
      w = new Parameter(meta([2]), (data) => data.slice().reverse());
    }
    const upload = recorder();
    await load(new Pre(), { w: new Float32Array([1, 2]) }, upload);
    expect(upload.mock.calls[0]![0]).toEqual(new Float32Array([2, 1]));
  });

  it('keeps a CpuParameter off the GPU: bf16 as raw bits, anything else as f32', async () => {
    class Tables extends Module {
      raw = new CpuParameter(meta([2]));
      wide = new CpuParameter(meta([2]));
    }
    const m = new Tables();
    const upload = recorder();
    await load(m, { raw: bf16([1, 2]), wide: new Float32Array([1, 2]) }, upload);
    expect(upload).not.toHaveBeenCalled();
    expect(m.raw.data).toEqual(bf16([1, 2]));
    expect(m.wide.data).toEqual(new Float32Array([1, 2]));
  });

  it('uploads buffers after the parameters, narrowing to f16 after half()', async () => {
    class WithBuffer extends Module {
      w = new Parameter(meta([1]));
      scale = new Buffer(new Float32Array([1, -1.5]), meta([2]));
    }
    const m = new WithBuffer().half();
    const upload = recorder();
    await load(m, { w: new Float32Array(1) }, upload);
    expect(upload.mock.calls.map(([, shape]) => shape.dtype)).toEqual(['f16', 'f16']);
    expect(upload.mock.calls[1]![0]).toEqual(new Uint16Array([F16[1], F16['-1.5']]));
    expect(m.scale.loaded).toBe(true);
  });
});

describe('a quantized tensor', () => {
  // A [2, 2] int8 weight in groups of 2: ints 1 2 | 3 4, scales 0.5 and 2.
  const ints = vi.fn(async () => new Int8Array([1, 2, 3, 4]));
  const scales = vi.fn(async () => new Float32Array([0.5, 2]));
  const quant = quantTensor({
    name: 'w',
    shape: [2, 2],
    bits: 8,
    groupSize: 2,
    byteLength: 8,
    sources: [],
    ints,
    scales,
  });
  const DEQUANTIZED = [0.5, 1, 6, 8];

  it.each([
    { name: 'f32', dtype: 'f32' as const, want: new Float32Array(DEQUANTIZED) },
    { name: 'f16', dtype: 'f16' as const, want: Uint16Array.from(DEQUANTIZED, f32ToF16Bits) },
  ])('loads into an $name parameter as its dequantized values', async ({ dtype, want }) => {
    class Table extends Module {
      weight = new Parameter(meta([2, 2], dtype));
    }
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const upload = recorder();
    await load(new Table(), { weight: quant }, upload);
    expect(upload.mock.calls).toEqual([[want, meta([2, 2], dtype)]]);
    // No quantized kernel reads a plain parameter, so there is nothing to say.
    expect(warn).not.toHaveBeenCalled();
    warn.mockRestore();
  });

  it('loads into a CpuParameter as its dequantized f32 values, off the GPU', async () => {
    class Table extends Module {
      weight = new CpuParameter(meta([2, 2]));
    }
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const upload = recorder();
    const table = new Table();
    await load(table, { weight: quant }, upload);
    expect(table.weight.data).toEqual(new Float32Array(DEQUANTIZED));
    expect(upload).not.toHaveBeenCalled();
    expect(warn).not.toHaveBeenCalled();
    warn.mockRestore();
  });

  it('is refused by a quantW parameter with CHECKPOINT_MISMATCH, before any read', async () => {
    class Packed extends Module {
      weight = new Parameter({ elems: 1, dtype: 'quantW', dims: [2, 2] });
    }
    ints.mockClear();
    const upload = recorder();
    await expect(load(new Packed(), { weight: quant }, upload)).rejects.toMatchObject({
      code: 'CHECKPOINT_MISMATCH',
      message:
        "loadStateDict: 'weight' is quantized (int8), but this quantW parameter takes packed " +
        'words, which only its layer makes',
    });
    expect(upload).not.toHaveBeenCalled();
    expect(ints).not.toHaveBeenCalled();
  });
});

describe('the warning about quantized weights loaded as float', () => {
  // int8 in groups of 4, all zeros.
  const named = (prefix: string, rows: number, cols: number) => ({
    [`${prefix}.weight`]: { data: new Int8Array(rows * cols), shape: [rows, cols] },
    [`${prefix}.weight_scale`]: {
      data: new Uint16Array((rows * cols) / 4),
      shape: [rows, cols / 4],
    },
  });
  const setting = { 'runntime.quant': JSON.stringify({ version: 1, scheme: 'symmetric' }) };
  const open = (tensors: Parameters<typeof safetensorsFile>[0]) =>
    fromSafetensors(bufferSource(safetensorsFile(tensors, setting)), { chunkBytes: 0 });
  const WARNING =
    'loadStateDict: quantized weights loaded as float, the quantized kernel of their layer ' +
    'cannot run their shape or group size: ';

  class Table extends Module {
    weight = new Parameter(meta([4, 4]));
  }
  class Net extends Module {
    // quantized: [out 4, in 8], in 8 % 8 == 0
    fc = new Linear(8, 4, { bias: false });
    // float, warned: [out 4, in 12], in 12 % 8 != 0
    odd = new Linear(12, 4, { bias: false });
    // float, silent: no quantized kernel reads a plain parameter
    table = new Table();
    // float, warned: Conv1d [out 4, in 3, kernel 4] as [4, 12], in 3 % 4 != 0
    conv = new Conv1d(3, 4, 4, { bias: false });
    // float, warned: ConvTranspose1d [in 4, out 2, kernel 2] as [4, 4], out 2 % 4 != 0
    up = new ConvTranspose1d(4, 2, 2, { stride: 2, bias: false });
  }
  class Holder extends Module {
    table = new Table();
  }
  class Fits extends Module {
    fc = new Linear(8, 4, { bias: false });
  }

  it('names every weight whose layer has a quantized kernel that cannot run it', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const sd = await open({
      ...named('fc', 4, 8),
      ...named('odd', 4, 12),
      ...named('table', 4, 4),
      ...named('conv', 4, 12),
      ...named('up', 4, 4),
    });
    const upload = recorder();
    await loadLazyStateDict(new Net(), sd, { upload });
    expect(warn.mock.calls).toEqual([[`${WARNING}odd.weight, conv.weight, up.weight`]]);
    expect(upload.mock.calls.map(([, shape]) => shape.dtype)).toEqual([
      'quantW',
      'f32',
      'f32',
      'f32',
      'f32',
      'f32',
    ]);
    warn.mockRestore();
  });

  it('lists every name, however many', async () => {
    class Many extends Module {
      layers = new ModuleList(Array.from({ length: 12 }, () => new Linear(12, 4, { bias: false })));
    }
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    let tensors: Parameters<typeof safetensorsFile>[0] = {};
    for (let i = 0; i < 12; i++) tensors = { ...tensors, ...named(`layers.${i}`, 4, 12) };
    await loadLazyStateDict(new Many(), await open(tensors), { upload: recorder() });
    const names = Array.from({ length: 12 }, (_, i) => `layers.${i}.weight`);
    expect(warn.mock.calls).toEqual([[WARNING + names.join(', ')]]);
    warn.mockRestore();
  });

  it("names a hook-stacked weight's tensors as the file names them", async () => {
    // q and k stack into one [8, 12] weight: in 12 % 8 != 0.
    class Fused extends Module {
      qk = new Linear(12, 8, { bias: false });
      override transformStateDict(sd: LazyStateDict): void {
        const take = (name: string) => {
          const t = sd.tensors.get(name)!;
          sd.tensors.delete(name);
          return t;
        };
        sd.tensors.set(
          'qk.weight',
          concatRows([scaleWeight(take('q.weight'), 0.5), take('k.weight')]),
        );
      }
    }
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    const sd = await open({ ...named('q', 4, 12), ...named('k', 4, 12) });
    await loadLazyStateDict(new Fused(), sd, { upload: recorder() });
    expect(warn.mock.calls).toEqual([[`${WARNING}q.weight, k.weight`]]);
    warn.mockRestore();
  });

  it('stays silent about a tensor no quantized kernel reads', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await loadLazyStateDict(new Holder(), await open(named('table', 4, 4)), {
      upload: recorder(),
    });
    expect(warn).not.toHaveBeenCalled();
    warn.mockRestore();
  });

  it('stays silent when every quantized weight loads quantized', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await loadLazyStateDict(new Fits(), await open(named('fc', 4, 8)), { upload: recorder() });
    expect(warn).not.toHaveBeenCalled();
    warn.mockRestore();
  });

  it('stays silent on a float checkpoint', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await load(new Fits(), { 'fc.weight': { data: new Float32Array(32), shape: [4, 8] } });
    expect(warn).not.toHaveBeenCalled();
    warn.mockRestore();
  });
});

describe('a quantized Linear', () => {
  // torch [out 4, in 8] int8 in groups of 4: ints o·8 + i, f16-exact scales
  // o + g/2, and an f32 bias.
  const file = safetensorsFile(
    {
      'fc.weight': { data: Int8Array.from({ length: 32 }, (_, i) => i), shape: [4, 8] },
      'fc.weight_scale': {
        data: Uint16Array.from([0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5], f32ToF16Bits),
        shape: [4, 2],
      },
      'fc.bias': { data: new Float32Array([1, 2, 3, 4]), shape: [4] },
    },
    {
      'runntime.quant': JSON.stringify({ version: 1, bits: 8, groupSize: 4, scheme: 'symmetric' }),
    },
  );
  class Net extends Module {
    fc = new Linear(8, 4);
  }

  it('uploads the words, the scales and the bias in file order', async () => {
    const sd = await fromSafetensors(bufferSource(file), { chunkBytes: 0 });
    const weight = sd.tensors.get('fc.weight')!;
    if (weight.kind !== 'quant') throw new Error('expected a quantized tensor');
    const ints = vi.spyOn(weight, 'ints');
    const upload = recorder();
    const progress: [string, number, number][] = [];
    const net = new Net();
    await loadLazyStateDict(net, sd, {
      upload,
      onProgress: (name, done, total) => progress.push([name, done, total]),
    });
    expect(upload.mock.calls.map(([data, shape]) => [data.constructor.name, shape])).toEqual([
      ['Uint32Array', { elems: 8, dtype: 'quantW', dims: [8, 4] }],
      ['Float32Array', meta([2, 4])],
      ['Float32Array', meta([4])],
    ]);
    expect(upload.mock.calls[1]![0]).toEqual(new Float32Array([0, 1, 2, 3, 0.5, 1.5, 2.5, 3.5]));
    expect(ints).toHaveBeenCalledTimes(1);
    // 32 int8 bytes + 16 scale bytes, then the 16-byte bias: the payload size
    expect(progress).toEqual([
      ['fc.weight', 48, 64],
      ['fc.weightScale', 48, 64],
      ['fc.bias', 64, 64],
    ]);
    expect(net.fc.weightScale!.loaded).toBe(true);
  });

  it('loads a float checkpoint into the same layer as before', async () => {
    const upload = recorder();
    const net = new Net();
    await load(
      net,
      {
        'fc.weight': { data: new Float32Array(32), shape: [4, 8] },
        'fc.bias': new Float32Array(4),
      },
      upload,
    );
    expect(upload.mock.calls.map(([, shape]) => shape)).toEqual([meta([8, 4]), meta([4])]);
    expect(net.fc.weightScale).toBeUndefined();
  });
});

describe('a quantW slot', () => {
  // int8 [4, 4]: 16 values in 4 words.
  class Packed extends Module {
    w = new Parameter({ elems: 4, dtype: 'quantW', dims: [4, 4] });
  }

  /** A file holding `words` as one U32 tensor `w` of `shape`. */
  const wordFile = (words: Uint32Array, shape: number[]) =>
    fromSafetensors(bufferSource(safetensorsFile({ w: { data: words, shape } })), {
      chunkBytes: 0,
    });

  it('takes pre-packed U32 words from a file, counted in words', async () => {
    const words = new Uint32Array([1, 2, 3, 4]);
    const upload = recorder();
    await loadLazyStateDict(new Packed(), await wordFile(words, [4]), { upload });
    expect(upload).toHaveBeenCalledWith(words, { elems: 4, dtype: 'quantW', dims: [4, 4] });
  });

  it('takes derived words counted in logical values', async () => {
    const words = new Uint32Array([1, 2, 3, 4]);
    const upload = recorder();
    await load(new Packed(), { w: derive([4, 4], [], () => Promise.resolve(words)) }, upload);
    expect(upload).toHaveBeenCalledWith(words, { elems: 4, dtype: 'quantW', dims: [4, 4] });
  });

  it.each([
    { name: 'U32 words of another count', sd: () => wordFile(new Uint32Array(5), [5]) },
    {
      name: 'derived words of another logical size',
      sd: async () => dict({ w: derive([4], [], () => Promise.resolve(new Uint32Array(4))) }),
    },
  ])('refuses $name with CHECKPOINT_MISMATCH', async ({ sd }) => {
    await expect(
      loadLazyStateDict(new Packed(), await sd(), { upload: recorder() }),
    ).rejects.toMatchObject({ code: 'CHECKPOINT_MISMATCH' });
  });
});

describe('materializing an f16 slot', () => {
  class Half extends Module {
    w = new Parameter(meta([2], 'f16'));
  }
  const bits = async (sd: LazyStateDict) => {
    const upload = recorder();
    await loadLazyStateDict(new Half(), sd, { upload });
    return upload.mock.calls[0]![0];
  };

  it('narrows f32 from a file', async () => {
    const file = safetensorsFile({ w: { data: new Float32Array([1, 2]), shape: [2] } });
    expect(await bits(await fromSafetensors(bufferSource(file)))).toEqual(
      new Uint16Array([F16[1], F16[2]]),
    );
  });

  it('converts bf16 bits', async () => {
    expect(await bits(dict({ w: bf16([1, -1.5]) }))).toEqual(
      new Uint16Array([F16[1], F16['-1.5']]),
    );
  });

  it.each([
    {
      name: 'halfwords straight through',
      data: new Uint16Array([7, 9]),
      want: new Uint16Array([7, 9]),
    },
    {
      name: 'f32 narrowed',
      data: new Float32Array([2, -1.5]),
      want: new Uint16Array([F16[2], F16['-1.5']]),
    },
  ])('takes a derived tensor with $name', async ({ data, want }) => {
    expect(await bits(dict({ w: derive([2], [], () => Promise.resolve(data)) }))).toEqual(want);
  });

  it('refuses packed u32 words', async () => {
    await expect(
      bits(dict({ w: derive([2], [], () => Promise.resolve(new Uint32Array(2))) })),
    ).rejects.toThrow(/'w' holds packed u32 words, not loadable as f16/);
  });

  it('warns once per tensor about values clamped to the f16 range', async () => {
    const warn = vi.spyOn(console, 'warn').mockImplementation(() => {});
    await bits(dict({ w: new Float32Array([1e5, -1e5]) }));
    expect(warn).toHaveBeenCalledWith("loadStateDict: 'w' clamped 2 values to f16 range");
    warn.mockRestore();
  });
});

describe('gpuUpload', () => {
  // Each check runs before any buffer is made, so the root is never touched.
  const upload = gpuUpload({} as unknown as TgpuRoot);
  it.each([
    {
      name: 'halfwords',
      data: new Uint16Array(2),
      dtype: 'f32',
      message: /halfword data needs an f16 shape, got 'f32'/,
    },
    {
      name: 'words',
      data: new Uint32Array(2),
      dtype: 'f32',
      message: /word data needs a quantW shape, got 'f32'/,
    },
    {
      name: 'f32',
      data: new Float32Array(2),
      dtype: 'quantW',
      message: /cannot fill a 'quantW' parameter from f32 data/,
    },
  ] as const)('refuses $name into a $dtype slot', ({ data, dtype, message }) => {
    expect(() => upload(data, meta([2], dtype))).toThrow(message);
  });
});
