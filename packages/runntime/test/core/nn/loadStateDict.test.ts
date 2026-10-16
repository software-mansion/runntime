import type { TgpuRoot } from 'typegpu';
import { describe, expect, it, vi } from 'vitest';
import { pending, type ValueMeta } from '../../../src/core/graph/value.ts';
import { gpuUpload, loadLazyStateDict } from '../../../src/core/nn/loadStateDict.ts';
import {
  Buffer,
  CpuParameter,
  Module,
  Parameter,
  type AnyModule,
} from '../../../src/core/nn/module.ts';
import {
  bufferSource,
  derivedTensor,
  fromSafetensors,
  memoryStateDict,
  type LazyStateDict,
  type LazyTensor,
  type MemoryStateDict,
} from '../../../src/core/weights/safetensors.ts';
import { meta } from '../graph/values.ts';

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
        const fused = sd.tensors.get('fused')!;
        sd.tensors.set(
          'w',
          derivedTensor([2], async () => (await fused.f32()).subarray(0, 2)),
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
});

describe('strict key check', () => {
  it.each<{ name: string; sd: Entries; message: RegExp }>([
    { name: 'missing', sd: { a: new Float32Array(2) }, message: /missing: b\./ },
    {
      name: 'unexpected',
      sd: { a: new Float32Array(2), b: new Float32Array(2), c: new Float32Array(2) },
      message: /unexpected: c\./,
    },
    {
      name: 'both',
      sd: { a: new Float32Array(2), c: new Float32Array(2) },
      message: /missing: b\. unexpected: c\./,
    },
  ])('reports $name keys before uploading anything', async ({ sd, message }) => {
    const upload = recorder();
    await expect(load(new Pair(), sd, upload)).rejects.toMatchObject({
      code: 'CHECKPOINT_MISMATCH',
      message,
    });
    expect(upload).not.toHaveBeenCalled();
  });

  it('lists eight names and counts the rest', async () => {
    const extra = Object.fromEntries(
      Array.from({ length: 10 }, (_, i) => [`x${i}`, new Float32Array(1)]),
    );
    await expect(
      load(new Pair(), { a: new Float32Array(2), b: new Float32Array(2), ...extra }),
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
          derivedTensor([1], () => Promise.resolve(new Float32Array(1))),
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

  it('reports cumulative file bytes, with derived keys adding none', async () => {
    const progress: [number, number][] = [];
    await loadLazyStateDict(
      new Pair(),
      dict({
        a: new Float32Array(2),
        b: derivedTensor([2], () => Promise.resolve(new Float32Array(2))),
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
    // A safetensors file built from the spec: 8-byte header length, JSON, payload.
    const header = new TextEncoder().encode(
      JSON.stringify({ w: { dtype: 'F32', shape: [2], data_offsets: [0, 8] } }),
    );
    const file = new Uint8Array(8 + header.length + 8);
    new DataView(file.buffer).setBigUint64(0, BigInt(header.length), true);
    file.set(header, 8);
    file.set(new Uint8Array(new Float32Array([1, 2]).buffer), 8 + header.length);
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
    expect(await bits(dict({ w: derivedTensor([2], () => Promise.resolve(data)) }))).toEqual(want);
  });

  it('refuses packed u32 words', async () => {
    await expect(
      bits(dict({ w: derivedTensor([2], () => Promise.resolve(new Uint32Array(2))) })),
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
