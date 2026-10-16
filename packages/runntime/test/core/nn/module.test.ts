import type { TgpuRoot } from 'typegpu';
import { describe, expect, it, vi } from 'vitest';
import { materialized, type GpuBufferRef, type Value } from '../../../src/core/graph/value.ts';
import {
  Buffer,
  CpuParameter,
  Module,
  ModuleList,
  Parameter,
} from '../../../src/core/nn/module.ts';
import { input, meta } from '../graph/values.ts';

const param = (n = 4) => new Parameter(meta([n]));

/** A GPU value whose buffer records destroy() calls. */
function uploaded(n = 4) {
  const destroy = vi.fn();
  // Only destroy() is ever called on a parameter's buffer outside a dispatch.
  const value = materialized(meta([n]), { destroy } as unknown as GpuBufferRef);
  return { value, destroy };
}

class Attention extends Module {
  q = param();
  k = param();
}

class Block extends Module {
  norm = param();
  attn = new Attention();
  scale = new Buffer(new Float32Array(4), meta([4]));
}

class Model extends Module {
  embed = param();
  layers = new ModuleList([new Block(), new Block()]);
}

describe('Module registration', () => {
  it('registers fields declared as class fields', () => {
    expect([...new Attention().namedParameters()].map(([n]) => n)).toEqual(['q', 'k']);
  });

  it('registers fields assigned in the constructor', () => {
    class Assigned extends Module {
      w: Parameter;
      constructor() {
        super();
        this.w = param();
      }
    }
    expect([...new Assigned().namedParameters()].map(([n]) => n)).toEqual(['w']);
  });

  it('skips fields starting with _ and anything that is not a slot or module', () => {
    class Mixed extends Module {
      _cache = param();
      size = 4;
      table = new Float32Array(4);
      w = param();
    }
    expect([...new Mixed().namedParameters()].map(([n]) => n)).toEqual(['w']);
  });
});

describe('Module walks', () => {
  it('names parameters by their dotted path, own ones before children', () => {
    expect([...new Model().namedParameters()].map(([n]) => n)).toEqual([
      'embed',
      'layers.0.norm',
      'layers.0.attn.q',
      'layers.0.attn.k',
      'layers.1.norm',
      'layers.1.attn.q',
      'layers.1.attn.k',
    ]);
  });

  it('writes the path into each parameter', () => {
    const model = new Model();
    const q = model.layers.items[1]!.attn.q;
    expect(new Map(model.namedParameters()).get('layers.1.attn.q')).toBe(q);
    expect(q.qualifiedName).toBe('layers.1.attn.q');
  });

  it('puts a prefix in front of every name', () => {
    expect([...new Attention().namedParameters('decoder.attn')].map(([n]) => n)).toEqual([
      'decoder.attn.q',
      'decoder.attn.k',
    ]);
  });

  it('names buffers the same way', () => {
    expect([...new Model().namedBuffers()].map(([n]) => n)).toEqual([
      'layers.0.scale',
      'layers.1.scale',
    ]);
  });

  it('lists submodules depth first, without the root', () => {
    expect([...new Model().namedModules()].map(([n]) => n)).toEqual([
      'layers',
      'layers.0',
      'layers.0.attn',
      'layers.1',
      'layers.1.attn',
    ]);
  });

  it('lists only direct children in declaration order', () => {
    expect([...new Block().namedChildren()].map(([n]) => n)).toEqual(['attn']);
  });

  it('walks an empty module', () => {
    class Empty extends Module {}
    expect([...new Empty().namedParameters()]).toEqual([]);
    expect([...new Empty().namedModules()]).toEqual([]);
  });
});

describe('Parameter', () => {
  it('throws on .value until bound, naming itself', () => {
    const model = new Attention();
    expect(new Map(model.namedParameters('enc')).get('enc.q')).toBe(model.q);
    expect(model.q.loaded).toBe(false);
    expect(() => model.q.value).toThrow(/Parameter enc.q not loaded/);
  });

  it('returns the bound value', () => {
    const p = param();
    const v = input([4]);
    p.bind(v);
    expect(p.loaded).toBe(true);
    expect(p.value).toBe(v);
  });

  it('frees its GPU buffer on dispose, and every later use of the value throws', () => {
    const p = param();
    const { value, destroy } = uploaded();
    p.bind(value);
    p.dispose();
    expect(destroy).toHaveBeenCalledOnce();
    expect(p.loaded).toBe(false);
    expect(value.state).toBe('released');
    expect(() => value.buffer).toThrow(/was disposed/);
  });
});

describe('CpuParameter', () => {
  it('has no GPU value, and its data throws until bound', () => {
    const p = new CpuParameter(meta([4]));
    expect(() => p.value).toThrow(/CPU-resident — read .data/);
    expect(() => p.data).toThrow(/not loaded/);
    p.bindData(new Float32Array(4));
    expect(p.data).toHaveLength(4);
  });
});

describe('Buffer', () => {
  it('keeps its CPU data after dispose, so it can upload again', () => {
    const data = new Float32Array([1, 2]);
    const b = new Buffer(data, meta([2]));
    const { value, destroy } = uploaded(2);
    b.bind(value);
    b.dispose();
    expect(destroy).toHaveBeenCalledOnce();
    expect(b.loaded).toBe(false);
    expect(b.data).toBe(data);
  });
});

describe('half() and float()', () => {
  class Mixed extends Module {
    w = param();
    pinnedF32 = new Parameter(meta([4]), undefined, true);
    pinnedF16 = new Parameter(meta([4], 'f16'), undefined, true);
    table = new CpuParameter(meta([4]));
    scale = new Buffer(new Float32Array(4), meta([4]));
  }
  const dtypes = (m: Module) =>
    Object.fromEntries(
      [...m.namedParameters(), ...m.namedBuffers()].map(([n, s]) => [n, s.shape.dtype]),
    );

  it('half() turns float slots to f16, except pinned ones', () => {
    expect(dtypes(new Mixed().half())).toEqual({
      w: 'f16',
      pinnedF32: 'f32',
      pinnedF16: 'f16',
      table: 'f16',
      scale: 'f16',
    });
  });

  it('float() undoes it, still skipping pinned ones', () => {
    expect(dtypes(new Mixed().half().float())).toEqual({
      w: 'f32',
      pinnedF32: 'f32',
      pinnedF16: 'f16',
      table: 'f32',
      scale: 'f32',
    });
  });

  it('keeps dims and element count', () => {
    const m = new Mixed().half();
    expect(m.w.shape).toEqual(meta([4], 'f16'));
  });

  it('refuses once weights are loaded', () => {
    const m = new Mixed();
    m.w.bind(input([4]));
    expect(() => m.half()).toThrow(/'w' is already loaded — retype the model before loadStateDict/);
  });

  it('retypes a loaded CPU table, which uploads at its dtype later', () => {
    const m = new Mixed();
    m.table.bindData(new Float32Array(4));
    expect(m.half().table.shape.dtype).toBe('f16');
  });

  it('refuses a model with quantized weights', () => {
    class Quant extends Module {
      w = new Parameter({ elems: 4, dtype: 'quantW', dims: [4, 4] });
    }
    expect(() => new Quant().half()).toThrow(/'w' is a quantized \(quantW\) weight/);
  });

  it('refuses a device without shader-f16', () => {
    const root = { enabledFeatures: new Set<string>() } as unknown as TgpuRoot;
    expect(() => new Mixed().half(root)).toThrow(
      expect.objectContaining({ code: 'UNSUPPORTED_DEVICE' }),
    );
  });
});

describe('dispose', () => {
  it('frees every parameter and buffer below, and is safe to call twice', () => {
    const m = new Block();
    const bound = [m.norm, m.attn.q, m.attn.k].map((p) => {
      const u = uploaded();
      p.bind(u.value);
      return u.destroy;
    });
    const scale = uploaded();
    m.scale.bind(scale.value);
    m.dispose();
    m.dispose();
    for (const destroy of [...bound, scale.destroy]) expect(destroy).toHaveBeenCalledOnce();
  });
});

describe('forward', () => {
  it('throws on a module that does not implement it', () => {
    expect(() => new Attention().forward(input([1]))).toThrow(
      /Attention: forward\(\) not implemented/,
    );
  });

  it('ModuleList chains its items in order', () => {
    const calls: string[] = [];
    class Step extends Module {
      constructor(readonly tag: string) {
        super();
      }
      override forward(x: Value): Value {
        calls.push(this.tag);
        return x;
      }
    }
    const x = input([2]);
    expect(new ModuleList([new Step('a'), new Step('b'), new Step('c')]).forward(x)).toBe(x);
    expect(calls).toEqual(['a', 'b', 'c']);
  });
});
