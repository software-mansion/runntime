import { describe, expect, it, vi } from 'vitest';
import { evalValues, toArray, type Executor } from '../../../src/core/graph/evalCore.ts';
import { add, cat, conv2d, mul, sigmoid, tanh, writeRows } from '../../../src/core/graph/ops.ts';
import {
  hwc4Meta,
  materialized,
  type GpuBufferRef,
  type Value,
} from '../../../src/core/graph/value.ts';
import { materializedInput } from './values.ts';

/** An executor that hands out stub buffers and logs every call, one line each:
 *  `a → #0` for a dispatch, `release #0`, `copy #0 → #1 @144`, `submit`.
 *  Buffers it acquires are #0, #1, … in order; caller-owned ones go by the
 *  name of their Value. It throws on a release of a buffer it does not hold
 *  and on a dispatch that reads a released one. */
function recorder(names: Record<string, Value>, { copyInto = false } = {}) {
  const nodeName = new Map<Value, string>();
  const bufName = new Map<GpuBufferRef, string>();
  for (const [name, v] of Object.entries(names)) {
    nodeName.set(v, name);
    if (v.state === 'materialized') bufName.set(v.buffer, name);
  }
  const log: string[] = [];
  const held = new Set<GpuBufferRef>();
  const released = new Set<GpuBufferRef>();
  let acquired = 0;

  const ex: Executor = {
    acquireOutput() {
      const buf = {} as GpuBufferRef;
      bufName.set(buf, `#${acquired++}`);
      held.add(buf);
      return buf;
    },
    releaseOutput(buf) {
      if (!held.delete(buf)) throw new Error(`release of ${bufName.get(buf)}, which is not held`);
      released.add(buf);
      log.push(`release ${bufName.get(buf)}`);
    },
    dispatch(node, inputs, out, extras) {
      for (const buf of inputs) {
        if (released.has(buf))
          throw new Error(`${nodeName.get(node)} reads released ${bufName.get(buf)}`);
      }
      let line = `${nodeName.get(node)} → ${bufName.get(out)}`;
      if (extras?.addend !== undefined) line += ` +${bufName.get(extras.addend)}`;
      if (extras?.outElems !== undefined) line += ` @${extras.outBase}`;
      log.push(line);
    },
    submit() {
      log.push('submit');
    },
  };
  if (copyInto) {
    ex.copyInto = (src, dst, _srcShape, _dstShape, base) => {
      log.push(`copy ${bufName.get(src)} → ${bufName.get(dst)} @${base}`);
    };
  }
  return { ex, log, held: () => [...held].map((buf) => bufName.get(buf)) };
}

describe('evalValues', () => {
  it('dispatches each node once, after its inputs, and frees it after its last reader', () => {
    const x = materializedInput([2, 3]);
    const a = add(x, 1);
    const b = sigmoid(a);
    const c = tanh(a);
    const d = add(b, c);
    const { ex, log, held } = recorder({ x, a, b, c, d });
    evalValues([d], ex);
    expect(log).toEqual([
      'a → #0',
      'b → #1',
      'c → #2',
      'release #0',
      'd → #3',
      'release #1',
      'release #2',
      'submit',
    ]);
    expect(held()).toEqual(['#3']);
  });

  it('frees a buffer once when a node reads it twice', () => {
    const x = materializedInput([2, 3]);
    const a = add(x, 1);
    const b = mul(a, a);
    const { ex, log } = recorder({ a, b });
    evalValues([b], ex);
    expect(log).toEqual(['a → #0', 'b → #1', 'release #0', 'submit']);
  });

  it('keeps every target, even one a later node reads', () => {
    const a = add(materializedInput([2, 3]), 1);
    const b = sigmoid(a);
    const { ex, log, held } = recorder({ a, b });
    evalValues([a, b], ex);
    expect(log).toEqual(['a → #0', 'b → #1', 'submit']);
    expect(held()).toEqual(['#0', '#1']);
  });

  it('reads a value from an earlier eval without running or freeing it', () => {
    const a = add(materializedInput([2, 3]), 1);
    const b = sigmoid(a);
    const { ex, log, held } = recorder({ a, b });
    evalValues([a], ex);
    evalValues([b], ex);
    expect(log).toEqual(['a → #0', 'submit', 'b → #1', 'submit']);
    expect(held()).toEqual(['#0', '#1']);
  });

  it('only submits when everything is already materialized', () => {
    const x = materializedInput([2, 3]);
    const { ex, log } = recorder({ x });
    evalValues([x], ex);
    expect(log).toEqual(['submit']);
  });

  it('fails loudly on a value whose buffer an earlier eval freed', () => {
    const a = add(materializedInput([2, 3]), 1);
    const b = sigmoid(a);
    evalValues([b], recorder({ a, b }).ex);
    expect(() => evalValues([tanh(a)], recorder({}).ex)).toThrow(
      /'addScalar': buffer was recycled by eval/,
    );
  });

  it('writes writeRows into the caller-owned buffer and never frees it', () => {
    // A KV cache with room for 8 rows; 2 new rows go in at row 3
    const cache = materializedInput([8, 4]);
    const kNew = add(materializedInput([2, 4]), 1);
    const k = writeRows(cache, kNew, 3);
    const y = sigmoid(k);
    const { ex, log } = recorder({ cache, kNew, k, y });
    evalValues([y], ex);
    expect(log).toEqual(['kNew → #0', 'k → cache', 'release #0', 'y → #1', 'submit']);
    expect(k.buffer).toBe(cache.buffer);
  });
});

describe('toArray', () => {
  it('evaluates the value and reads back its buffer', async () => {
    const a = add(materializedInput([2, 3]), 1);
    const { ex, log } = recorder({ a });
    const data = new Float32Array(6);
    const readback = vi.fn(() => Promise.resolve(data));
    await expect(toArray(a, { ...ex, readback })).resolves.toBe(data);
    expect(log).toEqual(['a → #0', 'submit']);
    expect(readback).toHaveBeenCalledWith(a.buffer, a.shape);
  });
});

// The fusion passes only run on hwc4 conv and concat graphs.
const image = (c: number) => materialized(hwc4Meta(c, 6, 6), {} as GpuBufferRef);
/** Elements of one 4-channel 6×6 hwc4 map: where the second concat input starts. */
const MAP4 = hwc4Meta(4, 6, 6).elems;

/** A 1×1 conv of `x` to `cOut` channels, with a stub packed weight. */
function conv(x: Value, cOut: number): Value {
  const cIn = x.shape.dims![0]!;
  const weight = materialized(
    { elems: cOut * cIn, dtype: 'f16', dims: [cOut, cIn], layout: 'hwc4' },
    {} as GpuBufferRef,
  );
  return conv2d(x, weight, undefined, { kernelSize: 1, stride: 1, padding: 0, groups: 1 });
}

describe('residual-add fusion', () => {
  it('adds x in the conv epilogue and gives the add the conv’s buffer', () => {
    const x = image(8);
    const c = conv(x, 8);
    const y = add(x, c);
    const { ex, log, held } = recorder({ x, c, y });
    evalValues([y], ex);
    expect(log).toEqual(['c → #0 +x', 'submit']);
    expect(y.buffer).toBe(c.buffer);
    expect(held()).toEqual(['#0']);
  });

  it('fuses an addend computed before the conv', () => {
    const x = image(8);
    const x2 = tanh(x);
    const c = conv(x, 8);
    const y = add(x2, c);
    const { ex, log } = recorder({ x2, c, y });
    evalValues([y], ex);
    expect(log).toEqual(['x2 → #0', 'c → #1 +#0', 'release #0', 'submit']);
  });

  it.each([
    {
      name: 'the conv has another reader',
      build: (x: Value, c: Value) => ({ y: add(x, c), t: tanh(c) }),
      targets: ['y', 't'],
      log: ['c → #0', 'y → #1', 't → #2', 'release #0', 'submit'],
    },
    {
      name: 'the conv is a target',
      build: (x: Value, c: Value) => ({ y: add(x, c) }),
      targets: ['y', 'c'],
      log: ['c → #0', 'y → #1', 'submit'],
    },
    {
      name: 'the addend is computed after the conv',
      build: (x: Value, c: Value) => {
        const x2 = tanh(x);
        return { x2, y: add(c, x2) };
      },
      targets: ['y'],
      log: ['c → #0', 'x2 → #1', 'y → #2', 'release #0', 'release #1', 'submit'],
    },
  ])('does not fuse when $name', ({ build, targets, log }) => {
    const x = image(8);
    const c = conv(x, 8);
    const names: Record<string, Value> = { x, c, ...build(x, c) };
    const rec = recorder(names);
    evalValues(
      targets.map((t) => names[t]!),
      rec.ex,
    );
    expect(rec.log).toEqual(log);
  });
});

describe('concat elision', () => {
  it('has each input write straight into its slot of the concat buffer', () => {
    const x = image(8);
    const a = conv(x, 4);
    const b = conv(x, 4);
    const c = cat([a, b], 0);
    const { ex, log, held } = recorder({ a, b, c }, { copyInto: true });
    evalValues([c], ex);
    expect(log).toEqual(['a → #0 @0', `b → #0 @${MAP4}`, 'submit']);
    expect(held()).toEqual(['#0']);
  });

  it('lands a chain of concats in one buffer', () => {
    const x = image(8);
    const a = conv(x, 4);
    const b = conv(x, 4);
    const e = conv(x, 4);
    const d = cat([cat([a, b], 0), e], 0);
    const { ex, log, held } = recorder({ a, b, e, d }, { copyInto: true });
    evalValues([d], ex);
    expect(log).toEqual(['a → #0 @0', `b → #0 @${MAP4}`, `e → #0 @${2 * MAP4}`, 'submit']);
    expect(held()).toEqual(['#0']);
  });

  it('copies an input that has another reader', () => {
    const x = image(8);
    const a = conv(x, 4);
    const b = conv(x, 4);
    const c = cat([a, b], 0);
    const t = tanh(a);
    const { ex, log } = recorder({ a, b, c, t }, { copyInto: true });
    evalValues([c, t], ex);
    expect(log).toEqual([
      'a → #0',
      `b → #1 @${MAP4}`,
      'copy #0 → #1 @0',
      't → #2',
      'release #0',
      'submit',
    ]);
  });

  it('runs the concat as its own dispatch when the executor cannot copy', () => {
    const x = image(8);
    const a = conv(x, 4);
    const b = conv(x, 4);
    const c = cat([a, b], 0);
    const { ex, log } = recorder({ a, b, c });
    evalValues([c], ex);
    expect(log).toEqual(['a → #0', 'b → #1', 'c → #2', 'release #0', 'release #1', 'submit']);
  });
});
