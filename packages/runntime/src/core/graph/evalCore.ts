/** The device-agnostic eval driver: walks the pending sub-DAG, dispatches each
 *  node once in dependency order, and submits one batch. */

import type { EagerDtype, GpuBufferRef, ValueMeta, Value } from './value.ts';

const isFloat = (dtype: EagerDtype): boolean => dtype === 'f32' || dtype === 'f16';

export interface DispatchExtras {
  addend?: GpuBufferRef;
  outBase?: number;
  outElems?: number;
}

export interface Executor {
  acquireOutput(shape: ValueMeta): GpuBufferRef;
  releaseOutput(buffer: GpuBufferRef, shape: ValueMeta): void;
  dispatch(
    node: Value,
    inputs: readonly GpuBufferRef[],
    out: GpuBufferRef,
    extras?: DispatchExtras,
  ): void;
  copyInto?(
    src: GpuBufferRef,
    dst: GpuBufferRef,
    srcShape: ValueMeta,
    dstShape: ValueMeta,
    dstBase: number,
  ): void;
  submit(): void;
}

export interface Releaser {
  releaseValues(values: readonly Value[]): void;
  dispose(): void;
}

export interface Readback {
  readback(buffer: GpuBufferRef, shape: ValueMeta): Promise<Float32Array>;
  readbackMany?(
    items: readonly { buffer: GpuBufferRef; shape: ValueMeta }[],
  ): Promise<Float32Array[]>;
}

function collectPending(targets: readonly Value[]): Value[] {
  const order: Value[] = [];
  const seen = new Set<Value>();
  const visit = (v: Value): void => {
    if (v.state === 'released') {
      throw new Error(`evalValues: '${v.pending?.op ?? 'upload'}': ${v.releasedReason}`);
    }
    if (v.state === 'materialized' || seen.has(v)) return;
    seen.add(v);
    for (const input of v.pending!.inputs) visit(input);
    order.push(v);
  };
  for (const t of targets) visit(t);
  return order;
}

const IN_PLACE_OPS = new Set(['writeRows']);

export function evalValues(targets: readonly Value[], ex: Executor): void {
  const order = collectPending(targets);
  const produced = new Set<Value>(order);
  const targetSet = new Set<Value>(targets);

  const lastUse = new Map<Value, number>();
  order.forEach((v, i) => {
    for (const input of new Set(v.pending!.inputs)) {
      if (produced.has(input)) lastUse.set(input, i);
    }
  });

  // A graph with no conv2d or concatChannels would pay the fusion passes'
  // bookkeeping every eval for nothing.
  const hasFusableOps = order.some(
    (v) =>
      v.pending!.op === 'concatChannels' ||
      v.pending!.op === 'conv2dHwc4' ||
      v.pending!.op === 'concatChannelsHwc4',
  );

  const pos = new Map<Value, number>();
  const uses = new Map<Value, number>();
  if (hasFusableOps) {
    order.forEach((v, i) => {
      pos.set(v, i);
      for (const input of new Set(v.pending!.inputs)) {
        if (produced.has(input)) uses.set(input, (uses.get(input) ?? 0) + 1);
      }
    });
  }

  const sameShape = (a: Value, b: Value) =>
    a.shape.elems === b.shape.elems &&
    a.shape.dtype === b.shape.dtype &&
    a.shape.layout === b.shape.layout &&
    String(a.shape.dims) === String(b.shape.dims);

  // ---- Pass 1: residual-add fusion. Where add(x, conv) is the conv's only
  // consumer, the conv's epilogue adds x and the add node aliases its buffer,
  // removing a full-tensor read/read/write per residual.
  const fusedAddend = new Map<Value, Value>(); // conv node → addend value
  const addAlias = new Map<Value, Value>(); // add node → conv it aliases
  const ownedElsewhere = new Set<Value>(); // buffer belongs to an alias/holder — never release via this node
  for (const v of hasFusableOps ? order : []) {
    if (v.pending!.op !== 'add' || !isFloat(v.shape.dtype)) continue;
    const [i0, i1] = v.pending!.inputs;
    if (!i0 || !i1 || i0 === i1) continue;
    if (!sameShape(v, i0) || !sameShape(v, i1)) continue;
    const fusable = (conv: Value, x: Value) =>
      produced.has(conv) &&
      conv.pending!.op === 'conv2dHwc4' &&
      uses.get(conv) === 1 &&
      !targetSet.has(conv) &&
      !fusedAddend.has(conv) &&
      (!produced.has(x) || pos.get(x)! < pos.get(conv)!);
    let conv: Value | undefined;
    let x: Value | undefined;
    if (fusable(i1, i0)) {
      conv = i1;
      x = i0;
    } else if (fusable(i0, i1)) {
      conv = i0;
      x = i1;
    }
    if (conv === undefined) continue;
    fusedAddend.set(conv, x!);
    addAlias.set(v, conv);
    ownedElsewhere.add(conv);
  }

  // ---- Pass 2: concat elision. When an input's only consumer is the concat,
  // its producer writes straight into the concat's buffer and the copy
  // disappears; nested concats chain offsets so a whole chain lands in one
  // buffer. Inputs with other consumers fall back to ex.copyInto.
  interface ConcatSlot {
    holder: Value; // root concat whose buffer is written
    base: number; // element offset within the holder's buffer
  }
  const concatInto = new Map<Value, ConcatSlot>(); // writer node → destination slot
  // The ops validate that layout never mixes inside one concat, so one set is
  // safe.
  const OFFSET_WRITERS = new Set([
    'sliceChannels',
    'reshape',
    'concatChannels',
    'conv2dHwc4',
    'maxPool2dHwc4',
    'upsample2dHwc4',
    'sliceChannelsHwc4',
    'concatChannelsHwc4',
  ]);
  if (ex.copyInto && hasFusableOps) {
    for (let i = order.length - 1; i >= 0; i--) {
      const c = order[i]!;
      const cop = c.pending!.op;
      if ((cop !== 'concatChannels' && cop !== 'concatChannelsHwc4') || !isFloat(c.shape.dtype))
        continue;
      const self = concatInto.get(c) ?? { holder: c, base: 0 };
      const [a, b] = c.pending!.inputs as [Value, Value];
      const offsets = [0, a.shape.elems];
      [a, b].forEach((inp, j) => {
        if (!produced.has(inp) || uses.get(inp) !== 1 || targetSet.has(inp)) return;
        // A fused add aliases its conv, so the conv is the actual writer.
        const writer = inp.pending!.op === 'add' ? addAlias.get(inp) : inp;
        if (writer === undefined || concatInto.has(writer)) return;
        if (!OFFSET_WRITERS.has(writer.pending!.op)) return;
        concatInto.set(writer, { holder: self.holder, base: self.base + offsets[j]! });
        ownedElsewhere.add(inp);
        if (writer !== inp)
          concatInto.set(inp, { holder: self.holder, base: self.base + offsets[j]! });
      });
    }
  }

  // Acquired by the first elided writer, always before the concat itself.
  const holderBuf = new Map<Value, GpuBufferRef>();
  const bufferFor = (holder: Value) => {
    let buf = holderBuf.get(holder);
    if (buf === undefined) {
      buf = ex.acquireOutput(holder.shape);
      holderBuf.set(holder, buf);
    }
    return buf;
  };

  order.forEach((v, i) => {
    const aliasedConv = addAlias.get(v);
    if (IN_PLACE_OPS.has(v.pending!.op)) {
      // In-place op: writes rows of its first input's caller-owned buffer.
      const out = v.pending!.inputs[0]!.buffer;
      const inputs = v.pending!.inputs.map((input) => input.buffer);
      ex.dispatch(v, inputs, out);
      v.setMaterialized(out);
    } else if (aliasedConv !== undefined) {
      // Fused add: the conv already wrote act(conv)+x — just take its buffer.
      v.setMaterialized(aliasedConv.buffer);
    } else if (
      (v.pending!.op === 'concatChannels' || v.pending!.op === 'concatChannelsHwc4') &&
      concatInto.has(v)
    ) {
      // Writers targeted the holder directly; copy only what was not elided.
      const slot = concatInto.get(v)!;
      const buf = bufferFor(slot.holder);
      const [a, b] = v.pending!.inputs as [Value, Value];
      [a, b].forEach((inp, j) => {
        if (!ownedElsewhere.has(inp)) {
          ex.copyInto!(
            inp.buffer,
            buf,
            inp.shape,
            slot.holder.shape,
            slot.base + (j === 0 ? 0 : a.shape.elems),
          );
        }
      });
      v.setMaterialized(buf);
    } else if (
      (v.pending!.op === 'concatChannels' || v.pending!.op === 'concatChannelsHwc4') &&
      ex.copyInto &&
      holderBuf.has(v)
    ) {
      // Root concat with at least one elided input.
      const buf = bufferFor(v);
      const [a, b] = v.pending!.inputs as [Value, Value];
      [a, b].forEach((inp, j) => {
        if (!ownedElsewhere.has(inp)) {
          ex.copyInto!(inp.buffer, buf, inp.shape, v.shape, j === 0 ? 0 : a.shape.elems);
        }
      });
      v.setMaterialized(buf);
    } else {
      const slot = concatInto.get(v);
      const out = slot !== undefined ? bufferFor(slot.holder) : ex.acquireOutput(v.shape);
      const inputs = v.pending!.inputs.map((input) => input.buffer);
      const addend = fusedAddend.get(v);
      const extras: DispatchExtras | undefined =
        addend !== undefined || slot !== undefined
          ? {
              addend: addend?.buffer,
              outBase: slot?.base ?? 0,
              outElems: slot?.holder.shape.elems,
            }
          : undefined;
      ex.dispatch(v, inputs, out, extras);
      v.setMaterialized(out);
      if (slot !== undefined) ownedElsewhere.add(v);
    }
    // Dedupe: mul(x, x) would otherwise release one buffer twice.
    // ownedElsewhere inputs belong to an alias or holder now.
    for (const input of new Set(v.pending!.inputs)) {
      if (
        produced.has(input) &&
        lastUse.get(input) === i &&
        !targetSet.has(input) &&
        !IN_PLACE_OPS.has(input.pending!.op) &&
        !ownedElsewhere.has(input)
      ) {
        ex.releaseOutput(input.buffer, input.shape);
        input.markReleased(); // any later use of this Value now fails loudly
      }
    }
  });
  ex.submit();
}

export async function toArray(value: Value, ex: Executor & Readback): Promise<Float32Array> {
  evalValues([value], ex);
  return ex.readback(value.buffer, value.shape);
}
