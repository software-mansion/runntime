/** PyTorch-flavored modules. Module returns a Proxy from its constructor, so
 *  assigning a Parameter, Buffer or Module field registers it. */

import type { TgpuRoot } from 'typegpu';
import type { Executor, Readback } from '../graph/evalCore.ts';
import { maybeDefaultRoot, supportsF16 } from '../gpu/context.ts';
import { toArray } from '../gpu/eval.ts';
import { materialized, type EagerDtype, type ValueMeta, type Value } from '../graph/value.ts';
import {
  fromSafetensors,
  memoryStateDict,
  type LazyStateDict,
  type MemoryStateDict,
} from '../weights/safetensors.ts';
import { loadLazyStateDict, type LoadStateDictOpts } from './loadStateDict.ts';

/** A declared slot for a learned weight, filled by loadStateDict() matching
 *  its dotted path. `.value` throws until the model is loaded. */
export class Parameter {
  qualifiedName = '';
  private _value: Value | undefined;
  private _shape: ValueMeta;
  constructor(
    shape: ValueMeta,
    /** CPU-side transform applied before upload, once per load. */
    readonly preprocess?: (
      data: Float32Array | Uint32Array | Uint16Array,
    ) => Float32Array | Uint32Array | Uint16Array,
    /** Set when the kernels fix this value's dtype. half() and float() skip
     *  it in both directions. */
    readonly pinned = false,
  ) {
    this._shape = shape;
  }
  /** Changes only through retype(), and only before loading. */
  get shape(): ValueMeta {
    return this._shape;
  }
  get loaded(): boolean {
    return this._value !== undefined;
  }
  get value(): Value {
    if (!this._value) throw new Error(`Parameter ${this.qualifiedName || '(unnamed)'} not loaded`);
    return this._value;
  }
  /** @internal Swaps the storage dtype, keeping elems and dims. */
  retype(dtype: EagerDtype): void {
    this._shape = { ...this._shape, dtype };
  }
  /** @internal */
  bind(v: Value): void {
    this._value = v;
  }
  /** Frees the GPU buffer; every use of the old Value throws. */
  dispose(): void {
    const v = this._value;
    this._value = undefined;
    if (v?.state !== 'materialized') return;
    v.buffer.destroy();
    v.markReleased(`Parameter ${this.qualifiedName || '(unnamed)'} was disposed`);
  }
}

/** A Parameter that stays in CPU memory, for weights where forward() reads only
 *  a few rows. BF16 keeps its raw bits. */
export class CpuParameter extends Parameter {
  private _data: Float32Array | Uint16Array | undefined;
  override get value(): Value {
    throw new Error(
      `Parameter ${this.qualifiedName || '(unnamed)'} is CPU-resident — read .data, it has no GPU Value`,
    );
  }
  get data(): Float32Array | Uint16Array {
    if (!this._data) throw new Error(`Parameter ${this.qualifiedName || '(unnamed)'} not loaded`);
    return this._data;
  }
  /** @internal */
  bindData(data: Float32Array | Uint16Array): void {
    this._data = data;
  }
  /** Drops the CPU data. */
  override dispose(): void {
    this._data = undefined;
  }
}

/** A model-computed constant; never read from the checkpoint. */
export class Buffer {
  qualifiedName = '';
  private _value: Value | undefined;
  private _shape: ValueMeta;
  constructor(
    readonly data: Float32Array,
    shape: ValueMeta,
  ) {
    this._shape = shape;
  }
  get shape(): ValueMeta {
    return this._shape;
  }
  get loaded(): boolean {
    return this._value !== undefined;
  }
  get value(): Value {
    if (!this._value) throw new Error(`Buffer ${this.qualifiedName || '(unnamed)'} not uploaded`);
    return this._value;
  }
  /** @internal */
  retype(dtype: EagerDtype): void {
    this._shape = { ...this._shape, dtype };
  }
  /** @internal */
  bind(v: Value): void {
    this._value = v;
  }
  /** Frees the GPU buffer; the CPU `data` stays and re-uploads. */
  dispose(): void {
    const v = this._value;
    this._value = undefined;
    if (v?.state !== 'materialized') return;
    v.buffer.destroy();
    v.markReleased(`Buffer ${this.qualifiedName || '(unnamed)'} was disposed`);
  }
}

/** Swaps every float parameter and buffer from `from` to `to`, skipping
 *  `pinned` ones. Module-level because private members are unreachable through
 *  Module's Proxy. */
function retypeFloats(model: AnyModule, from: EagerDtype, to: EagerDtype): void {
  const slots = [...model.namedParameters(), ...model.namedBuffers()] as const;
  for (const [name, slot] of slots) {
    // A CpuParameter's dtype still names the width its layer uploads at.
    if (slot instanceof Parameter && slot.pinned) continue;
    if (slot instanceof CpuParameter) {
      if (slot.shape.dtype === from) slot.retype(to);
      continue;
    }
    if (slot.loaded) {
      throw new Error(
        `half()/float(): '${name}' is already loaded — retype the model before loadStateDict`,
      );
    }
    if (slot.shape.dtype === from) slot.retype(to);
  }
}

const REGISTRY = Symbol('eagerModuleRegistry');

interface Registry {
  params: Map<string, Parameter>;
  buffers: Map<string, Buffer>;
  modules: Map<string, AnyModule>;
}

/** A Module of any forward signature, for walking without running. Calling
 *  `.forward` on it is a compile error. */
export type AnyModule = Module<never[], unknown>;

export abstract class Module<In extends unknown[] = [Value], Out = Value> {
  /** @internal symbol-keyed so the Proxy trap skips it. */
  readonly [REGISTRY]: Registry = { params: new Map(), buffers: new Map(), modules: new Map() };

  constructor() {
    // Both [[Set]] and [[Define]], since ES2022 class fields use the latter.
    // Numeric-string keys pass, which is how ModuleList registers children.
    const record = (target: AnyModule, key: string | symbol, value: unknown): void => {
      if (typeof key !== 'string' || key.startsWith('_')) return;
      const reg = target[REGISTRY];
      if (value instanceof Parameter) reg.params.set(key, value);
      else if (value instanceof Buffer) reg.buffers.set(key, value);
      else if (value instanceof Module) reg.modules.set(key, value);
    };
    return new Proxy(this, {
      set(target, key, value, receiver) {
        record(target, key, value);
        return Reflect.set(target, key, value, receiver);
      },
      defineProperty(target, key, descriptor) {
        if ('value' in descriptor) record(target, key, descriptor.value);
        return Reflect.defineProperty(target, key, descriptor);
      },
    });
  }

  /** Subclasses declare shapes via `In` and `Out`; containers inherit this
   *  throwing stub. */
  forward(..._args: In): Out {
    throw new Error(`${this.constructor.name}: forward() not implemented`);
  }

  /** Stores every float parameter and buffer as f16; reducing kernels still
   *  accumulate in f32.
   *
   *  Call it before loadStateDict — afterwards the dtype is baked into every
   *  pipeline that read the buffer, so it throws. Also throws on quantW
   *  weights, whose kernels need f32. */
  half(root?: TgpuRoot): this {
    // Graph-only contexts have no device and never compile a shader; let the
    // first real dispatch fail there instead.
    const checkRoot = root ?? maybeDefaultRoot();
    if (checkRoot && !supportsF16(checkRoot)) {
      throw new Error(
        "half(): this device has no shader-f16 — check supportsF16() first, and request it with tgpu.init({ device: { optionalFeatures: ['shader-f16'] } })",
      );
    }
    for (const [name, p] of this.namedParameters()) {
      if (p.shape.dtype === 'quantW') {
        throw new Error(
          `half(): '${name}' is a quantized (quantW) weight — quantized models don't support f16 activations, leave this model at f32`,
        );
      }
    }
    retypeFloats(this, 'f32', 'f16');
    return this;
  }

  /** The inverse of half(). Needs no device feature. */
  float(): this {
    retypeFloats(this, 'f16', 'f32');
    return this;
  }

  /** Assigns each Parameter.qualifiedName as it walks. */
  *namedParameters(prefix = ''): Generator<readonly [string, Parameter]> {
    const reg = this[REGISTRY];
    for (const [name, p] of reg.params) {
      p.qualifiedName = prefix ? `${prefix}.${name}` : name;
      yield [p.qualifiedName, p] as const;
    }
    for (const [name, m] of reg.modules) {
      yield* m.namedParameters(prefix ? `${prefix}.${name}` : name);
    }
  }

  *namedBuffers(prefix = ''): Generator<readonly [string, Buffer]> {
    const reg = this[REGISTRY];
    for (const [name, b] of reg.buffers) {
      b.qualifiedName = prefix ? `${prefix}.${name}` : name;
      yield [b.qualifiedName, b] as const;
    }
    for (const [name, m] of reg.modules) {
      yield* m.namedBuffers(prefix ? `${prefix}.${name}` : name);
    }
  }

  /** Depth-first dotted walk over submodules, excluding this one. */
  *namedModules(prefix = ''): Generator<readonly [string, AnyModule]> {
    for (const [name, m] of this[REGISTRY].modules) {
      const path = prefix ? `${prefix}.${name}` : name;
      yield [path, m] as const;
      yield* m.namedModules(path);
    }
  }

  /** Direct submodules, in declaration order. */
  *namedChildren(): Generator<readonly [string, AnyModule]> {
    yield* this[REGISTRY].modules;
  }

  /** Adjusts the lazy state dict for this subtree before binding — fold BN,
   *  permute layouts, split fused tensors. Must be synchronous: produce
   *  derivedTensor entries, never fetch. */
  transformStateDict?(sd: LazyStateDict, prefix: string, parent?: AnyModule): void;

  /** Binds every Parameter, streaming one tensor at a time. Takes a lazy state
   *  dict, a safetensors URL, or a name→array record. */
  async loadStateDict(
    sd: LazyStateDict | string | MemoryStateDict,
    opts: LoadStateDictOpts = {},
  ): Promise<void> {
    const dict =
      typeof sd === 'string'
        ? await fromSafetensors(sd)
        : sd.tensors instanceof Map
          ? (sd as LazyStateDict)
          : memoryStateDict(sd as MemoryStateDict);
    await loadLazyStateDict(this, dict, opts);
  }

  /** Frees every parameter and buffer here and below. The shape survives, so
   *  loadStateDict refills it. Safe to call twice. */
  dispose(): void {
    for (const [, p] of this.namedParameters()) p.dispose();
    for (const [, b] of this.namedBuffers()) b.dispose();
  }
}

/** Submodules registered under their index, so the walk yields
 *  `block.3.attn.qkv.weight` like nn.ModuleList. */
export class ModuleList<M extends AnyModule> extends Module {
  constructor(readonly items: readonly M[]) {
    super();
    items.forEach((m, i) => {
      (this as Record<string, unknown>)[String(i)] = m;
    });
  }
  /** Chains items like nn.Sequential. Only valid when every item is
   *  Value→Value, which the compiler cannot check here. */
  override forward(x: Value): Value {
    let y = x;
    for (const m of this.items) y = (m as Module).forward(y);
    return y;
  }
}

/** Runs a plain x→y model and reads the result back. */
export async function run(
  model: Module<[Value], Value>,
  x: Value,
  ex?: Executor & Readback,
): Promise<Float32Array> {
  return toArray(model.forward(x), ex);
}

export { materialized };
