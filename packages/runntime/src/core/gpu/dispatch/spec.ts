/** One KernelSpec per op: decode the node's attrs, build a pipeline config,
 *  compile for (dtype, config), encode the handles. */

import type { TgpuRoot } from 'typegpu';
import type { F16Buffer, F32Buffer, KernelHandle, U32Buffer } from '../../kernels/common.ts';
import type { EagerDtype, GpuBufferRef, PendingOp, Value } from '../../graph/value.ts';
import type { DispatchExtras } from '../../graph/evalCore.ts';

export type SpecCfg = Readonly<Record<string, number | string | boolean>> | undefined;

export interface DispatchCtx {
  readonly root: TgpuRoot;
  readonly subgroupsOk: boolean;
  dummy(dtype: 'f32'): F32Buffer;
  dummy(dtype: 'f16'): F16Buffer;
  dummy(dtype: FloatDtype): F32Buffer | F16Buffer;
  scratch(key: string, elems: number): F32Buffer;
}

export interface EncodeArgs<Attrs, Cfg extends SpecCfg, Pipe> {
  readonly node: Value;
  readonly attrs: Attrs;
  readonly cfg: Cfg;
  readonly pipeline: Pipe;
  readonly inputs: readonly GpuBufferRef[];
  readonly out: GpuBufferRef;
  readonly extras: DispatchExtras | undefined;
  readonly ctx: DispatchCtx;
}

export interface KernelSpec<Attrs, Cfg extends SpecCfg, Pipe> {
  readonly memo?: false;
  attrs(pending: PendingOp): Attrs;
  cfg(node: Value, attrs: Attrs, ctx: DispatchCtx, extras: DispatchExtras | undefined): Cfg;
  pipeline(root: TgpuRoot, dtype: EagerDtype, cfg: Cfg): Pipe;
  encode(args: EncodeArgs<Attrs, Cfg, Pipe>): KernelHandle[];
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export type AnySpec = KernelSpec<any, any, any>;

export function defineSpec<Attrs, Cfg extends SpecCfg, Pipe>(
  spec: KernelSpec<Attrs, Cfg, Pipe>,
): KernelSpec<Attrs, Cfg, Pipe> {
  return spec;
}

export function cfgKey(dtype: EagerDtype, cfg: SpecCfg): string {
  if (cfg === undefined) return dtype;
  const keys = Object.keys(cfg).sort();
  let s = dtype;
  for (const k of keys) s += `:${k}=${String(cfg[k])}`;
  return s;
}

const pipelinesPerRoot = new WeakMap<TgpuRoot, WeakMap<AnySpec, Map<string, unknown>>>();

export function pipelineFor<A, C extends SpecCfg, P>(
  root: TgpuRoot,
  spec: KernelSpec<A, C, P>,
  dtype: EagerDtype,
  cfg: C,
): P {
  let bySpec = pipelinesPerRoot.get(root);
  if (bySpec === undefined) {
    bySpec = new WeakMap();
    pipelinesPerRoot.set(root, bySpec);
  }
  let byKey = bySpec.get(spec);
  if (byKey === undefined) {
    byKey = new Map();
    bySpec.set(spec, byKey);
  }
  const key = cfgKey(dtype, cfg);
  let p = byKey.get(key);
  if (p === undefined) {
    p = spec.pipeline(root, dtype, cfg);
    byKey.set(key, p);
  }
  // spec.pipeline is the map's only writer, so the entry is a P.
  return p as P;
}

// Check the buffer's element schema and narrow the GpuBufferRef union.

type ElemKey = 'f32' | 'f16' | 'u32';

function elemKeyOf(buf: GpuBufferRef): ElemKey {
  return (buf as { dataType: { elementType: { type: ElemKey } } }).dataType.elementType.type;
}

export function narrow(buf: GpuBufferRef, want: 'f32'): F32Buffer;
export function narrow(buf: GpuBufferRef, want: 'f16'): F16Buffer;
export function narrow(buf: GpuBufferRef, want: 'u32'): U32Buffer;
export function narrow(buf: GpuBufferRef, want: 'f32' | 'f16'): F32Buffer | F16Buffer;
export function narrow(buf: GpuBufferRef, want: ElemKey): GpuBufferRef {
  const have = elemKeyOf(buf);
  if (have !== want) {
    throw new Error(`dispatch: expected a ${want} buffer, got ${have}`);
  }
  return buf;
}

export type FloatDtype = 'f32' | 'f16';

export function floatDtype(dtype: EagerDtype): FloatDtype {
  if (dtype !== 'f32' && dtype !== 'f16') {
    throw new Error(`dispatch: '${dtype}' is not a float activation dtype`);
  }
  return dtype;
}

export function narrowFloat(buf: GpuBufferRef, dtype: EagerDtype): F32Buffer | F16Buffer {
  return narrow(buf, floatDtype(dtype));
}
