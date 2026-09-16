/** Per-root caches for the two objects a kernel dispatch would otherwise
 *  allocate fresh: the small dims uniform buffer and the bind group. A camera
 *  loop dispatches on the order of 150 kernels per frame, and allocating both
 *  per dispatch is enough GPU allocation churn to outrun GC on iOS WebKit.
 *
 *  Uniforms are keyed by content, meaning schema plus field values. Dims recur,
 *  so a fixed-shape vision model converges to a fixed set and steady-state
 *  frames allocate nothing.
 *
 *  Bind groups are keyed by the identity of their entries, meaning the layout
 *  plus which buffers fill the slots. The BufferPool recycles the same buffers
 *  frame over frame, so those combinations recur too.
 *
 *  Growth is bounded by the distinct dim values and buffer combinations
 *  actually seen. On the LLM path that is one small entry per distinct token
 *  count. */

import type { TgpuBindGroup, TgpuRoot } from 'typegpu';

interface RootCaches {
  uniforms: WeakMap<object, Map<string, object>>;
  bindGroups: WeakMap<object, Map<string, TgpuBindGroup>>;
  ids: WeakMap<object, number>;
  nextId: number;
}

const perRoot = new WeakMap<TgpuRoot, RootCaches>();

function caches(root: TgpuRoot): RootCaches {
  let c = perRoot.get(root);
  if (c === undefined) {
    c = { uniforms: new WeakMap(), bindGroups: new WeakMap(), ids: new WeakMap(), nextId: 1 };
    perRoot.set(root, c);
  }
  return c;
}

function idOf(c: RootCaches, o: object): number {
  let v = c.ids.get(o);
  if (v === undefined) {
    v = c.nextId++;
    c.ids.set(o, v);
  }
  return v;
}

export function cachedUniform(
  root: TgpuRoot,
  schema: object,
  values: number | Record<string, number>,
): object {
  const c = caches(root);
  let bySchema = c.uniforms.get(schema);
  if (bySchema === undefined) {
    bySchema = new Map();
    c.uniforms.set(schema, bySchema);
  }
  const key = typeof values === 'number' ? String(values) : JSON.stringify(values);
  let buf = bySchema.get(key);
  if (buf === undefined) {
    buf = (root.createBuffer as (s: object, v: unknown) => { $usage: (u: string) => object })(
      schema,
      values,
    ).$usage('uniform');
    bySchema.set(key, buf);
  }
  return buf;
}

export function cachedBindGroup(
  root: TgpuRoot,
  layout: object,
  entries: Record<string, object>,
): TgpuBindGroup {
  const c = caches(root);
  let byLayout = c.bindGroups.get(layout);
  if (byLayout === undefined) {
    byLayout = new Map();
    c.bindGroups.set(layout, byLayout);
  }
  const names = Object.keys(entries).sort();
  let key = '';
  for (const n of names) key += `${n}=${idOf(c, entries[n]!)},`;
  let bg = byLayout.get(key);
  if (bg === undefined) {
    bg = (root.createBindGroup as (l: object, e: object) => TgpuBindGroup)(layout, entries);
    byLayout.set(key, bg);
  }
  return bg;
}
