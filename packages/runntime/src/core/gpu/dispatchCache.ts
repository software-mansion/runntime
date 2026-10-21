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
 *  frame over frame, so those combinations recur too. The bind group cache is
 *  weak in every entry, so a buffer the program drops takes its bind groups
 *  with it.
 *
 *  Uniform growth is bounded by the distinct dim values actually seen. On the
 *  LLM path that is one small entry per distinct token count. */

import type { TgpuBindGroup, TgpuRoot } from 'typegpu';

/** A node of the bind group cache. Each entry, in name order, picks the next
 *  node by its object; the node after the last entry holds the bind group. */
interface BindGroupNode {
  next: WeakMap<object, BindGroupNode>;
  bindGroup?: TgpuBindGroup;
}

interface RootCaches {
  uniforms: WeakMap<object, Map<string, object>>;
  /** Layout, then the sorted entry names, to the first node. */
  bindGroups: WeakMap<object, Map<string, BindGroupNode>>;
}

const perRoot = new WeakMap<TgpuRoot, RootCaches>();

function caches(root: TgpuRoot): RootCaches {
  let c = perRoot.get(root);
  if (c === undefined) {
    c = { uniforms: new WeakMap(), bindGroups: new WeakMap() };
    perRoot.set(root, c);
  }
  return c;
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
  const namesKey = names.join(',');
  let first = byLayout.get(namesKey);
  if (first === undefined) {
    first = { next: new WeakMap() };
    byLayout.set(namesKey, first);
  }
  let node = first;
  for (const n of names) {
    const entry = entries[n]!;
    let child = node.next.get(entry);
    if (child === undefined) {
      child = { next: new WeakMap() };
      node.next.set(entry, child);
    }
    node = child;
  }
  node.bindGroup ??= (root.createBindGroup as (l: object, e: object) => TgpuBindGroup)(
    layout,
    entries,
  );
  return node.bindGroup;
}
