/** A scope collects GPU resources while a factory builds a runner, so one
 *  dispose() frees them all — including when the build throws half-way. */

export interface Disposable {
  dispose(): void;
}

export interface ResourceScope {
  /** Returns the resource, so it wraps a call in place. */
  track<R extends Disposable>(resource: R): R;
  /** Last added first. A second call does nothing. */
  dispose(): void;
  readonly disposed: boolean;
}

export function createResourceScope(): ResourceScope {
  const tracked: Disposable[] = [];
  let disposed = false;
  return {
    track<R extends Disposable>(resource: R): R {
      tracked.push(resource);
      return resource;
    },
    // Last added first: a resource goes before the one it was built from.
    // splice() empties the list, so a second call finds nothing.
    dispose(): void {
      disposed = true;
      for (const r of tracked.splice(0).reverse()) r.dispose();
    },
    get disposed() {
      return disposed;
    },
  };
}
