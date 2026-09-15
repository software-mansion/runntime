/** API-surface stub of runntime/core. The types match what the real library
 *  exports; every function throws. Here so apps/docs builds and typechecks
 *  before the library itself lands in this repo. */

import type { TgpuRoot } from 'typegpu';

/** Reads a byte range out of a weights file. */
export interface RangeSource {
  read(begin: number, end: number): Promise<Uint8Array>;
}

/** Saves fetched weight bytes so the next load skips the network. */
export interface WeightCache {
  get(key: string): Promise<ArrayBuffer | undefined>;
  getOrCompute(key: string, compute: () => Promise<ArrayBufferView>): Promise<ArrayBuffer>;
  /** Drop every entry whose key starts with `prefix`. */
  deletePrefix(prefix: string): Promise<void>;
  /** Drop everything. For a user-facing "clear cache", not error recovery. */
  clear(): Promise<void>;
}

const notImplemented = (name: string): never => {
  throw new Error(
    `runntime: ${name}() is a stub in this repo. The library is not published yet.`,
  );
};

/** Reads a weights file already in memory. */
export function bufferSource(_buf: ArrayBuffer | Uint8Array): RangeSource {
  return notImplemented('bufferSource');
}

/** Opens an OPFS-backed weight cache, or undefined where OPFS is missing. */
export function createOpfsCache(_namespace: string): Promise<WeightCache | undefined> {
  return notImplemented('createOpfsCache');
}

/** Points the engine at a TypeGPU root. Every task awaits this first. */
export function initRunntime(_root: TgpuRoot): unknown {
  return notImplemented('initRunntime');
}
