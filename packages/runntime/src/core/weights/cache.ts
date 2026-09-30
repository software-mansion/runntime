/** Keeps downloaded weights between visits. createOpfsCache() is the
 *  built-in one; any object with these methods works. An entry with no bytes
 *  counts as missing. */
export interface WeightCache {
  get(key: string): Promise<ArrayBuffer | undefined>;
  getOrCompute(key: string, compute: () => Promise<ArrayBufferView>): Promise<ArrayBuffer>;
  /** Drops every entry whose key starts with `prefix`. The entries of one
   *  file share a prefix, so this removes one bad download and keeps the
   *  other models. */
  deletePrefix(prefix: string): Promise<void>;
  /** Drop everything. For a user-facing "clear cache", not error recovery. */
  clear(): Promise<void>;
}
