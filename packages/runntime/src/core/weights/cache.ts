/** Platform-neutral weight-cache contract: the loaders accept any
 *  implementation (web: OPFS via zoo's createOpfsCache; React Native:
 *  expo-file-system in apps/mobile). Values must be non-empty — zero-byte
 *  entries are treated as misses (crash artifacts of interrupted writes). */
export interface WeightCache {
  get(key: string): Promise<ArrayBuffer | undefined>;
  getOrCompute(key: string, compute: () => Promise<ArrayBufferView>): Promise<ArrayBuffer>;
  /** Drop every entry whose key starts with `prefix`. One file's entries
   *  share a key prefix (cachedRangeSource writes `st/<fileId>/<range>` per
   *  tensor), so evicting a bad download means dropping a prefix, not a
   *  single key. Callers recovering from one corrupt file MUST use this —
   *  clear() throws away every other model the app has cached. */
  deletePrefix(prefix: string): Promise<void>;
  /** Drop everything. For a user-facing "clear cache", not error recovery. */
  clear(): Promise<void>;
}
