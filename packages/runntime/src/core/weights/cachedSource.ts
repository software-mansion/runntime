/** RangeSource wrapper persisting each read's bytes in a WeightCache, keyed by
 *  fileId + byte range. Tensor offsets are deterministic per file, so warm
 *  loads replay entirely from the cache; fileId must change when the file's
 *  content changes (the pf-vN filename convention). */
import type { RangeSource } from './safetensors.ts';
import type { WeightCache } from './cache.ts';

export function cachedRangeSource(
  inner: RangeSource,
  cache: WeightCache,
  fileId: string,
): RangeSource {
  return {
    async read(begin, end) {
      const buf = await cache.getOrCompute(`st/${fileId}/${begin}-${end}`, () =>
        inner.read(begin, end),
      );
      return new Uint8Array(buf);
    },
  };
}
