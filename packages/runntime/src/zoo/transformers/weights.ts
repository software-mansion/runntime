import {
  bufferSource,
  createOpfsCache,
  fromSafetensors,
  httpRangeSource,
  type RangeSource,
} from '../../core/index.ts';

/** The cache key of a weights URL: the whole path without the scheme, so
 *  same-named files of two sizes or precisions stay apart. */
export function cacheKey(url: string): string {
  return url.replace(/^https?:\/\//, '');
}

/** Loads a safetensors file too big to hold in memory: byte ranges are
 *  fetched per tensor and saved in OPFS under `cacheId`. */
export async function loadStreamedWeights(
  url: string,
  cacheId: string,
): Promise<Awaited<ReturnType<typeof fromSafetensors>>> {
  const cache = await createOpfsCache('runntime-weights');
  return fromSafetensors(url, { cache, cacheId });
}

export async function cachedWeightsSource(url: string, key: string): Promise<RangeSource> {
  const cache = await createOpfsCache('runntime-weights');
  if (!cache) return httpRangeSource(url);
  const fetchAll = async () => {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`runntime backend: fetching weights failed (${res.status})`);
    return new Uint8Array(await res.arrayBuffer());
  };
  const fromCache = async () => bufferSource(await cache.getOrCompute(key, fetchAll));
  let source = await fromCache();
  try {
    await fromSafetensors(source); // header parse only
  } catch {
    // A truncated cache entry (a write interrupted mid-download) fails the
    // header parse. Drop THIS entry — not the whole store, which holds every
    // other model — and reload from the network.
    await cache.deletePrefix(key);
    source = await fromCache();
    try {
      await fromSafetensors(source);
    } catch (err) {
      // Still unparseable after a fresh download: the URL is not a
      // safetensors file. Say so here rather than failing later on a read.
      throw new Error(`runntime backend: ${url} is not a safetensors file`, { cause: err });
    }
  }
  return source;
}

export async function loadCachedWeights(
  url: string,
  key: string,
): Promise<Awaited<ReturnType<typeof fromSafetensors>>> {
  return fromSafetensors(await cachedWeightsSource(url, key));
}
