/** Loading helpers shared by every create<Task>() factory: one options
 *  shape, one way to open a weights file, one way to fetch a JSON asset. */

import {
  fromSafetensors,
  type LazyStateDict,
  type RangeSource,
  type WeightCache,
} from '../core/index.ts';

/** Where the weights come from: a URL, or any byte-range reader. */
export type ModelPath = string | RangeSource;

/** Options accepted by every create<Task>() factory. */
export interface LoadOptions {
  /** Saves fetched bytes. The next load reads them from here, no network. */
  cache?: WeightCache;
  /** Cache key. Defaults to the URL's file name; required for a RangeSource. */
  cacheId?: string;
  /** Called while tensors upload: tensor name, bytes done, bytes total. */
  onProgress?: (name: string, doneBytes: number, totalBytes: number) => void;
  /** Called per fetched network chunk with its size in bytes. */
  onBytes?: (chunkBytes: number) => void;
  /** Stops the load between steps. A step already running finishes first. */
  signal?: AbortSignal;
}

/** Throws when the signal is already aborted. */
export function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw new Error('model load aborted', { cause: signal.reason });
  }
}

/** Opens a safetensors file and reads its header. Tensor bytes download
 *  later, during loadStateDict. Every failure names the file it came from. */
export async function openWeights(
  modelPath: ModelPath,
  opts: LoadOptions = {},
): Promise<LazyStateDict> {
  throwIfAborted(opts.signal);
  try {
    return await fromSafetensors(modelPath, {
      onBytes: opts.onBytes,
      cache: opts.cache,
      cacheId: opts.cacheId,
    });
  } catch (err) {
    const where = typeof modelPath === 'string' ? modelPath : 'weights';
    throw new Error(`${where}: ${(err as Error).message}`, { cause: err });
  }
}

/** Downloads and parses a JSON file (a tokenizer, a config). With a cache,
 *  the bytes are saved under `cacheId` (the file name by default) and the
 *  next load reads them from there. Every failure names the URL it came
 *  from. */
export async function fetchJson<T>(
  url: string,
  opts: Pick<LoadOptions, 'cache' | 'cacheId' | 'signal'> = {},
): Promise<T> {
  throwIfAborted(opts.signal);
  const download = async (): Promise<Uint8Array> => {
    let res: Response;
    try {
      res = await fetch(url, { signal: opts.signal });
    } catch (err) {
      throw new Error(`${url}: ${(err as Error).message}`, { cause: err });
    }
    if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
    return new Uint8Array(await res.arrayBuffer());
  };
  const bytes = opts.cache
    ? await opts.cache.getOrCompute(`json/${opts.cacheId ?? url.split('/').pop()!}`, download)
    : await download();
  try {
    return JSON.parse(new TextDecoder().decode(bytes)) as T;
  } catch (err) {
    throw new Error(`${url}: not valid JSON`, { cause: err });
  }
}
