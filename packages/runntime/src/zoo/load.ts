/** Loading helpers shared by every create<Task>() factory: one options
 *  shape, one way to open a weights file, one way to fetch a JSON asset. */

import {
  fromSafetensors,
  isRunntimeError,
  RunntimeError,
  type LazyStateDict,
  type RangeSource,
  type WeightCache,
} from '../core/index.ts';
import { asLoadError, isAbort, messageOf } from './errors.ts';

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
  /** Runs the model once on a blank input before returning, which compiles
   *  the kernels and fails the load with EXECUTION_FAILED when the GPU
   *  errors or returns unusable output. With false, the first call pays the
   *  compile time, and a GPU error is not caught: most tasks return zeros
   *  instead of throwing. Default: true. */
  warmUp?: boolean;
}

/** The LoadOptions a model constructor reads. */
export type ModelLoadOptions = Pick<LoadOptions, 'onProgress' | 'warmUp'>;

/** Throws LOAD_ABORTED when the signal is already aborted. */
export function throwIfAborted(signal?: AbortSignal): void {
  if (signal?.aborted) {
    throw new RunntimeError('LOAD_ABORTED', 'model load aborted', { cause: signal.reason });
  }
}

/** A file of a hosted model: its repo, a `runntime-*` one or Reverb-TTS. */
const HOSTED = /^https:\/\/huggingface\.co\/software-mansion\/(runntime-[^/]+|Reverb-TTS)\//;

/** Counts one download on the Hugging Face Hub when `path` is a file of a
 *  hosted model. The Hub counts requests for a repo's config.json, and the
 *  loaders read model.safetensors instead. Sends nothing for any other URL
 *  or a RangeSource. Not awaited, and a failed request is ignored. */
export function countDownloads(path: ModelPath): void {
  const hosted = typeof path === 'string' ? HOSTED.exec(path) : null;
  if (!hosted) return;
  const dir = hosted[1] === 'Reverb-TTS' ? 'model/safetensors/fp16/' : '';
  try {
    const counter = fetch(`${hosted[0]}resolve/main/${dir}config.json`, { method: 'HEAD' });
    void Promise.resolve(counter).catch(() => {});
  } catch {
    // A fetch that is missing or throws counts nothing.
  }
}

/** An onBytes for the file at `path`: counts its download on the first chunk
 *  fetched from the network, then passes every chunk to `onBytes`. A file
 *  read from a cache fetches no chunk, so it counts nothing. */
export function countOnFirstBytes(
  path: ModelPath,
  onBytes?: (chunkBytes: number) => void,
): (chunkBytes: number) => void {
  let counted = false;
  return (chunkBytes) => {
    if (!counted) countDownloads(path);
    counted = true;
    onBytes?.(chunkBytes);
  };
}

/** Opens a safetensors file and reads its header. Tensor bytes download
 *  later, in loadStateDict. A hosted file counts one download when its first
 *  bytes come from the network. Fails with LOAD_FAILED, naming the file, or
 *  LOAD_ABORTED. */
export async function openWeights(
  modelPath: ModelPath,
  opts: LoadOptions = {},
): Promise<LazyStateDict> {
  throwIfAborted(opts.signal);
  try {
    return await fromSafetensors(modelPath, {
      onBytes: countOnFirstBytes(modelPath, opts.onBytes),
      cache: opts.cache,
      cacheId: opts.cacheId,
    });
  } catch (err) {
    if (isRunntimeError(err)) throw err;
    const where = typeof modelPath === 'string' ? modelPath : 'weights';
    throw new RunntimeError('LOAD_FAILED', `${where}: ${messageOf(err)}`, { cause: err });
  }
}

/** One onProgress for several files loading at once: `track(file)` gives a
 *  file's callback, and each call reports the bytes of every file together. */
export function combinedProgress(onProgress: LoadOptions['onProgress']) {
  const progress = new Map<string, [number, number]>();
  return (file: string) => (name: string, done: number, total: number) => {
    progress.set(file, [done, total]);
    let d = 0;
    let t = 0;
    for (const [fd, ft] of progress.values()) {
      d += fd;
      t += ft;
    }
    onProgress?.(name, d, t);
  };
}

/** Downloads and parses a JSON file (a tokenizer, a config). With a cache,
 *  the bytes are saved under `cacheId` (the file name by default) and reused
 *  on the next load. Fails with LOAD_FAILED, naming the URL, or
 *  LOAD_ABORTED. */
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
      if (opts.signal?.aborted || isAbort(err)) throw asLoadError(err, opts.signal);
      throw new RunntimeError('LOAD_FAILED', `${url}: ${messageOf(err)}`, { cause: err });
    }
    if (!res.ok) throw new RunntimeError('LOAD_FAILED', `${url}: HTTP ${res.status}`);
    return new Uint8Array(await res.arrayBuffer());
  };
  const bytes = opts.cache
    ? await opts.cache.getOrCompute(`json/${opts.cacheId ?? url.split('/').pop()!}`, download)
    : await download();
  try {
    return JSON.parse(new TextDecoder().decode(bytes)) as T;
  } catch (err) {
    throw new RunntimeError('LOAD_FAILED', `${url}: not valid JSON`, { cause: err });
  }
}
