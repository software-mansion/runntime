import { afterEach, describe, expect, it, vi } from 'vitest';
import type { WeightCache } from '../../../src/core/weights/cache.ts';
import { models } from '../../../src/zoo/models.ts';
import { cachedWeightsSource } from '../../../src/zoo/transformers/weights.ts';
import { memoryCache, safetensorsFile } from '../../core/weights/checkpoint.ts';
import { hubServer } from '../hubServer.ts';

// Node has no OPFS, so the cache the plugin opens is the one set here.
const opfs = vi.hoisted(() => ({ cache: undefined as WeightCache | undefined }));
vi.mock('../../../src/core/weights/opfsCache.ts', () => ({
  createOpfsCache: () => Promise.resolve(opfs.cache),
}));

describe('cachedWeightsSource', () => {
  const url = models.depthEstimation.DEPTHART.S.modelPath;
  const counter =
    'HEAD https://huggingface.co/software-mansion/runntime-depthart/resolve/main/config.json';
  const file = safetensorsFile({ w: { data: new Float32Array([1, 2]), shape: [2] } });
  afterEach(() => vi.unstubAllGlobals());

  it('counts a download on a cache miss and nothing on a cache hit', async () => {
    opfs.cache = memoryCache().cache;
    const { fetchFn, calls } = hubServer(file);
    vi.stubGlobal('fetch', fetchFn);

    await cachedWeightsSource(url, 'whole/depthart');
    expect(calls).toEqual([counter, `GET ${url}`]);

    calls.length = 0;
    const warm = await cachedWeightsSource(url, 'whole/depthart');
    expect(await warm.read(0, file.length)).toEqual(file);
    expect(calls).toEqual([]);
  });
});
