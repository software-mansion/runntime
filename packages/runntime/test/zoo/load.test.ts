import { afterEach, describe, expect, it, vi } from 'vitest';
import { bufferSource } from '../../src/core/index.ts';
import { combinedProgress, countDownloads, openWeights } from '../../src/zoo/load.ts';
import { models } from '../../src/zoo/models.ts';
import { memoryCache, safetensorsFile } from '../core/weights/checkpoint.ts';
import { hubServer } from './hubServer.ts';

const HUB = 'https://huggingface.co/software-mansion';

describe('combinedProgress', () => {
  it('reports the bytes of every file together', () => {
    const calls: [string, number, number][] = [];
    const track = combinedProgress((name, done, total) => calls.push([name, done, total]));
    const a = track('a');
    const b = track('b');
    a('a.0', 10, 100);
    b('b.0', 5, 20);
    a('a.1', 100, 100);
    b('b.1', 20, 20);
    expect(calls).toEqual([
      ['a.0', 10, 100],
      ['b.0', 15, 120],
      ['a.1', 105, 120],
      ['b.1', 120, 120],
    ]);
  });

  it('calls nothing without an onProgress', () => {
    expect(() => combinedProgress(undefined)('a')('a.0', 1, 1)).not.toThrow();
  });
});

describe('countDownloads', () => {
  const stubFetch = (result: () => Promise<unknown> = () => Promise.resolve()) => {
    const fetchFn = vi.fn((_url: string, _init?: RequestInit) => result());
    vi.stubGlobal('fetch', fetchFn);
    return fetchFn;
  };
  afterEach(() => vi.unstubAllGlobals());

  it("sends a HEAD for the config.json at the root of a model's repo", () => {
    const fetchFn = stubFetch();
    countDownloads(models.objectDetection.YOLO26.M.modelPath);
    expect(fetchFn.mock.calls).toEqual([
      [`${HUB}/runntime-yolo26/resolve/main/config.json`, { method: 'HEAD' }],
    ]);
  });

  it('counts every file of a repo', () => {
    const fetchFn = stubFetch();
    const { modelPath, visionPath, audioPath } = models.embedding.EMBEDDINGGEMMA_2.DEFAULT;
    for (const path of [modelPath, visionPath, audioPath]) countDownloads(path);
    expect(fetchFn.mock.calls.map(([url]) => url)).toEqual(
      Array(3).fill(`${HUB}/runntime-embeddinggemma-2/resolve/main/config.json`),
    );
  });

  it('counts Reverb by the config.json next to its f16 weights, and not its codec', () => {
    const fetchFn = stubFetch();
    const { modelPath, codecPath } = models.textToSpeech.REVERB.DEFAULT;
    countDownloads(modelPath);
    countDownloads(codecPath);
    expect(fetchFn.mock.calls.map(([url]) => url)).toEqual([
      `${HUB}/Reverb-TTS/resolve/main/model/safetensors/fp16/config.json`,
    ]);
  });

  it('sends nothing for another URL or a RangeSource', () => {
    const fetchFn = stubFetch();
    countDownloads('https://example.com/runntime-yolo26/resolve/main/n/f16/model.safetensors');
    countDownloads(
      'https://huggingface.co/someone/runntime-yolo26/resolve/main/n/f16/model.safetensors',
    );
    countDownloads(bufferSource(new Uint8Array()));
    expect(fetchFn).not.toHaveBeenCalled();
  });

  it('ignores a failed request', () => {
    const failed = Promise.reject(new TypeError('Failed to fetch'));
    const then = vi.spyOn(failed, 'then');
    stubFetch(() => failed);
    expect(() => countDownloads(models.depthEstimation.DEPTHART.S.modelPath)).not.toThrow();
    expect(then).toHaveBeenCalledWith(undefined, expect.any(Function));
  });

  it('ignores a fetch that throws', () => {
    vi.stubGlobal('fetch', () => {
      throw new Error('blocked');
    });
    expect(() => countDownloads(models.depthEstimation.DEPTHART.S.modelPath)).not.toThrow();
  });

  it('ignores a missing fetch', () => {
    vi.stubGlobal('fetch', undefined);
    expect(() => countDownloads(models.depthEstimation.DEPTHART.S.modelPath)).not.toThrow();
  });
});

describe('openWeights', () => {
  const url = models.objectDetection.YOLO26.M.modelPath;
  const counter = `HEAD ${HUB}/runntime-yolo26/resolve/main/config.json`;
  const file = safetensorsFile({ w: { data: new Float32Array([1, 2]), shape: [2] } });
  const isHead = (call: string) => call.startsWith('HEAD ');
  /** Opens `path` and reads its tensor, so the whole file is fetched. */
  const load = async (path: string, { cache } = memoryCache()) => {
    const sd = await openWeights(path, { cache });
    return sd.tensors.get('w')!.f32();
  };
  afterEach(() => vi.unstubAllGlobals());

  it('counts one download for a hosted file fetched from the network', async () => {
    const { fetchFn, calls } = hubServer(file);
    vi.stubGlobal('fetch', fetchFn);
    await load(url);
    expect(calls.filter(isHead)).toEqual([counter]);
    expect(calls.length).toBeGreaterThan(1);
  });

  it('sends no request for a file read from the cache', async () => {
    const { fetchFn, calls } = hubServer(file);
    vi.stubGlobal('fetch', fetchFn);
    const cached = memoryCache();
    await load(url, cached);
    calls.length = 0;
    expect(await load(url, cached)).toEqual(new Float32Array([1, 2]));
    expect(calls).toEqual([]);
  });

  it('counts nothing for another URL', async () => {
    const { fetchFn, calls } = hubServer(file);
    vi.stubGlobal('fetch', fetchFn);
    await load('https://example.com/model.safetensors');
    expect(calls.length).toBeGreaterThan(0);
    expect(calls.filter(isHead)).toEqual([]);
  });

  it('loads when the count request fails', async () => {
    const { fetchFn, calls } = hubServer(file, () => Promise.reject(new TypeError('offline')));
    vi.stubGlobal('fetch', fetchFn);
    expect(await load(url)).toEqual(new Float32Array([1, 2]));
    expect(calls).toContain(counter);
  });
});
