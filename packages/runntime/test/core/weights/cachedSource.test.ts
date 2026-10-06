import { describe, expect, it } from 'vitest';
import type { WeightCache } from '../../../src/core/weights/cache.ts';
import { cachedRangeSource } from '../../../src/core/weights/cachedSource.ts';
import { bufferSource, type RangeSource } from '../../../src/core/weights/safetensors.ts';

const file = Uint8Array.from({ length: 16 }, (_, i) => i);

function setup(fileId = 'model') {
  const entries = new Map<string, ArrayBuffer>();
  const cache: WeightCache = {
    get: async (key) => entries.get(key),
    async getOrCompute(key, compute) {
      const hit = entries.get(key);
      if (hit) return hit;
      const view = await compute();
      const buf = new Uint8Array(view.buffer, view.byteOffset, view.byteLength).slice().buffer;
      entries.set(key, buf);
      return buf;
    },
    deletePrefix: async () => {},
    clear: async () => entries.clear(),
  };
  let reads = 0;
  const inner: RangeSource = {
    read(begin, end) {
      reads++;
      return bufferSource(file).read(begin, end);
    },
  };
  return { source: cachedRangeSource(inner, cache, fileId), cache, entries, reads: () => reads };
}

describe('cachedRangeSource', () => {
  it('reads through on a miss and from the cache on a hit', async () => {
    const { source, reads } = setup();
    expect(await source.read(2, 6)).toEqual(new Uint8Array([2, 3, 4, 5]));
    expect(await source.read(2, 6)).toEqual(new Uint8Array([2, 3, 4, 5]));
    expect(reads()).toBe(1);
  });

  it('keys entries by file and byte range', async () => {
    // safetensors.ts evicts one file with deletePrefix(`st/${fileId}/`)
    const { source, entries } = setup('pf-v3');
    await source.read(0, 8);
    await source.read(8, 12);
    expect([...entries.keys()]).toEqual(['st/pf-v3/0-8', 'st/pf-v3/8-12']);
  });

  it('caches overlapping ranges separately', async () => {
    const { source, reads } = setup();
    expect(await source.read(0, 4)).toEqual(new Uint8Array([0, 1, 2, 3]));
    expect(await source.read(2, 4)).toEqual(new Uint8Array([2, 3]));
    expect(reads()).toBe(2);
  });

  it('keeps two files with different IDs apart', async () => {
    const a = setup('a');
    const b = cachedRangeSource(bufferSource(new Uint8Array([9, 9, 9, 9])), a.cache, 'b');
    await a.source.read(0, 4);
    expect(await b.read(0, 4)).toEqual(new Uint8Array([9, 9, 9, 9]));
  });

  it('caches an empty range', async () => {
    const { source, entries } = setup();
    expect(await source.read(4, 4)).toEqual(new Uint8Array(0));
    expect(entries.has('st/model/4-4')).toBe(true);
  });

  it('passes a read error through', async () => {
    const { source } = setup();
    await expect(source.read(10, 20)).rejects.toThrow(/past end 16/);
  });
});
