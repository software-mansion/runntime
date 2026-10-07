import { afterEach, describe, expect, it, vi } from 'vitest';
import type { WeightCache } from '../../../src/core/weights/cache.ts';
import {
  bufferSource,
  chunkedSource,
  derivedTensor,
  fromSafetensors,
  httpRangeSource,
  memoryStateDict,
  parseSafetensorsHeader,
  type RangeSource,
  type SafeTensorInfo,
} from '../../../src/core/weights/safetensors.ts';

// Files are built here from the format spec, not with the code under test:
// an 8-byte little-endian header length, the JSON header, then the payload.
function fileBytes(header: unknown, payload = new Uint8Array(0)): Uint8Array {
  const json = new TextEncoder().encode(JSON.stringify(header));
  const out = new Uint8Array(8 + json.length + payload.length);
  new DataView(out.buffer).setBigUint64(0, BigInt(json.length), true);
  out.set(json, 8);
  out.set(payload, 8 + json.length);
  return out;
}

/** A file with just the 8-byte length prefix, for header-length errors. */
function lengthPrefix(headerLen: bigint): Uint8Array {
  const out = new Uint8Array(8);
  new DataView(out.buffer).setBigUint64(0, headerLen, true);
  return out;
}

type TensorSpec = Pick<SafeTensorInfo, 'dtype' | 'shape'> & { bytes: Uint8Array };

/** A valid file holding `tensors` back to back, in the given order. */
function tensorFile(tensors: Record<string, TensorSpec>, metadata?: Record<string, string>) {
  const header: Record<string, unknown> = metadata ? { __metadata__: metadata } : {};
  const payload = new Uint8Array(Object.values(tensors).reduce((n, t) => n + t.bytes.length, 0));
  let offset = 0;
  for (const [name, { dtype, shape, bytes }] of Object.entries(tensors)) {
    header[name] = { dtype, shape, data_offsets: [offset, offset + bytes.length] };
    payload.set(bytes, offset);
    offset += bytes.length;
  }
  return fileBytes(header, payload);
}

/** Where the payload starts in a file made by fileBytes. */
const payloadStart = (file: Uint8Array) =>
  8 + Number(new DataView(file.buffer).getBigUint64(0, true));

const f32Bytes = (values: number[]) => new Uint8Array(new Float32Array(values).buffer);
const u16Bytes = (values: number[]) => new Uint8Array(new Uint16Array(values).buffer);
const u32Bytes = (values: number[]) => new Uint8Array(new Uint32Array(values).buffer);
const sequence = (n: number) => Uint8Array.from({ length: n }, (_, i) => i);

/** Wraps a source and records every read, to count requests. */
function counted(inner: RangeSource) {
  const reads: [number, number][] = [];
  const source: RangeSource = {
    read(begin, end) {
      reads.push([begin, end]);
      return inner.read(begin, end);
    },
  };
  return { source, reads };
}

/** A source whose first `failures` reads reject. */
function flaky(inner: RangeSource, failures: number): RangeSource {
  let left = failures;
  return {
    read: (begin, end) =>
      left-- > 0 ? Promise.reject(new Error('network down')) : inner.read(begin, end),
  };
}

const failingSource: RangeSource = {
  read: () => Promise.reject(new Error('the network must not be used')),
};

function memoryCache() {
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
    async deletePrefix(prefix) {
      for (const key of [...entries.keys()]) if (key.startsWith(prefix)) entries.delete(key);
    },
    async clear() {
      entries.clear();
    },
  };
  return { cache, entries };
}

describe('parseSafetensorsHeader', () => {
  it('reads dtype, shape and absolute offsets', async () => {
    const aBytes = 2 * 3 * 4; // 6 f32 values, 4 bytes each
    const bBytes = 5 * 2; // 5 f16 values, 2 bytes each
    const file = tensorFile({
      a: { dtype: 'F32', shape: [2, 3], bytes: new Uint8Array(aBytes) },
      b: { dtype: 'F16', shape: [5], bytes: new Uint8Array(bBytes) },
    });
    // The header stores offsets from the start of the payload; the parser
    // turns them into positions in the whole file.
    const start = payloadStart(file);
    const { tensors } = await parseSafetensorsHeader(bufferSource(file));
    expect(tensors.get('a')).toEqual({
      dtype: 'F32',
      shape: [2, 3],
      begin: start,
      end: start + aBytes,
    });
    expect(tensors.get('b')).toEqual({
      dtype: 'F16',
      shape: [5],
      begin: start + aBytes, // right after a
      end: start + aBytes + bBytes,
    });
  });

  it('returns the metadata, or {} when there is none', async () => {
    const withMeta = tensorFile({}, { format: 'pt' });
    expect((await parseSafetensorsHeader(bufferSource(withMeta))).metadata).toEqual({
      format: 'pt',
    });
    expect((await parseSafetensorsHeader(bufferSource(tensorFile({})))).metadata).toEqual({});
  });

  it('orders tensors by where they sit in the file, not by header order', async () => {
    const file = fileBytes(
      {
        second: { dtype: 'U8', shape: [2], data_offsets: [2, 4] },
        first: { dtype: 'U8', shape: [2], data_offsets: [0, 2] },
      },
      new Uint8Array(4),
    );
    const { tensors } = await parseSafetensorsHeader(bufferSource(file));
    expect([...tensors.keys()]).toEqual(['first', 'second']);
  });

  it.each<{ name: string; tensors: Record<string, TensorSpec> }>([
    { name: 'an empty header', tensors: {} },
    {
      name: 'a scalar (shape [])',
      tensors: { s: { dtype: 'F32', shape: [], bytes: f32Bytes([1]) } },
    },
    {
      name: 'a zero-size tensor',
      tensors: { z: { dtype: 'F32', shape: [0, 4], bytes: new Uint8Array(0) } },
    },
  ])('accepts $name', async ({ tensors }) => {
    const index = await parseSafetensorsHeader(bufferSource(tensorFile(tensors)));
    expect(index.tensors.size).toBe(Object.keys(tensors).length);
  });

  it.each([
    { name: 'an entry that is not an object', entry: 5, error: /is not an object/ },
    {
      name: 'an unsupported dtype',
      entry: { dtype: 'F8_E4M3', shape: [4], data_offsets: [0, 4] },
      error: /unsupported dtype "F8_E4M3"/,
    },
    {
      name: 'a missing dtype',
      entry: { shape: [4], data_offsets: [0, 4] },
      error: /unsupported dtype/,
    },
    {
      name: 'a negative dim',
      entry: { dtype: 'U8', shape: [-1], data_offsets: [0, 0] },
      error: /malformed shape/,
    },
    {
      name: 'a fractional dim',
      entry: { dtype: 'U8', shape: [1.5], data_offsets: [0, 1] },
      error: /malformed shape/,
    },
    {
      name: 'a shape that is not an array',
      entry: { dtype: 'U8', shape: 4, data_offsets: [0, 4] },
      error: /malformed shape/,
    },
    {
      name: 'one data offset',
      entry: { dtype: 'U8', shape: [4], data_offsets: [4] },
      error: /malformed data_offsets/,
    },
    {
      name: 'a negative data offset',
      entry: { dtype: 'U8', shape: [4], data_offsets: [-4, 0] },
      error: /malformed data_offsets/,
    },
    {
      name: 'data offsets running backwards',
      entry: { dtype: 'U8', shape: [4], data_offsets: [8, 4] },
      error: /running backwards/,
    },
    {
      name: 'a span too short for the shape',
      entry: { dtype: 'F32', shape: [2], data_offsets: [0, 4] },
      error: /spans 4 bytes but shape \[2\] of F32 needs 8/,
    },
    {
      name: 'a span too long for the shape',
      entry: { dtype: 'F16', shape: [3], data_offsets: [0, 8] },
      error: /spans 8 bytes but shape \[3\] of F16 needs 6/,
    },
  ])('rejects $name', async ({ entry, error }) => {
    const file = fileBytes({ w: entry }, new Uint8Array(8));
    await expect(parseSafetensorsHeader(bufferSource(file))).rejects.toThrow(error);
  });

  it.each([
    { name: 'a zero header length', file: lengthPrefix(0n), error: /implausible header length 0/ },
    {
      name: 'a header length over the 100 MB cap',
      file: lengthPrefix(200n * 1024n * 1024n),
      error: /implausible header length/,
    },
    {
      name: 'a header length past 2^53',
      file: lengthPrefix(2n ** 63n),
      error: /implausible header length/,
    },
    {
      name: 'an HTML page instead of a file',
      file: new TextEncoder().encode('<!doctype html><html></html>'),
      error: /not a safetensors payload/,
    },
    { name: 'a file shorter than 8 bytes', file: new Uint8Array(4), error: /past end 4/ },
    {
      name: 'a header longer than the file',
      file: new Uint8Array([...lengthPrefix(100n), 0x7b, 0x7d]),
      error: /past end 10/,
    },
    {
      name: 'a header that is not JSON',
      file: (() => {
        const file = fileBytes('xxxxx');
        file.set(new TextEncoder().encode('{nope'), 8);
        return file;
      })(),
      error: /not valid JSON/,
    },
  ])('rejects $name', async ({ file, error }) => {
    await expect(parseSafetensorsHeader(bufferSource(file))).rejects.toThrow(error);
  });

  it.each([
    { name: 'null', header: null },
    { name: 'a number', header: 5 },
    { name: 'an array', header: [] },
  ])('rejects a header that is $name instead of an object', async ({ header }) => {
    await expect(parseSafetensorsHeader(bufferSource(fileBytes(header)))).rejects.toThrow(
      /safetensors: header is not a JSON object/,
    );
  });
});

describe('fromSafetensors', () => {
  const file = tensorFile({
    f32: { dtype: 'F32', shape: [3], bytes: f32Bytes([0.5, -2.25, 3]) },
    f16: { dtype: 'F16', shape: [3], bytes: u16Bytes([0x3e00, 0xb400, 0x3c00]) }, // 1.5, -0.25, 1
    bf16: { dtype: 'BF16', shape: [2], bytes: u16Bytes([0x3fc0, 0xbe80]) }, // 1.5, -0.25
    u32: { dtype: 'U32', shape: [2], bytes: u32Bytes([7, 0xffffffff]) },
    u8: { dtype: 'U8', shape: [1], bytes: new Uint8Array([9]) },
  });

  async function getTensor(name: string) {
    const sd = await fromSafetensors(bufferSource(file));
    return sd.tensors.get(name)!;
  }

  it('decodes F32, F16 and BF16 to f32', async () => {
    expect(await (await getTensor('f32')).f32()).toEqual(new Float32Array([0.5, -2.25, 3]));
    expect(await (await getTensor('f16')).f32()).toEqual(new Float32Array([1.5, -0.25, 1]));
    expect(await (await getTensor('bf16')).f32()).toEqual(new Float32Array([1.5, -0.25]));
  });

  it('gives F16 and BF16 bits unconverted', async () => {
    expect(await (await getTensor('f16')).u16()).toEqual(new Uint16Array([0x3e00, 0xb400, 0x3c00]));
    expect(await (await getTensor('bf16')).u16()).toEqual(new Uint16Array([0x3fc0, 0xbe80]));
  });

  it('packs F16 into u32 words, zero-padding an odd count', async () => {
    // little-endian: the first half sits in the low 16 bits
    expect(await (await getTensor('f16')).halfWords()).toEqual(
      new Uint32Array([0xb4003e00, 0x00003c00]),
    );
  });

  it('reads U32 words', async () => {
    expect(await (await getTensor('u32')).words()).toEqual(new Uint32Array([7, 0xffffffff]));
  });

  it.each([
    { name: 'f32() on U8', tensor: 'u8', call: 'f32' },
    { name: 'u16() on F32', tensor: 'f32', call: 'u16' },
    { name: 'halfWords() on BF16', tensor: 'bf16', call: 'halfWords' },
    { name: 'words() on F16', tensor: 'f16', call: 'words' },
  ] as const)('rejects $name with CHECKPOINT_MISMATCH', async ({ tensor: name, call }) => {
    await expect((await getTensor(name))[call]()).rejects.toMatchObject({
      name: 'RunntimeError',
      code: 'CHECKPOINT_MISMATCH',
    });
  });

  it('reports shape, dtype and byte sizes', async () => {
    const sd = await fromSafetensors(bufferSource(file));
    expect(sd.tensors.get('f16')).toMatchObject({ kind: 'file', dtype: 'F16', shape: [3] });
    expect(sd.tensors.get('f16')!.byteLength).toBe(3 * 2); // 3 values, 2 bytes each
    // f32: 3×4, f16: 3×2, bf16: 2×2, u32: 2×4, u8: 1×1
    expect(sd.totalBytes()).toBe(12 + 6 + 4 + 8 + 1);
  });

  it('reads a scalar and a zero-size tensor', async () => {
    const sd = await fromSafetensors(
      bufferSource(
        tensorFile({
          scalar: { dtype: 'F32', shape: [], bytes: f32Bytes([0.75]) },
          empty: { dtype: 'F32', shape: [0], bytes: new Uint8Array(0) },
        }),
      ),
    );
    expect(await sd.tensors.get('scalar')!.f32()).toEqual(new Float32Array([0.75]));
    expect(await sd.tensors.get('empty')!.f32()).toEqual(new Float32Array(0));
  });

  it('opens a file with no tensors', async () => {
    const sd = await fromSafetensors(bufferSource(tensorFile({})));
    expect(sd.tensors.size).toBe(0);
    expect(sd.totalBytes()).toBe(0);
  });

  it('fails the read, not the open, when the payload is cut short', async () => {
    const cut = file.slice(0, file.length - 4);
    const sd = await fromSafetensors(bufferSource(cut));
    await expect(sd.tensors.get('u8')!.f32()).rejects.toThrow(/past end/);
  });

  it('reads tensors that span several chunks', async () => {
    const values = Array.from({ length: 10 }, (_, i) => i + 0.5);
    const big = tensorFile({
      a: { dtype: 'U8', shape: [3], bytes: new Uint8Array([1, 2, 3]) },
      b: { dtype: 'F32', shape: [10], bytes: f32Bytes(values) },
    });
    const sd = await fromSafetensors(bufferSource(big), { chunkBytes: 8 });
    // 'a' is 3 bytes, so 'b' starts mid-chunk and crosses 5 chunk edges
    expect(await sd.tensors.get('b')!.f32()).toEqual(new Float32Array(values));
  });

  describe('with a cache', () => {
    it('loads a second time without touching the source', async () => {
      const { cache } = memoryCache();
      const cold = await fromSafetensors(bufferSource(file), { cache, cacheId: 'model' });
      await cold.tensors.get('f16')!.f32();
      await cold.tensors.get('bf16')!.u16();

      const warm = await fromSafetensors(failingSource, { cache, cacheId: 'model' });
      expect(await warm.tensors.get('f16')!.f32()).toEqual(new Float32Array([1.5, -0.25, 1]));
      expect(await warm.tensors.get('bf16')!.u16()).toEqual(new Uint16Array([0x3fc0, 0xbe80]));
    });

    it('keeps files with different cache IDs apart', async () => {
      const { cache } = memoryCache();
      const other = tensorFile({ f16: { dtype: 'F16', shape: [1], bytes: u16Bytes([0x3800]) } });
      const a = await fromSafetensors(bufferSource(file), { cache, cacheId: 'a' });
      const b = await fromSafetensors(bufferSource(other), { cache, cacheId: 'b' });
      expect(await a.tensors.get('f16')!.f32()).toEqual(new Float32Array([1.5, -0.25, 1]));
      expect(await b.tensors.get('f16')!.f32()).toEqual(new Float32Array([0.5]));
    });

    it('needs a cache ID when the source is not a URL', async () => {
      const { cache } = memoryCache();
      await expect(fromSafetensors(bufferSource(file), { cache })).rejects.toMatchObject({
        name: 'RunntimeError',
        code: 'INVALID_ARGUMENT',
      });
    });
  });

  describe('from a URL', () => {
    afterEach(() => {
      vi.unstubAllGlobals();
    });

    it('drops a cached HTML page and downloads the file again', async () => {
      const { cache, entries } = memoryCache();
      const html = new TextEncoder().encode('<!doctype html><html></html>');
      entries.set('st/model.safetensors/0-8', html.slice(0, 8).buffer);
      const { fetchFn, calls } = rangeServer(file);
      vi.stubGlobal('fetch', fetchFn);

      const sd = await fromSafetensors('https://example.com/model.safetensors', { cache });
      expect(await sd.tensors.get('f32')!.f32()).toEqual(new Float32Array([0.5, -2.25, 3]));
      expect(calls.length).toBeGreaterThan(0);
      expect(new Uint8Array(entries.get('st/model.safetensors/0-8')!)).toEqual(file.slice(0, 8));
    });

    it('uses the file name as the default cache ID', async () => {
      const { cache, entries } = memoryCache();
      vi.stubGlobal('fetch', rangeServer(file).fetchFn);
      await fromSafetensors('https://example.com/org/repo/weights-v2.safetensors', { cache });
      expect([...entries.keys()].every((k) => k.startsWith('st/weights-v2.safetensors/'))).toBe(
        true,
      );
    });
  });
});

/** A fake server that answers Range requests with 206, like a real CDN. */
function rangeServer(file: Uint8Array) {
  const calls: string[] = [];
  const fetchFn: typeof fetch = async (_input, init) => {
    const range = new Headers(init?.headers).get('Range') ?? '';
    calls.push(range);
    const match = /^bytes=(\d+)-(\d+)$/.exec(range);
    if (!match) return new Response(file.slice(), { status: 200 });
    return new Response(file.slice(Number(match[1]), Number(match[2]) + 1), { status: 206 });
  };
  return { fetchFn, calls };
}

describe('httpRangeSource', () => {
  const file = sequence(16);
  const url = 'https://example.com/model.safetensors';

  afterEach(() => {
    vi.useRealTimers();
  });

  /** Retries wait 0.5 s and 1 s; fake timers skip the wait. */
  async function settle<T>(promise: Promise<T>): Promise<T> {
    vi.useFakeTimers({ toFake: ['setTimeout'] });
    const result = promise.then(
      (value) => ({ value }),
      (error: unknown) => ({ error }),
    );
    await vi.runAllTimersAsync();
    const r = await result;
    if ('error' in r) throw r.error;
    return r.value;
  }

  it('asks for the exact byte range', async () => {
    const { fetchFn, calls } = rangeServer(file);
    const onBytes = vi.fn();
    const source = httpRangeSource(url, { fetchFn, onBytes });
    expect(await source.read(2, 6)).toEqual(new Uint8Array([2, 3, 4, 5]));
    expect(calls).toEqual(['bytes=2-5']); // HTTP ranges include the last byte
    expect(onBytes).toHaveBeenCalledWith(4);
  });

  it('downloads the whole file once when the server ignores Range', async () => {
    let fetches = 0;
    const fetchFn: typeof fetch = async () => {
      fetches++;
      return new Response(file.slice(), { status: 200 });
    };
    const onBytes = vi.fn();
    const source = httpRangeSource(url, { fetchFn, onBytes });
    const [a, b] = await Promise.all([source.read(0, 2), source.read(10, 12)]);
    expect(a).toEqual(new Uint8Array([0, 1]));
    expect(b).toEqual(new Uint8Array([10, 11]));
    expect(await source.read(4, 5)).toEqual(new Uint8Array([4]));
    expect(fetches).toBe(1);
    expect(onBytes).toHaveBeenCalledOnce();
    expect(onBytes).toHaveBeenCalledWith(16);
  });

  it('retries a short body', async () => {
    let fetches = 0;
    const fetchFn: typeof fetch = async () =>
      new Response(file.slice(0, ++fetches === 1 ? 2 : 4), { status: 206 });
    const source = httpRangeSource(url, { fetchFn });
    expect(await settle(source.read(0, 4))).toEqual(new Uint8Array([0, 1, 2, 3]));
    expect(fetches).toBe(2);
  });

  it.each([
    {
      name: 'a body that stays short',
      response: () => new Response(file.slice(0, 2), { status: 206 }),
      error: /short body: got 2, expected 4/,
    },
    {
      name: 'a 404',
      response: () => new Response('missing', { status: 404 }),
      error: /status 404/,
    },
    {
      name: 'an HTML fallback page',
      response: () =>
        new Response('<html></html>', { status: 200, headers: { 'content-type': 'text/html' } }),
      error: /HTML fallback page/,
    },
  ])('gives up after 3 tries on $name', async ({ response, error }) => {
    let fetches = 0;
    const fetchFn: typeof fetch = async () => {
      fetches++;
      return response();
    };
    const source = httpRangeSource(url, { fetchFn });
    await expect(settle(source.read(0, 4))).rejects.toThrow(error);
    expect(fetches).toBe(3);
  });
});

describe('chunkedSource', () => {
  const file = sequence(10);

  it.each([
    { name: 'inside one chunk', begin: 1, end: 3, inner: [[0, 4]] },
    {
      name: 'across two chunks',
      begin: 2,
      end: 7,
      inner: [
        [0, 4],
        [4, 8],
      ],
    },
    {
      name: 'across three chunks, the last one short',
      begin: 3,
      end: 10,
      inner: [
        [0, 4],
        [4, 8],
        [8, 10],
      ],
    },
    { name: 'exactly one chunk', begin: 4, end: 8, inner: [[4, 8]] },
    { name: 'a single byte', begin: 9, end: 10, inner: [[8, 10]] },
  ])('reads $name', async ({ begin, end, inner }) => {
    const { source, reads } = counted(bufferSource(file));
    const got = await chunkedSource(source, file.length, 4).read(begin, end);
    expect(got).toEqual(file.slice(begin, end));
    expect(reads).toEqual(inner);
  });

  it('returns a standalone copy', async () => {
    const got = await chunkedSource(bufferSource(file), file.length, 4).read(1, 3);
    expect(got.byteOffset).toBe(0);
    expect(got.buffer.byteLength).toBe(2);
  });

  it('fetches a chunk once for several reads', async () => {
    const { source, reads } = counted(bufferSource(file));
    const chunked = chunkedSource(source, file.length, 4);
    await Promise.all([chunked.read(0, 1), chunked.read(1, 2), chunked.read(2, 4)]);
    expect(reads).toEqual([[0, 4]]);
  });

  it('keeps the 4 most recent chunks', async () => {
    const big = sequence(20);
    const { source, reads } = counted(bufferSource(big));
    const chunked = chunkedSource(source, big.length, 4);
    for (const begin of [0, 4, 8, 12, 16]) await chunked.read(begin, begin + 1);
    await chunked.read(4, 5); // still kept
    expect(reads).toHaveLength(5);
    await chunked.read(0, 1); // evicted by the fifth chunk
    expect(reads).toHaveLength(6);
  });

  it('does not keep a failed fetch', async () => {
    const chunked = chunkedSource(flaky(bufferSource(file), 1), file.length, 4);
    await expect(chunked.read(0, 2)).rejects.toThrow(/network down/);
    expect(await chunked.read(0, 2)).toEqual(new Uint8Array([0, 1]));
  });

  it('rejects a read past the end', async () => {
    await expect(chunkedSource(bufferSource(file), file.length, 4).read(8, 11)).rejects.toThrow(
      /past end 10/,
    );
  });
});

describe('bufferSource', () => {
  it('returns a standalone copy', async () => {
    const bytes = sequence(8);
    const got = await bufferSource(bytes).read(2, 4);
    expect(got.byteOffset).toBe(0);
    got[0] = 99;
    expect(bytes[2]).toBe(2);
  });

  it('reads from a view with an offset', async () => {
    const view = new Uint8Array(sequence(8).buffer, 3, 4); // bytes 3..6
    expect(await bufferSource(view).read(0, 2)).toEqual(new Uint8Array([3, 4]));
  });

  it('takes an ArrayBuffer', async () => {
    expect(await bufferSource(sequence(4).buffer).read(1, 3)).toEqual(new Uint8Array([1, 2]));
  });

  it('reads an empty range', async () => {
    expect(await bufferSource(sequence(4)).read(2, 2)).toEqual(new Uint8Array(0));
  });

  it('rejects a read past the end', async () => {
    await expect(bufferSource(sequence(4)).read(2, 5)).rejects.toThrow(/past end 4/);
  });
});

describe('memoryStateDict', () => {
  it('gives bare arrays a flat shape', async () => {
    const sd = memoryStateDict({ w: new Float32Array([0.5, 1.5, -1]) });
    expect(sd.tensors.get('w')!.shape).toEqual([3]);
    expect(await sd.tensors.get('w')!.f32()).toEqual(new Float32Array([0.5, 1.5, -1]));
  });

  it('keeps the shape of the object form', async () => {
    const sd = memoryStateDict({ w: { data: new Float32Array(6), shape: [2, 3] } });
    expect(sd.tensors.get('w')!.shape).toEqual([2, 3]);
  });

  it('treats a Uint16Array as bf16 bits', async () => {
    const t = memoryStateDict({ w: new Uint16Array([0x3fc0, 0xbe80]) }).tensors.get('w')!;
    expect(t.dtype).toBe('BF16');
    expect(await t.f32()).toEqual(new Float32Array([1.5, -0.25]));
    expect(await t.u16()).toEqual(new Uint16Array([0x3fc0, 0xbe80]));
  });

  it('sums byte sizes', () => {
    const sd = memoryStateDict({ a: new Float32Array(3), b: new Uint16Array(5) });
    expect(sd.totalBytes()).toBe(3 * 4 + 5 * 2); // 3 f32 + 5 bf16
  });

  it('handles an empty dict', () => {
    const sd = memoryStateDict({});
    expect(sd.tensors.size).toBe(0);
    expect(sd.totalBytes()).toBe(0);
  });
});

describe('derivedTensor', () => {
  it('computes once, however often it is read', async () => {
    const compute = vi.fn(async () => new Float32Array([0.25]));
    const t = derivedTensor([1], compute);
    await Promise.all([t.f32(), t.f32(), t.data()]);
    expect(compute).toHaveBeenCalledOnce();
  });

  it('rejects an accessor that does not match what compute made', async () => {
    const words = derivedTensor([1], async () => new Uint32Array([1]));
    const floats = derivedTensor([1], async () => new Float32Array([1]));
    await expect(words.f32()).rejects.toThrow(/not f32/);
    await expect(floats.words()).rejects.toThrow(/not words/);
    await expect(floats.u16()).rejects.toThrow(/no raw u16/);
  });
});
