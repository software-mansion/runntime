/** Safetensors container: header parsing and byte-range access. The header's
 *  data_offsets are payload-relative; this module stores absolute file
 *  offsets. */

import { RunntimeError } from '../error.ts';
import { bf16ToF32, f16ToF32 } from './convert.ts';
import type { WeightCache } from './cache.ts';
import { cachedRangeSource } from './cachedSource.ts';

export type SafeDtype =
  'F64' | 'F32' | 'F16' | 'BF16' | 'U32' | 'I64' | 'I32' | 'I16' | 'I8' | 'U8' | 'BOOL';

export interface SafeTensorInfo {
  dtype: SafeDtype;
  shape: number[];
  begin: number;
  end: number;
}

export interface SafetensorsIndex {
  metadata: Record<string, string>;
  tensors: Map<string, SafeTensorInfo>;
}

/** Bytes [begin, end) of a checkpoint file. Must return a standalone
 *  Uint8Array at byteOffset 0, viewable at any width. */
export interface RangeSource {
  read(begin: number, end: number): Promise<Uint8Array>;
}

/** Wraps bytes already in memory as a RangeSource, for a `modelPath` that is
 *  not a URL. */
export function bufferSource(buf: ArrayBuffer | Uint8Array): RangeSource {
  const bytes = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  return {
    read: (begin, end) => {
      if (end > bytes.length) {
        return Promise.reject(
          new Error(`safetensors: read [${begin},${end}) past end ${bytes.length}`),
        );
      }
      // Explicit, because Node's Buffer overrides .slice with a view.
      return Promise.resolve(Uint8Array.prototype.slice.call(bytes, begin, end));
    },
  };
}

const MAX_HEADER_BYTES = 100 * 1024 * 1024; // matches the reference impl's sanity cap

/** Bytes per element, by safetensors dtype. Doubles as the set of dtypes we
 *  accept: a header naming anything else is rejected rather than carried to
 *  the read, where the unknown width would mis-slice every later tensor. */
const DTYPE_BYTES: Record<SafeDtype, number> = {
  F64: 8,
  F32: 4,
  F16: 2,
  BF16: 2,
  U32: 4,
  I64: 8,
  I32: 4,
  I16: 2,
  I8: 1,
  U8: 1,
  BOOL: 1,
};

/** Validate one header entry and resolve it to absolute file offsets. The
 *  header is untrusted input: without these checks a malformed entry yields
 *  begin/end of NaN, and every read built on it silently returns the wrong
 *  bytes. The byte-span cross-check also catches a truncated payload here,
 *  at parse time, rather than as garbage weights much later. */
function tensorInfoOf(name: string, value: unknown, dataStart: number): SafeTensorInfo {
  const bad = (why: string): never => {
    throw new Error(`safetensors: header entry '${name}' ${why}`);
  };
  if (typeof value !== 'object' || value === null) bad('is not an object');
  const e = value as { dtype?: unknown; shape?: unknown; data_offsets?: unknown };

  if (typeof e.dtype !== 'string' || !(e.dtype in DTYPE_BYTES)) {
    bad(`has unsupported dtype ${JSON.stringify(e.dtype)}`);
  }
  const dtype = e.dtype as SafeDtype;

  if (!Array.isArray(e.shape) || !e.shape.every((n) => Number.isSafeInteger(n) && n >= 0)) {
    bad(`has a malformed shape ${JSON.stringify(e.shape)}`);
  }
  const shape = e.shape as number[];

  if (
    !Array.isArray(e.data_offsets) ||
    e.data_offsets.length !== 2 ||
    !e.data_offsets.every((n) => Number.isSafeInteger(n) && n >= 0)
  ) {
    bad(`has malformed data_offsets ${JSON.stringify(e.data_offsets)}`);
  }
  const [from, to] = e.data_offsets as [number, number];
  if (to < from) bad(`has data_offsets [${from}, ${to}] running backwards`);

  const elems = shape.reduce((a, b) => a * b, 1);
  const want = elems * DTYPE_BYTES[dtype];
  if (to - from !== want) {
    bad(`spans ${to - from} bytes but shape [${shape.join(', ')}] of ${dtype} needs ${want}`);
  }
  return { dtype, shape, begin: dataStart + from, end: dataStart + to };
}

export async function parseSafetensorsHeader(source: RangeSource): Promise<SafetensorsIndex> {
  const lenBytes = await source.read(0, 8);
  const headerLen = Number(
    new DataView(lenBytes.buffer, lenBytes.byteOffset, lenBytes.byteLength).getBigUint64(0, true),
  );
  if (!Number.isSafeInteger(headerLen) || headerLen <= 0 || headerLen > MAX_HEADER_BYTES) {
    // A dev server's HTML fallback decodes to a huge u64; name the real cause.
    const printable = [...lenBytes].every(
      (b) => b === 0x09 || b === 0x0a || b === 0x0d || (b >= 0x20 && b <= 0x7e),
    );
    if (printable) {
      throw new Error(
        `safetensors: got text starting ${JSON.stringify(new TextDecoder().decode(lenBytes))} ` +
          'instead of a binary header — this is not a safetensors payload. Most likely the ' +
          'file is missing at this URL and the server returned an HTML fallback page.',
      );
    }
    throw new Error(`safetensors: implausible header length ${headerLen}`);
  }
  const headerBytes = await source.read(8, 8 + headerLen);
  let header: unknown;
  try {
    header = JSON.parse(new TextDecoder().decode(headerBytes));
  } catch (err) {
    throw new Error(`safetensors: header is not valid JSON: ${(err as Error).message}`, {
      cause: err,
    });
  }
  if (typeof header !== 'object' || header === null || Array.isArray(header)) {
    throw new Error('safetensors: header is not a JSON object');
  }
  // data_offsets are relative to the payload start.
  const dataStart = 8 + headerLen;
  let metadata: Record<string, string> = {};
  const entries: [string, SafeTensorInfo][] = [];
  for (const [name, value] of Object.entries(header)) {
    if (name === '__metadata__') {
      metadata = value as Record<string, string>;
      continue;
    }
    entries.push([name, tensorInfoOf(name, value, dataStart)]);
  }
  // Map order becomes file order; the format does not require the header to
  // be written that way.
  entries.sort((a, b) => a[1].begin - b[1].begin);
  return { metadata, tensors: new Map(entries) };
}

/** RangeSource over HTTP. A short 206 body is retried, so no silent zeros reach
 *  a weight buffer; a 200 means no Range support, so the whole file downloads
 *  once and is memoized.
 *
 *  The first read runs alone — until one response arrives there is no way to
 *  know whether Range is ignored, and racing reads would each fetch the whole
 *  file. */
export function httpRangeSource(
  url: string,
  opts: { onBytes?: (n: number) => void; fetchFn?: typeof fetch } = {},
): RangeSource {
  const fetchFn = opts.fetchFn ?? fetch;
  let whole: Promise<Uint8Array> | undefined;
  let firstSettled: Promise<void> | undefined;

  const attemptRead = async (begin: number, end: number): Promise<Uint8Array> => {
    if (whole) {
      const file = await whole;
      return file.slice(begin, end);
    }
    const res = await fetchFn(url, { headers: { Range: `bytes=${begin}-${end - 1}` } });
    if (!res.ok) throw new Error(`status ${res.status}`);
    // Refuse a dev server's index.html before a WeightCache persists it.
    if ((res.headers.get('content-type') ?? '').startsWith('text/html')) {
      throw new Error(
        `got text/html from ${url} — the file is missing and the server returned its HTML fallback page`,
      );
    }
    const body = new Uint8Array(await res.arrayBuffer());
    if (res.status === 200) {
      // Range ignored; keep the whole file and serve slices from it.
      whole = Promise.resolve(body);
      opts.onBytes?.(body.length);
      return body.slice(begin, end);
    }
    if (body.length !== end - begin) {
      throw new Error(`short body: got ${body.length}, expected ${end - begin}`);
    }
    opts.onBytes?.(body.length);
    return body;
  };

  const readWithRetries = async (begin: number, end: number): Promise<Uint8Array> => {
    let lastErr: Error | undefined;
    for (let attempt = 0; attempt < 3; attempt++) {
      try {
        return await attemptRead(begin, end);
      } catch (err) {
        lastErr = err as Error;
        if (attempt < 2) await new Promise((r) => setTimeout(r, 500 * (attempt + 1)));
      }
    }
    throw new Error(
      `safetensors range [${begin},${end}) of ${url}: fetch failed after retries (${lastErr?.message})`,
    );
  };

  return {
    async read(begin, end) {
      if (firstSettled) {
        await firstSettled;
        return readWithRetries(begin, end);
      }
      const first = readWithRetries(begin, end);
      firstSettled = first.then(
        () => undefined,
        () => undefined, // gate only orders requests; the error surfaces via `first`
      );
      return first;
    },
  };
}

export const DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024;

/** Coalesces small reads into aligned `chunkBytes` windows, so a checkpoint
 *  costs about ceil(size/chunk) requests rather than one per tensor. Only the
 *  two most recent chunks are kept. `totalBytes` clamps the last one, which
 *  would otherwise trip the short-body check. */
export function chunkedSource(
  inner: RangeSource,
  totalBytes: number,
  chunkBytes = DEFAULT_CHUNK_BYTES,
): RangeSource {
  const chunks = new Map<number, Promise<Uint8Array>>(); // insertion order = LRU order
  // A fold reads its conv weight and its norm params in one Promise.all, and
  // exporters often park the small tensors at the front of the file, so two
  // windows let a pair like that evict each other every time.
  const KEEP = 4;
  const chunk = (i: number): Promise<Uint8Array> => {
    let p = chunks.get(i);
    if (p) {
      chunks.delete(i); // re-insert below to mark most-recently-used
    } else {
      const begin = i * chunkBytes;
      p = inner.read(begin, Math.min(begin + chunkBytes, totalBytes));
      p.catch(() => chunks.delete(i)); // failed fetches must not be memoized
    }
    chunks.set(i, p);
    for (const k of chunks.keys()) {
      if (chunks.size <= KEEP) break;
      chunks.delete(k);
    }
    return p;
  };

  return {
    async read(begin, end) {
      if (end > totalBytes) {
        throw new Error(`safetensors: read [${begin},${end}) past end ${totalBytes}`);
      }
      const first = Math.floor(begin / chunkBytes);
      const last = Math.floor((end - 1) / chunkBytes);
      const parts = await Promise.all(
        Array.from({ length: last - first + 1 }, (_, k) => chunk(first + k)),
      );
      if (parts.length === 1) {
        const offset = begin - first * chunkBytes;
        return parts[0]!.slice(offset, offset + (end - begin)); // slice = standalone copy
      }
      const out = new Uint8Array(end - begin);
      let written = 0;
      for (let i = first; i <= last; i++) {
        const from = i === first ? begin - i * chunkBytes : 0;
        const to = i === last ? end - i * chunkBytes : chunkBytes;
        out.set(parts[i - first]!.subarray(from, to), written);
        written += to - from;
      }
      return out;
    },
  };
}

/** A tensor that may not be fetched or computed yet. File-backed ones carry
 *  the real dtype; derived ones report a nominal 'F32' and say what they
 *  hold through data(). */
interface LazyTensorBase {
  shape: number[];
  byteLength: number;
  f32(): Promise<Float32Array>;
  halfWords(): Promise<Uint32Array>;
  /** U32 file tensors, or a derived Uint32Array. */
  words(): Promise<Uint32Array>;
  /** Raw BF16 or F16 bits, unconverted. */
  u16(): Promise<Uint16Array>;
}

/** Bytes that sit in the file (or were handed in as bits): `dtype` says
 *  exactly what they are. */
export interface FileTensor extends LazyTensorBase {
  kind: 'file';
  dtype: SafeDtype;
}

/** A tensor a transform computes. `dtype` is nominal — the loader must not
 *  branch on it; `data()` returns whatever compute() produced, and its
 *  runtime type says what it holds. A Uint16Array holds f16 halfwords. */
export interface DerivedTensor extends LazyTensorBase {
  kind: 'derived';
  dtype: 'F32';
  data(): Promise<Float32Array | Uint16Array | Uint32Array>;
}

/** Symmetric int weights with one scale per `groupSize` values along the last
 *  dim: value = int · scale of its group. `shape` is logical, one element per
 *  int even for int4, and `byteLength` covers the ints and the scales. The
 *  float accessors reject with CHECKPOINT_MISMATCH. */
export interface QuantTensor extends LazyTensorBase {
  kind: 'quant';
  /** What the tensor is, for error messages: its name in the file, or the
   *  helper that made it and the sources. */
  name: string;
  bits: 4 | 8;
  groupSize: number;
  /** One signed value per logical element, row-major; int4 comes unpacked. */
  ints(): Promise<Int8Array>;
  /** Row-major, shape `[...lead, last / groupSize]`. F16 halfwords as stored
   *  in the file; f32 once a helper changed them. */
  scales(): Promise<Float32Array | Uint16Array>;
}

export type LazyTensor = FileTensor | DerivedTensor | QuantTensor;

export interface LazyStateDict {
  metadata: Record<string, string>;
  tensors: Map<string, LazyTensor>;
  totalBytes(): number;
}

/** Views `bytes` as u16 halfwords; copies only on an odd offset. */
function toHalfwords(bytes: Uint8Array): Uint16Array {
  if (bytes.byteOffset % 2 === 0)
    return new Uint16Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 2);
  return new Uint16Array(bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength));
}

/** Views `bytes` as f32; copies when misaligned. */
function toF32(bytes: Uint8Array): Float32Array {
  if (bytes.byteOffset % 4 === 0)
    return new Float32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  return new Float32Array(
    bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength),
  );
}

/** Zero-pads `bytes` to a multiple of 4 and views it as u32 words. */
function toWords(bytes: Uint8Array): Uint32Array {
  if (bytes.byteOffset % 4 === 0 && bytes.byteLength % 4 === 0)
    return new Uint32Array(bytes.buffer, bytes.byteOffset, bytes.byteLength / 4);
  const padded = new Uint8Array(Math.ceil(bytes.byteLength / 4) * 4);
  padded.set(bytes);
  return new Uint32Array(padded.buffer);
}

function fileTensor(name: string, info: SafeTensorInfo, source: RangeSource): FileTensor {
  const fetchBytes = () => source.read(info.begin, info.end);
  return {
    kind: 'file',
    dtype: info.dtype,
    shape: info.shape,
    byteLength: info.end - info.begin,
    async f32() {
      const bytes = await fetchBytes();
      if (info.dtype === 'F32') return toF32(bytes);
      if (info.dtype === 'BF16') return bf16ToF32(toHalfwords(bytes));
      if (info.dtype === 'F16') return f16ToF32(toHalfwords(bytes));
      throw new RunntimeError(
        'CHECKPOINT_MISMATCH',
        `tensor ${name}: cannot decode ${info.dtype} to f32`,
      );
    },
    async halfWords() {
      if (info.dtype !== 'F16')
        throw new RunntimeError(
          'CHECKPOINT_MISMATCH',
          `tensor ${name}: halfWords() needs F16, got ${info.dtype}`,
        );
      return toWords(await fetchBytes());
    },
    async words() {
      if (info.dtype !== 'U32')
        throw new RunntimeError(
          'CHECKPOINT_MISMATCH',
          `tensor ${name}: words() needs U32, got ${info.dtype}`,
        );
      return toWords(await fetchBytes());
    },
    async u16() {
      if (info.dtype !== 'BF16' && info.dtype !== 'F16')
        throw new RunntimeError(
          'CHECKPOINT_MISMATCH',
          `tensor ${name}: u16() needs BF16 or F16, got ${info.dtype}`,
        );
      return toHalfwords(await fetchBytes());
    },
  };
}

/** A QuantTensor over `ints` and `scales`. Like derivedTensor it holds
 *  nothing between reads. */
export function quantTensor(
  parts: Pick<
    QuantTensor,
    'name' | 'shape' | 'bits' | 'groupSize' | 'byteLength' | 'ints' | 'scales'
  >,
): QuantTensor {
  const refuse = () =>
    Promise.reject(
      new RunntimeError(
        'CHECKPOINT_MISMATCH',
        `tensor ${parts.name} is quantized (int${parts.bits}, groups of ${parts.groupSize}) and has no float reading`,
      ),
    );
  return { kind: 'quant', ...parts, f32: refuse, halfWords: refuse, words: refuse, u16: refuse };
}

/** Unpacks int4 bytes, two values each with the low nibble first, into one
 *  signed value per element. */
function unpackInt4(bytes: Uint8Array): Int8Array {
  const out = new Int8Array(bytes.length * 2);
  for (let i = 0; i < bytes.length; i++) {
    const byte = bytes[i]!;
    // (n ^ 8) - 8 sign-extends a 4-bit two's-complement nibble.
    out[2 * i] = ((byte & 0x0f) ^ 8) - 8;
    out[2 * i + 1] = ((byte >> 4) ^ 8) - 8;
  }
  return out;
}

/** The int weight `name` (I8 for int8, U8 holding two int4 per byte) paired
 *  with its F16 `scale`, checked against the file's quantization setting. */
function fileQuantTensor(
  name: string,
  info: SafeTensorInfo,
  scale: SafeTensorInfo | undefined,
  quant: QuantConfig,
  source: RangeSource,
): QuantTensor {
  const bad = (why: string): never => {
    throw new RunntimeError('CHECKPOINT_MISMATCH', `safetensors: quantized tensor ${name} ${why}`);
  };
  const { bits, groupSize } = quant;
  const dtype = bits === 8 ? 'I8' : 'U8';
  if (info.dtype !== dtype) bad(`is ${info.dtype}, but int${bits} is stored as ${dtype}`);
  if (info.shape.length !== 2) bad(`needs a 2D shape, got [${info.shape.join(', ')}]`);
  const out = info.shape[0]!;
  const cols = bits === 8 ? info.shape[1]! : info.shape[1]! * 2;
  if (cols % groupSize !== 0) bad(`has ${cols} columns, not a multiple of groupSize ${groupSize}`);
  if (!scale) return bad(`has no ${name}_scale`);
  const scaleShape = [out, cols / groupSize];
  if (scale.dtype !== 'F16' || scale.shape.join() !== scaleShape.join()) {
    bad(
      `needs an F16 ${name}_scale of shape [${scaleShape.join(', ')}], ` +
        `got ${scale.dtype} [${scale.shape.join(', ')}]`,
    );
  }
  return quantTensor({
    name,
    shape: [out, cols],
    bits,
    groupSize,
    byteLength: info.end - info.begin + (scale.end - scale.begin),
    async ints() {
      const bytes = await source.read(info.begin, info.end);
      if (bits === 4) return unpackInt4(bytes);
      return new Int8Array(bytes.buffer, bytes.byteOffset, bytes.byteLength);
    },
    scales: async () => toHalfwords(await source.read(scale.begin, scale.end)),
  });
}

export function eagerTensor(data: Float32Array | Uint32Array, shape: number[]): LazyTensor {
  return derivedTensor(shape, () => Promise.resolve(data), data.byteLength);
}

/** `byteLength` is this tensor's share of the download. A transform over
 *  several file tensors gives the sum to one output and 0 to the rest.
 *  Every read runs `compute` again: the tensor holds nothing between reads,
 *  so a load keeps only the tensor being bound in host memory. A transform
 *  with several outputs shares one read itself, like `splitRows`. */
export function derivedTensor(
  shape: number[],
  compute: () => Promise<Float32Array | Uint16Array | Uint32Array>,
  byteLength = 0,
): DerivedTensor {
  /** Shared by halfWords() and words(); both want a Uint32Array back. */
  const asWords = async (): Promise<Uint32Array> => {
    const data = await compute();
    if (!(data instanceof Uint32Array))
      throw new Error('derived tensor holds f32 or halfwords, not words');
    return data;
  };
  return {
    kind: 'derived',
    dtype: 'F32', // nominal only — see DerivedTensor
    shape,
    byteLength,
    data: compute,
    /** Halfwords decode, so an f16 transform still loads into an f32
     *  parameter or a hook that reads f32. */
    async f32() {
      const data = await compute();
      if (data instanceof Uint16Array) return f16ToF32(data);
      if (!(data instanceof Float32Array))
        throw new Error('derived tensor holds packed words, not f32');
      return data;
    },
    halfWords: asWords,
    words: asWords,
    /** Lets a transform that already made f16 halfwords skip the f32 detour. */
    async u16(): Promise<Uint16Array> {
      const data = await compute();
      if (!(data instanceof Uint16Array)) throw new Error('derived tensor has no raw u16 bits');
      return data;
    },
  };
}

/** In-memory LazyStateDict, for weights already in RAM. Bare arrays get a flat
 *  shape; packed and quantized tensors need the object form. Uint16Array data
 *  is raw bf16 bits. */
export type MemoryStateDict = Record<
  string,
  | Float32Array
  | Uint32Array
  | Uint16Array
  | { data: Float32Array | Uint32Array | Uint16Array; shape: number[] }
>;

export function memoryStateDict(entries: MemoryStateDict): LazyStateDict {
  const tensors = new Map<string, LazyTensor>();
  for (const [name, entry] of Object.entries(entries)) {
    const { data, shape } = ArrayBuffer.isView(entry)
      ? { data: entry, shape: [entry.length] }
      : entry;
    tensors.set(
      name,
      data instanceof Uint16Array ? bf16Tensor(data, shape) : eagerTensor(data, shape),
    );
  }
  return {
    metadata: {},
    tensors,
    totalBytes: () => [...tensors.values()].reduce((sum, t) => sum + t.byteLength, 0),
  };
}

/** In-memory bf16-bits tensor; f32() expands the bits into an f32's top
 *  half. */
function bf16Tensor(bits: Uint16Array, shape: number[]): FileTensor {
  return {
    kind: 'file',
    dtype: 'BF16',
    shape,
    byteLength: bits.byteLength,
    u16: () => Promise.resolve(bits),
    f32: () => Promise.resolve(bf16ToF32(bits)),
    halfWords: () => Promise.reject(new Error('bf16 tensor holds raw bits, not packed f16')),
    words: () => Promise.reject(new Error('bf16 tensor holds raw bits, not u32 words')),
  };
}

/** Metadata key of a quantized file: a JSON string
 *  `{"version":1,"bits":8|4,"groupSize":N,"scheme":"symmetric"}`. */
const QUANT_METADATA_KEY = 'runntime.quant';

/** How a quantized file stores its int weights: `bits` per value, one f16
 *  scale per `groupSize` values along each row. */
interface QuantConfig {
  bits: 4 | 8;
  groupSize: number;
}

/** The file's quantization setting, or undefined for a float file. Throws
 *  CHECKPOINT_MISMATCH on a setting this reader does not know. */
function quantConfigOf(metadata: Record<string, string>): QuantConfig | undefined {
  const raw: unknown = metadata[QUANT_METADATA_KEY];
  if (raw === undefined) return undefined;
  const bad = (why: string): never => {
    throw new RunntimeError(
      'CHECKPOINT_MISMATCH',
      `safetensors: '${QUANT_METADATA_KEY}' metadata ${JSON.stringify(raw)} ${why}`,
    );
  };
  if (typeof raw !== 'string') return bad('is not a string');
  let json: unknown;
  try {
    json = JSON.parse(raw);
  } catch {
    return bad('is not valid JSON');
  }
  if (typeof json !== 'object' || json === null || Array.isArray(json)) {
    return bad('is not a JSON object');
  }
  const { version, bits, groupSize, scheme } = json as Record<string, unknown>;
  if (version !== 1) return bad('has a version other than 1');
  if (bits !== 4 && bits !== 8) return bad('needs bits 4 or 8');
  if (typeof groupSize !== 'number' || !Number.isSafeInteger(groupSize) || groupSize <= 0) {
    return bad('needs a positive integer groupSize');
  }
  if (scheme !== 'symmetric') return bad("needs scheme 'symmetric'");
  return { bits, groupSize };
}

export async function fromSafetensors(
  src: string | RangeSource,
  opts: {
    onBytes?: (n: number) => void;
    chunkBytes?: number;
    /** Saves every fetched byte range here, so warm loads touch no network.
     *  Cached bytes are never revalidated, so a file that changes content must
     *  also change name. */
    cache?: WeightCache;
    cacheId?: string;
  } = {},
): Promise<LazyStateDict> {
  let raw = typeof src === 'string' ? httpRangeSource(src, { onBytes: opts.onBytes }) : src;
  let cacheId: string | undefined;
  if (opts.cache) {
    cacheId = opts.cacheId ?? (typeof src === 'string' ? src.split('/').pop() : undefined);
    if (!cacheId) {
      throw new RunntimeError(
        'INVALID_ARGUMENT',
        'fromSafetensors: cache with a RangeSource src needs an explicit cacheId',
      );
    }
  }
  const cached = (inner: RangeSource) =>
    opts.cache && cacheId ? cachedRangeSource(inner, opts.cache, cacheId) : inner;
  let source = cached(raw);
  let index: SafetensorsIndex;
  try {
    index = await parseSafetensorsHeader(source);
  } catch (err) {
    // A cached HTML fallback page would replay forever, so evict this file's
    // entries (never the whole cache: other models live there) and retry
    // once from a fresh source.
    if (!opts.cache || typeof src !== 'string' || !(err instanceof Error)) throw err;
    if (!/not a safetensors payload|implausible header/.test(err.message)) throw err;
    await opts.cache.deletePrefix(`st/${cacheId}/`);
    raw = httpRangeSource(src, { onBytes: opts.onBytes });
    source = cached(raw);
    index = await parseSafetensorsHeader(source);
  }
  let payloadEnd = 0;
  for (const info of index.tensors.values()) payloadEnd = Math.max(payloadEnd, info.end);
  // Header reads stay on the raw source. Tensor reads coalesce into chunks
  // below the cache, so a cold load costs one request per chunk while the
  // cache keys stay per tensor and a warm load never touches the network.
  const chunkBytes = opts.chunkBytes ?? DEFAULT_CHUNK_BYTES;
  const data = cached(
    chunkBytes > 0 && payloadEnd > 0 ? chunkedSource(raw, payloadEnd, chunkBytes) : raw,
  );
  const quant = quantConfigOf(index.metadata);
  const isInt = (info: SafeTensorInfo) => info.dtype === 'I8' || info.dtype === 'U8';
  // Each int weight X takes X_scale into its QuantTensor, so the scales get
  // no entry of their own.
  const scaleNames = new Set(
    [...index.tensors].filter(([, info]) => isInt(info)).map(([name]) => `${name}_scale`),
  );
  const tensors = new Map<string, LazyTensor>();
  for (const [name, info] of index.tensors) {
    if (scaleNames.has(name)) continue;
    if (!isInt(info)) {
      tensors.set(name, fileTensor(name, info, data));
      continue;
    }
    if (!quant) {
      throw new RunntimeError(
        'CHECKPOINT_MISMATCH',
        `safetensors: tensor ${name} is ${info.dtype}, but the file has no '${QUANT_METADATA_KEY}' metadata`,
      );
    }
    tensors.set(name, fileQuantTensor(name, info, index.tensors.get(`${name}_scale`), quant, data));
  }
  return {
    metadata: index.metadata,
    tensors,
    totalBytes: () => [...tensors.values()].reduce((sum, t) => sum + t.byteLength, 0),
  };
}
