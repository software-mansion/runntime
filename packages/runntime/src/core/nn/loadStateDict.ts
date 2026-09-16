/** Streaming state-dict loader: transform hooks, then a strict name check over
 *  headers alone, then per-tensor fetch, adapt, preprocess and upload. Peak
 *  host memory is one tensor's working set. */
import type { TgpuRoot } from 'typegpu';
import { uploadF16, uploadF32, uploadU32 } from '../gpu/buffers.ts';
import { defaultRoot } from '../gpu/context.ts';
import { inGpuErrorScopes } from '../gpu/errorScopes.ts';
import { materialized, type Value, type ValueMeta } from '../graph/value.ts';
import { bf16ToF16Bits, f32ArrayToF16Bits } from '../weights/convert.ts';
import type { LazyStateDict, LazyTensor } from '../weights/safetensors.ts';
import { CpuParameter, type AnyModule } from './module.ts';

export type UploadFn = (data: Float32Array | Uint32Array | Uint16Array, shape: ValueMeta) => Value;

export interface LoadStateDictOpts {
  /** The GPU to upload onto. */
  root?: TgpuRoot;
  /** Custom sink overriding the default GPU upload. */
  upload?: UploadFn;
  /** After each bound tensor: cumulative file bytes done / total. */
  onProgress?: (name: string, doneBytes: number, totalBytes: number) => void;
}

/** The default GPU upload.
 *
 *  The shape decides the buffer, not the array type, so f32 data bound for a
 *  narrower parameter converts here. That is what makes `model.half()` work
 *  against a loader that only produces f32: uploading f32 words into an f16
 *  slot would otherwise build an f32 buffer wearing an f16 label, with no error
 *  and garbage in every kernel that reads it. */
export function gpuUpload(root: TgpuRoot): UploadFn {
  return (data, shape) => {
    if (data instanceof Uint16Array) {
      if (shape.dtype !== 'f16') {
        throw new Error(`upload: halfword data needs an f16 shape, got '${shape.dtype}'`);
      }
      return materialized(shape, uploadF16(root, data));
    }
    if (data instanceof Uint32Array) {
      if (shape.dtype !== 'quantW') {
        throw new Error(`upload: word data needs a quantW shape, got '${shape.dtype}'`);
      }
      return materialized(shape, uploadU32(root, data));
    }
    if (shape.dtype === 'f16') {
      return materialized(shape, uploadF16(root, f32ArrayToF16Bits(data).data));
    }
    if (shape.dtype !== 'f32') {
      throw new Error(`upload: cannot fill a '${shape.dtype}' parameter from f32 data`);
    }
    return materialized(shape, uploadF32(root, data));
  };
}

async function materialize(name: string, t: LazyTensor, want: ValueMeta['dtype']) {
  if (want === 'f16') {
    // F16 bytes are already the buffer contents; BF16 converts directly, and
    // anything else rounds down from f32.
    if (t.dtype === 'F16') return t.u16();
    if (t.dtype === 'BF16') {
      const { data, clamped } = bf16ToF16Bits(await t.u16());
      if (clamped > 0)
        console.warn(`loadStateDict: '${name}' clamped ${clamped} values to f16 range`);
      return data;
    }
    const narrow = (f32: Float32Array): Uint16Array => {
      const { data, clamped } = f32ArrayToF16Bits(f32);
      if (clamped > 0)
        console.warn(`loadStateDict: '${name}' clamped ${clamped} values to f16 range`);
      return data;
    };
    // A derived tensor's dtype is nominal: data() says what its transform
    // produced. Halfwords go straight in, f32 narrows, packed words have no
    // f16 reading. Must come after the BF16 branch: those bits reinterpreted
    // as f16 give the right magnitude with wrong values.
    if (t.kind === 'derived') {
      const data = await t.data();
      if (data instanceof Uint16Array) return data;
      if (data instanceof Float32Array) return narrow(data);
      throw new Error(`loadStateDict: '${name}' holds packed u32 words, not loadable as f16`);
    }
    return narrow(await t.f32());
  }
  if (want === 'quantW') return t.words();
  if (want === 'f32') return t.f32();
  throw new Error(`loadStateDict: '${name}' has unsupported param dtype '${want}'`);
}

const list = (names: string[]) =>
  names.slice(0, 8).join(', ') + (names.length > 8 ? ` … (+${names.length - 8})` : '');

export async function loadLazyStateDict(
  model: AnyModule,
  sd: LazyStateDict,
  opts: LoadStateDictOpts = {},
): Promise<void> {
  // Explicit sink, then explicit root, then the initRunntime() default. GPU uploads
  // run inside error scopes: OOM and validation errors never throw in JS, so
  // without them a failed load binds silently-zero weights.
  if (opts.upload) return bindStateDict(model, sd, opts.upload, opts);
  const root = opts.root ?? defaultRoot();
  return inGpuErrorScopes(root.device, 'weight upload', () =>
    bindStateDict(model, sd, gpuUpload(root), opts),
  );
}

async function bindStateDict(
  model: AnyModule,
  sd: LazyStateDict,
  upload: UploadFn,
  opts: LoadStateDictOpts,
): Promise<void> {
  model.transformStateDict?.(sd, '');
  const visit = (parent: AnyModule, parentPrefix: string): void => {
    for (const [name, child] of parent.namedChildren()) {
      const prefix = parentPrefix ? `${parentPrefix}.${name}` : name;
      child.transformStateDict?.(sd, prefix, parent);
      visit(child, prefix);
    }
  };
  visit(model, '');

  const params = [...model.namedParameters()];
  const paramNames = new Set(params.map(([n]) => n));
  const missing = params.filter(([n]) => !sd.tensors.has(n)).map(([n]) => n);
  const unexpected = [...sd.tensors.keys()].filter((n) => !paramNames.has(n));
  if (missing.length || unexpected.length) {
    throw new Error(
      'loadStateDict: state dict does not match model.' +
        (missing.length ? ` missing: ${list(missing)}.` : '') +
        (unexpected.length ? ` unexpected: ${list(unexpected)}.` : ''),
    );
  }

  // Bind in state-dict (file) order, so each chunk is read once. Module order
  // jumps around the payload and refetches evicted chunks. Keys a transform
  // added sort last; they read nothing new.
  const position = new Map([...sd.tensors.keys()].map((name, i) => [name, i]));
  params.sort(([a], [b]) => (position.get(a) ?? Infinity) - (position.get(b) ?? Infinity));

  const total = sd.totalBytes();
  let done = 0;
  for (const [name, p] of params) {
    const t = sd.tensors.get(name)!;
    const numel = t.shape.reduce((a, b) => a * b, 1);
    const expected = numel;
    if (expected !== p.shape.elems) {
      throw new Error(
        `loadStateDict: '${name}' checkpoint shape [${t.shape}] (${numel} elems) != model ${JSON.stringify(p.shape.dims)} (${p.shape.elems} ${p.shape.dtype === 'quantW' ? 'words' : 'elems'})`,
      );
    }
    if (p instanceof CpuParameter) {
      // CPU-resident. BF16 keeps its raw bits; anything else converts to f32.
      p.bindData(t.dtype === 'BF16' ? await t.u16() : await t.f32());
    } else {
      const data = await materialize(name, t, p.shape.dtype);
      p.bind(upload(p.preprocess ? p.preprocess(data) : data, p.shape));
    }
    done += t.byteLength;
    opts.onProgress?.(name, done, total);
  }
  for (const [, b] of model.namedBuffers()) {
    // A Buffer's data is computed CPU-side and always f32, and half() only
    // retyped the slot, so the narrowing happens here.
    const data = b.shape.dtype === 'f16' ? f32ArrayToF16Bits(b.data).data : b.data;
    b.bind(upload(data, b.shape));
  }
}
