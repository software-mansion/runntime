/** Loads a model from a checkpoint read in chunks and reports the chunks
 *  fetched, for the tests that check a load walks its file once. */
import { pending } from '../../src/core/graph/value.ts';
import { Linear } from '../../src/core/nn/layers.ts';
import type { AnyModule } from '../../src/core/nn/module.ts';
import {
  bufferSource,
  fromSafetensors,
  type LazyStateDict,
} from '../../src/core/weights/safetensors.ts';
import { safetensorsFile } from '../core/weights/checkpoint.ts';

/** A checkpoint of zeros that `model` loads: a tensor for each declared
 *  parameter, a Linear's weight in torch's [out, in], without the names in
 *  `derived` (the parameters a hook computes) and with `sources` (the
 *  tensors it computes them from). Names are sorted, as a safetensors file
 *  holds them. */
export function zeroCheckpoint(
  model: AnyModule,
  derived: readonly string[],
  sources: Record<string, number[]>,
): Uint8Array {
  const linears = new Set(
    [...model.namedModules()].filter(([, m]) => m instanceof Linear).map(([path]) => path),
  );
  const shapes: Record<string, number[]> = { ...sources };
  for (const [name, p] of model.namedParameters()) {
    if (derived.includes(name)) continue;
    const dims = [...(p.shape.dims ?? [p.shape.elems])];
    const linearWeight = name.endsWith('.weight') && linears.has(name.slice(0, -'.weight'.length));
    shapes[name] = linearWeight ? dims.reverse() : dims;
  }
  return safetensorsFile(
    Object.fromEntries(
      Object.keys(shapes)
        .sort()
        .map((name) => {
          const shape = shapes[name]!;
          return [name, { data: new Float32Array(shape.reduce((a, b) => a * b, 1)), shape }];
        }),
    ),
  );
}

/** Loads `file` into `model` in chunks of `chunkBytes` and returns the start
 *  of each chunk fetched, in fetch order. `prepare` is what the model's
 *  runner does to the state dict before the load. */
export async function chunksFetched(
  model: AnyModule,
  file: Uint8Array,
  chunkBytes: number,
  prepare: (sd: LazyStateDict) => void = () => {},
): Promise<number[]> {
  const inner = bufferSource(file);
  const reads: number[] = [];
  const source = {
    read: (begin: number, end: number) => {
      reads.push(begin);
      return inner.read(begin, end);
    },
  };
  const sd = await fromSafetensors(source, { chunkBytes });
  const headerReads = reads.length;
  prepare(sd);
  await model.loadStateDict(sd, { upload: (_data, shape) => pending(shape, 'reshape', []) });
  return reads.slice(headerReads);
}

/** `chunks` as a load that walks the file once fetches them: each once, in
 *  file order. */
export const onceInFileOrder = (chunks: readonly number[]): number[] =>
  [...new Set(chunks)].sort((a, b) => a - b);
