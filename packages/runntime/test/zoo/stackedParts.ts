/** Loads a model with random values in the Linears its FusedLinears stack
 *  and reports which part each block of a stacked weight holds, for the
 *  tests that check a model stacks q, k and v in that order. */
import { pending, type Value } from '../../src/core/graph/value.ts';
import { FusedLinear, Linear } from '../../src/core/nn/layers.ts';
import type { AnyModule } from '../../src/core/nn/module.ts';
import { f16ToF32 } from '../../src/core/weights/convert.ts';
import {
  bufferSource,
  fromSafetensors,
  memoryStateDict,
  type LazyStateDict,
} from '../../src/core/weights/safetensors.ts';
import { randomF32 } from '../gpu.ts';

/** Loads `model` from `file`, a safetensors file or a state dict, with seeded random values in the weight of each
 *  Linear a FusedLinear stacks. Maps each FusedLinear's path to its parts'
 *  paths, relative to the FusedLinear's parent, in the order its stacked
 *  weight holds them: each block of columns is matched by value against the
 *  part's weight after the hooks. A block no part matches reads '?'. */
export async function stackedParts(
  model: AnyModule,
  file: Uint8Array | LazyStateDict,
): Promise<Record<string, string[]>> {
  const sd = file instanceof Uint8Array ? await fromSafetensors(bufferSource(file)) : file;
  const modules = [...model.namedModules()];
  const parts = modules
    .filter(([, m]) => m instanceof Linear && m.fused)
    .map(([path]) => `${path}.weight`);
  for (const [i, name] of parts.entries()) {
    const shape = sd.tensors.get(name)?.shape;
    if (!shape) throw new Error(`stackedParts: the file has no ${name}`);
    const data = randomF32(
      shape.reduce((a, b) => a * b, 1),
      i + 1,
    );
    sd.tensors.set(
      name,
      memoryStateDict({ [name]: { data, shape: [...shape] } }).tensors.get(name)!,
    );
  }

  const uploaded = new Map<Value, Float32Array | Uint32Array | Uint16Array>();
  await model.loadStateDict(sd, {
    upload: (data, shape) => {
      const value = pending(shape, 'reshape', []);
      uploaded.set(value, data);
      return value;
    },
  });

  const storage = new Map(model.storageParameters());
  const found: Record<string, string[]> = {};
  for (const [path, fused] of modules) {
    if (!(fused instanceof FusedLinear)) continue;
    const parent = path.slice(0, path.lastIndexOf('.') + 1);
    const raw = uploaded.get(storage.get(`${path}.weight`)!.value)!;
    // [in, out] as the Linear stores it, out the sum of the parts' rows.
    const stacked = raw instanceof Uint16Array ? f16ToF32(raw) : Float32Array.from(raw);
    const candidates = await Promise.all(
      parts
        .filter((name) => name.startsWith(parent))
        .map(async (name) => {
          const t = sd.tensors.get(name)!;
          return {
            name: name.slice(parent.length, -'.weight'.length),
            shape: t.shape,
            data: await t.f32(),
          };
        }),
    );
    const inFeatures = candidates[0]!.shape[1]!;
    const total = stacked.length / inFeatures;
    const order: string[] = [];
    for (let offset = 0; offset < total;) {
      // A part's [out, in] rows are the block's columns.
      const part = candidates.find(
        ({ shape: [out = 0], data }) =>
          offset + out <= total &&
          data.every((want, j) => {
            const got = stacked[(j % inFeatures) * total + offset + Math.floor(j / inFeatures)]!;
            return Math.abs(got - want) <= 1e-3 + 1e-3 * Math.abs(want);
          }),
      );
      if (!part) {
        order.push('?');
        break;
      }
      order.push(part.name);
      offset += part.shape[0]!;
    }
    found[path] = order;
  }
  return found;
}
