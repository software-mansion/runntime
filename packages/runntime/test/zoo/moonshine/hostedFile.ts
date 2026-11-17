/** Zero checkpoints with the tensor names of the hosted Moonshine and
 *  Moonshine Streaming files, for the model load tests. */
import type { AnyModule } from '../../../src/core/nn/module.ts';
import { zeroCheckpoint } from '../chunkedLoad.ts';

/** A zero checkpoint that `model`, a Moonshine or Moonshine Streaming model
 *  of width `hidden`, loads, with the tied `proj_out.weight` of `vocab` rows
 *  and `extra` on top under checkpoint names. `renamed` maps a parameter the
 *  hook computes to the tensor it reads, both 1D. */
export function moonshineFile(
  model: AnyModule,
  { hidden, vocab }: { hidden: number; vocab: number },
  extra: Record<string, number[]> = {},
  renamed: Record<string, string> = {},
): Uint8Array {
  const shapes = new Map([...model.namedParameters()].map(([name, p]) => [name, p.shape.dims]));
  return zeroCheckpoint(model, Object.keys(renamed), {
    ...Object.fromEntries(
      Object.entries(renamed).map(([ours, theirs]) => [theirs, [...shapes.get(ours)!]]),
    ),
    'proj_out.weight': [vocab, hidden],
    ...extra,
  });
}
