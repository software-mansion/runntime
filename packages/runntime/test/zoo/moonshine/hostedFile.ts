/** Zero checkpoints with the tensor names of the hosted Moonshine and
 *  Moonshine Streaming files, for the model load tests. */
import type { AnyModule } from '../../../src/core/nn/module.ts';
import { zeroCheckpoint } from '../chunkedLoad.ts';

/** A zero checkpoint that `model`, a Moonshine or Moonshine Streaming model
 *  of width `hidden`, loads: q, k and v apart for each fused projection it
 *  declares, the tied `proj_out.weight` of `vocab` rows, and `extra` on top
 *  under checkpoint names. `renamed` maps a parameter the hook computes to
 *  the tensor it reads, both 1D. */
export function moonshineFile(
  model: AnyModule,
  { hidden, vocab }: { hidden: number; vocab: number },
  extra: Record<string, number[]> = {},
  renamed: Record<string, string> = {},
): Uint8Array {
  const names = [...model.namedParameters()].map(([name]) => name);
  const fused = names.filter((name) => /\.(qkv|kv)\.weight$/.test(name));
  const parts = fused.flatMap((name) => {
    const attention = name.slice(0, name.lastIndexOf('.', name.length - '.weight'.length - 1));
    return [...(name.endsWith('.qkv.weight') ? ['q'] : []), 'k', 'v'].map(
      (part) => `${attention}.${part}_proj.weight`,
    );
  });
  const shapes = new Map([...model.namedParameters()].map(([name, p]) => [name, p.shape.dims]));
  return zeroCheckpoint(model, [...fused, ...Object.keys(renamed)], {
    ...Object.fromEntries(parts.map((name) => [name, [hidden, hidden]])),
    ...Object.fromEntries(
      Object.entries(renamed).map(([ours, theirs]) => [theirs, [...shapes.get(ours)!]]),
    ),
    'proj_out.weight': [vocab, hidden],
    ...extra,
  });
}
