/** Reads the pair layout every rope op of a model's graph runs with, on
 *  stand-in weights and inputs, with no GPU. */
import type { Value } from '../../src/core/graph/value.ts';
import { ROPE_LAYOUTS, type RopeLayout } from '../../src/core/kernels/attention/rope.ts';

/** The layout (attr 3) and ropes a head (attr 4) of every rope node under
 *  `outputs`, each node once, in no set order. */
export function ropeLayouts(
  ...outputs: Value[]
): { layout: RopeLayout | undefined; ropesPerHead: number }[] {
  const seen = new Set<Value>();
  const layouts: { layout: RopeLayout | undefined; ropesPerHead: number }[] = [];
  const stack = [...outputs];
  while (stack.length > 0) {
    const v = stack.pop()!;
    if (seen.has(v)) continue;
    seen.add(v);
    if (!v.pending) continue;
    if (v.pending.op === 'rope') {
      const [, , , layout, ropesPerHead] = v.pending.attrs!;
      layouts.push({ layout: ROPE_LAYOUTS[layout!], ropesPerHead: ropesPerHead! });
    }
    stack.push(...v.pending.inputs);
  }
  return layouts;
}
