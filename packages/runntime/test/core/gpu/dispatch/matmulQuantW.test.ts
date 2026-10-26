import { describe, expect, it } from 'vitest';
import { matmulQuantWSpec } from '../../../../src/core/gpu/dispatch/matmul.ts';
import type { DispatchCtx } from '../../../../src/core/gpu/dispatch/spec.ts';
import { matmul } from '../../../../src/core/graph/ops.ts';
import type { Value } from '../../../../src/core/graph/value.ts';
import { input, quantWeight } from '../../graph/values.ts';

/** Whether the quantW spec routes this node to the tiled kernel. */
function routesTiled(node: Value): boolean {
  const attrs = matmulQuantWSpec.attrs(node.pending!);
  // The routing reads only the node and its attrs, never the context.
  return matmulQuantWSpec.cfg(node, attrs, {} as DispatchCtx, undefined).tiled === 1;
}

describe('matmulQuantW routing', () => {
  // K 16 in groups of 4, 8 rows: enough rows for the tiled kernel.
  const q = { scales: input([4, 16]), bits: 8 as const, groupSize: 4 };

  it('runs a whole weight on the tiled kernel', () => {
    expect(routesTiled(matmul(input([8, 16]), quantWeight(16, 16, 8), q))).toBe(true);
  });

  it('keeps every expert of a fused weight on the plain kernel, the first one too', () => {
    const fused = quantWeight(32, 16, 8);
    const scales = input([8, 16]);
    for (const baseRow of [0, 16]) {
      const node = matmul(input([8, 16]), fused, {
        ...q,
        scales,
        baseRow,
        scaleBase: baseRow === 0 ? 0 : 64,
      });
      expect(routesTiled(node)).toBe(false);
    }
  });
});
