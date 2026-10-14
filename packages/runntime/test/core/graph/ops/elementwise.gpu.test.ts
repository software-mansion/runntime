import { describe, it } from 'vitest';
import { tensor } from '../../../../src/core/gpu/buffers.ts';
import { toArray } from '../../../../src/core/gpu/eval.ts';
import { add, mul } from '../../../../src/core/graph/ops.ts';
import { matrix } from '../../../../src/core/graph/value.ts';
import { expectClose, gpu, requireFeature } from '../../../gpu.ts';

const { root, ex } = await gpu();

const A = [1, 2, 3, 4, 5, 6]; // [2, 3]

describe('add / mul on the GPU', () => {
  it.each([
    {
      name: 'add, same shape',
      op: add,
      b: [10, 20, 30, 40, 50, 60],
      bDims: [2, 3],
      want: [11, 22, 33, 44, 55, 66],
    },
    {
      name: 'add, row [N]',
      op: add,
      b: [10, 20, 30],
      bDims: [1, 3],
      want: [11, 22, 33, 14, 25, 36],
    },
    {
      name: 'add, column [M, 1]',
      op: add,
      b: [10, 20],
      bDims: [2, 1],
      want: [11, 12, 13, 24, 25, 26],
    },
    {
      name: 'mul, same shape',
      op: mul,
      b: [2, 2, 2, 0.5, 0.5, 0.5],
      bDims: [2, 3],
      want: [2, 4, 6, 2, 2.5, 3],
    },
    { name: 'mul, single value', op: mul, b: [-1], bDims: [1, 1], want: [-1, -2, -3, -4, -5, -6] },
  ])('$name', async ({ op, b, bDims, want }) => {
    const a = tensor(root, new Float32Array(A), [2, 3]);
    const bv = tensor(root, new Float32Array(b), [bDims[0]!, bDims[1]!]);
    expectClose(await toArray(op(a, bv), ex), want, { atol: 0, rtol: 0 });
  });

  it('adds f16 inputs', async (ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    // All values are exact in f16, so the sum is too.
    const a = tensor(root, new Float32Array(A), { ...matrix(2, 3), dtype: 'f16' });
    const b = tensor(root, new Float32Array([0.5, 0.25, -1]), { ...matrix(1, 3), dtype: 'f16' });
    expectClose(await toArray(add(a, b), ex), [1.5, 2.25, 2, 4.5, 5.25, 5], { atol: 0, rtol: 0 });
  });
});
