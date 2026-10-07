import { describe, expect, it } from 'vitest';
import { layerNorm, mean, meanSquare, softmax, topk } from '../../../../src/core/graph/ops.ts';
import { input, meta } from '../values.ts';

describe('reductions over the last dim', () => {
  it('mean gives one value per row, in the input dtype', () => {
    expect(mean(input([4, 8], 'f16')).shape).toEqual(meta([4, 1], 'f16'));
  });

  it('meanSquare gives one f32 value per row, even for f16', () => {
    // x² overflows f16 at realistic activations, so the result stays f32
    expect(meanSquare(input([4, 8], 'f16')).shape).toEqual(meta([4, 1]));
  });

  it('softmax keeps the shape', () => {
    expect(softmax(input([4, 8], 'f16')).shape).toEqual(meta([4, 8], 'f16'));
  });

  it.each([
    ['mean', mean],
    ['meanSquare', meanSquare],
    ['softmax', softmax],
  ] as const)('%s rejects anything but 2D', (_, op) => {
    expect(() => op(input([8]))).toThrow(/expected a 2D/);
  });
});

describe('layerNorm', () => {
  const x = input([4, 8]);

  it('keeps the shape and carries eps', () => {
    const weight = input([8]);
    const bias = input([8]);
    const c = layerNorm(x, weight, 1e-6, bias);
    expect(c.shape).toEqual(meta([4, 8]));
    expect(c.pending).toMatchObject({ scalar: 1e-6, inputs: [x, weight, bias] });
  });

  it('works without a bias, with eps 1e-5 by default', () => {
    expect(layerNorm(x, input([8])).pending).toMatchObject({ scalar: 1e-5 });
  });

  it.each([
    {
      name: 'a weight of the wrong length',
      weight: input([4]),
      error: /weight must be f32 \[N\] = 8 elems/,
    },
    { name: 'an f16 weight on f32', weight: input([8], 'f16'), error: /weight must be f32/ },
    {
      name: 'a bias of the wrong length',
      bias: input([4]),
      error: /bias must be f32 \[N\] = 8 elems/,
    },
    { name: 'eps of 0', eps: 0, error: /eps must be positive/ },
    { name: 'a negative eps', eps: -1e-5, error: /eps must be positive/ },
    { name: 'eps NaN', eps: NaN, error: /eps must be positive/ },
  ])('rejects $name', ({ weight = input([8]), bias, eps = 1e-5, error }) => {
    expect(() => layerNorm(x, weight, eps, bias)).toThrow(error);
  });
});

describe('topk', () => {
  it('packs 4 ids then 4 weights per row, as f32', () => {
    expect(topk(input([3, 32], 'f16'), 4).shape).toEqual(meta([3, 8]));
  });

  it('takes exactly 4 columns', () => {
    expect(topk(input([3, 4]), 4).shape).toEqual(meta([3, 8]));
  });

  it.each([
    { name: 'k other than 4', x: input([3, 32]), k: 2, error: /only k=4 is supported, got 2/ },
    {
      name: 'fewer columns than k',
      x: input([3, 3]),
      k: 4,
      error: /need at least 4 columns, got 3/,
    },
  ])('rejects $name', ({ x, k, error }) => {
    expect(() => topk(x, k)).toThrow(error);
  });
});
