import { describe, it } from 'vitest';
import { checkMatmul } from './checkMatmul.ts';

describe('matmul kernel: plain and tiled routes', () => {
  it.each([
    { name: 'one row under the tiled cutoff (plain)', m: 7, k: 16, n: 12 },
    { name: 'odd sizes (plain)', m: 3, k: 7, n: 5 },
    { name: 'M past the tile edge (tiled)', m: 65, k: 33, n: 67 },
  ])('$name: [$m, $k] × [$k, $n]', ({ m, k, n }) => checkMatmul(m, k, n));

  // Known bug: the tiled kernel's B index goes through f32, which counts
  // exactly only up to 2^24. Turn this into a plain `it` with the fix.
  it.fails('B larger than 2^24 elements (tiled): [9, 2560] × [2560, 9728]', () =>
    checkMatmul(9, 2560, 9728),
  );
});
