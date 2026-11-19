import { describe, expect, it } from 'vitest';
import { combinedProgress } from '../../src/zoo/load.ts';

describe('combinedProgress', () => {
  it('reports the bytes of every file together', () => {
    const calls: [string, number, number][] = [];
    const track = combinedProgress((name, done, total) => calls.push([name, done, total]));
    const a = track('a');
    const b = track('b');
    a('a.0', 10, 100);
    b('b.0', 5, 20);
    a('a.1', 100, 100);
    b('b.1', 20, 20);
    expect(calls).toEqual([
      ['a.0', 10, 100],
      ['b.0', 15, 120],
      ['a.1', 105, 120],
      ['b.1', 120, 120],
    ]);
  });

  it('calls nothing without an onProgress', () => {
    expect(() => combinedProgress(undefined)('a')('a.0', 1, 1)).not.toThrow();
  });
});
