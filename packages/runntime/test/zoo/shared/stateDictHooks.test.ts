import { describe, expect, it } from 'vitest';
import { memoryStateDict } from '../../../src/core/weights/safetensors.ts';
import { remap, ropeOrder } from '../../../src/zoo/shared/stateDictHooks.ts';

describe('ropeOrder', () => {
  it('takes each block from half-split to interleaved pairs', () => {
    // Rows (p, p + 2) of each block of 4 become (2p, 2p + 1).
    expect(ropeOrder(8, 4)).toEqual([0, 2, 1, 3, 4, 6, 5, 7]);
  });
});

describe('remap', () => {
  const sd = () =>
    memoryStateDict({
      'm.a': new Float32Array(1),
      'm.b': new Float32Array(1),
      other: new Float32Array(1),
    });

  it('moves the tensors under the sources to new names under the prefix', () => {
    const dict = sd();
    const a = dict.tensors.get('m.a')!;
    const { take, put, done } = remap(dict, 'p', ['m.'], 'test');
    put('x', take('m.a'));
    put('y', take('m.b'));
    done();
    expect([...dict.tensors.keys()]).toEqual(['p.x', 'p.y']);
    expect(dict.tensors.get('p.x')).toBe(a);
  });

  it.each([
    {
      name: 'a missing tensor',
      use: (io: ReturnType<typeof remap>) => io.take('m.c'),
      error: /test: missing tensor 'm.c'/,
    },
    {
      name: 'an unused tensor',
      use: (io: ReturnType<typeof remap>) => io.done(),
      error: /test: unused tensors: m.a,m.b/,
    },
  ])('throws CHECKPOINT_MISMATCH for $name', ({ use, error }) => {
    const io = remap(sd(), '', ['m.'], 'test');
    expect(() => use(io)).toThrow(
      expect.objectContaining({
        code: 'CHECKPOINT_MISMATCH',
        message: expect.stringMatching(error),
      }),
    );
  });
});
