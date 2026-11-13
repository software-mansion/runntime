import { describe, expect, it } from 'vitest';
import { rng, sampleToken } from '../../../src/zoo/decoder/sampling.ts';

const logits = Float32Array.from([0, 3, 1, 2.5, -Infinity, 2]);
const random = rng(1);

/** How often each id comes out of `draws` samples. */
function histogram(draw: () => number, draws: number): Map<number, number> {
  const h = new Map<number, number>();
  for (let i = 0; i < draws; i++) {
    const id = draw();
    h.set(id, (h.get(id) ?? 0) + 1);
  }
  return h;
}

describe('sampleToken', () => {
  it('is argmax at temperature 0, over the candidates when given', () => {
    expect(sampleToken(logits, undefined, 0, 64, 0.95, Math.random)).toBe(1);
    expect(sampleToken(logits, [0, 2, 5], 0, 64, 0.95, Math.random)).toBe(5);
  });

  it('is argmax with topK 1 at any temperature', () => {
    for (let i = 0; i < 20; i++)
      expect(sampleToken(logits, undefined, 5, 1, 1, Math.random)).toBe(1);
  });

  it('keeps the smallest nucleus that reaches topP', () => {
    // Softmax of [3, 2.5, 2, 1, 0]: 0.46, 0.28, 0.17, 0.06, 0.02. The first
    // two sum to 0.74, so topP 0.7 keeps ids 1 and 3, renormalized.
    const h = histogram(() => sampleToken(logits, undefined, 1, 64, 0.7, random), 20_000);
    expect([...h.keys()].sort()).toEqual([1, 3]);
    const p1 = Math.exp(3) / (Math.exp(3) + Math.exp(2.5));
    expect(h.get(1)! / 20_000).toBeCloseTo(p1, 1);
  });

  it('flattens the distribution with temperature', () => {
    // At temperature 100 the kept ids are near uniform.
    const h = histogram(() => sampleToken(logits, undefined, 100, 64, 1, random), 20_000);
    expect(h.has(4)).toBe(false);
    for (const id of [0, 1, 2, 3, 5]) expect(h.get(id)! / 20_000).toBeCloseTo(0.2, 1);
  });

  it('refuses when every candidate is masked', () => {
    const masked = new Float32Array(4).fill(-Infinity);
    expect(() => sampleToken(masked, undefined, 1, 64, 0.95, Math.random)).toThrow(/masked/);
    expect(() => sampleToken(masked, [0, 1], 0, 64, 0.95, Math.random)).toThrow(/masked/);
  });

  it('repeats a draw from the same seed', () => {
    const draw = (seed: number) => {
      const r = rng(seed);
      return Array.from({ length: 32 }, () => sampleToken(logits, undefined, 2, 64, 1, r));
    };
    expect(draw(7)).toEqual(draw(7));
    expect(draw(7)).not.toEqual(draw(8));
  });
});
