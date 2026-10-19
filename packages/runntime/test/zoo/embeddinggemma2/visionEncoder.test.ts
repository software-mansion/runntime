import { describe, expect, it } from 'vitest';
import { patchGrid } from '../../../src/zoo/embeddinggemma2/visionEncoder.ts';

const PATCH = 16;
const POOL = 3;

describe('patchGrid', () => {
  // [cols, rows] from Hugging Face's get_aspect_ratio_preserving_size in
  // transformers' Gemma4 image processor, as width / 16 and height / 16.
  it.each<[string, number, number, number, [number, number]]>([
    ['landscape', 640, 480, 280, [57, 42]],
    ['portrait', 480, 640, 280, [42, 57]],
    ['16:9', 1920, 1080, 280, [66, 36]],
    ['square, smallest budget', 224, 224, 70, [24, 24]],
    ['large photo, largest budget', 4032, 3024, 1120, [114, 84]],
    ['one pixel', 1, 1, 280, [48, 48]],
    ['almost square', 49, 47, 140, [36, 33]],
    ['thin', 37, 1013, 280, [9, 261]],
    ['a line, capped', 5000, 1, 280, [840, 3]],
    ['a column, capped', 1, 5000, 70, [3, 210]],
  ])('%s: %ix%i at %i tokens like Hugging Face', (_, width, height, tokens, want) => {
    expect(patchGrid(width, height, tokens, PATCH, POOL)).toEqual(want);
  });

  it('fits the budget in whole pool squares', () => {
    for (const tokens of [70, 140, 280, 560, 1120]) {
      for (const [width, height] of [
        [1, 1],
        [333, 777],
        [4000, 3],
        [1024, 1024],
      ] as const) {
        const [cols, rows] = patchGrid(width, height, tokens, PATCH, POOL);
        expect(cols % POOL).toBe(0);
        expect(rows % POOL).toBe(0);
        expect((cols * rows) / (POOL * POOL)).toBeLessThanOrEqual(tokens);
      }
    }
  });

  it('throws INVALID_ARGUMENT for a size that is not positive', () => {
    expect(() => patchGrid(0, 10, 280, PATCH, POOL)).toThrow(
      expect.objectContaining({ code: 'INVALID_ARGUMENT' }),
    );
  });
});
