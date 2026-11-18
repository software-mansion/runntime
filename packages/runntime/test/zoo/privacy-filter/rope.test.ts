import { describe, expect, it } from 'vitest';
import { ropeConstants } from '../../../src/zoo/privacy-filter/rope.ts';

describe('ropeConstants', () => {
  // The score scale is sdpa's, so the kernel tables are the YaRN tables
  // with each pair duplicated and nothing else multiplied in.
  it('duplicates each YaRN cos and sin into the pair, unscaled', () => {
    const t = 3;
    const { cosE, sinE, rawCos, rawSin } = ropeConstants(t);
    const half = rawCos.length / t;
    for (let r = 0; r < t; r++) {
      for (let p = 0; p < half; p++) {
        for (const i of [2 * p, 2 * p + 1]) {
          expect(cosE[r * 2 * half + i]).toBe(rawCos[r * half + p]);
          expect(sinE[r * 2 * half + i]).toBe(rawSin[r * half + p]);
        }
      }
    }
  });
});
