import { describe, expect, it } from 'vitest';
import { MOONSHINE_TINY } from '../../../src/zoo/moonshine/config.ts';
import { buildRopeTables } from '../../../src/zoo/moonshine/rope.ts';

describe('buildRopeTables', () => {
  // The score scale is sdpa's, so every rotated pair is a pure rotation and
  // the dims past rotaryDim pass through as they are.
  it('builds unscaled rotations, and the identity past rotaryDim', () => {
    const cfg = MOONSHINE_TINY;
    const positions = 5;
    const { cos, sin } = buildRopeTables(positions, cfg);
    for (let t = 0; t < positions; t++) {
      for (let c = 0; c < cfg.headDim; c++) {
        const i = t * cfg.headDim + c;
        if (c < cfg.rotaryDim) {
          expect(cos[i]! ** 2 + sin[i]! ** 2).toBeCloseTo(1, 6);
        } else {
          expect([cos[i], sin[i]]).toEqual([1, 0]);
        }
      }
    }
  });
});
