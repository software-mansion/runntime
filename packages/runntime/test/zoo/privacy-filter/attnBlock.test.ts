import { describe, expect, it } from 'vitest';
import { EagerAttnBlock, HEAD } from '../../../src/zoo/privacy-filter/attnBlock.ts';
import { input } from '../../core/graph/values.ts';
import { attentionScales, standInWeights } from '../attentionScales.ts';

describe('EagerAttnBlock', () => {
  // The reference model's sdpa scales by 1/√headDim = 1/8.
  it('scales the scores by 1/√headDim', () => {
    const block = standInWeights(new EagerAttnBlock());
    const t = 5;
    const out = block.forward(input([t, 640]), input([t, HEAD]), input([t, HEAD]));
    expect(attentionScales(out)).toEqual([1 / Math.sqrt(HEAD)]);
  });
});
