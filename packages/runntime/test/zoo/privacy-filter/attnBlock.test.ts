import { describe, expect, it } from 'vitest';
import { EagerAttnBlock, HEAD } from '../../../src/zoo/privacy-filter/attnBlock.ts';
import { input } from '../../core/graph/values.ts';
import { attentionScales, standInWeights } from '../attentionScales.ts';
import { ropeLayouts } from '../ropeLayouts.ts';

describe('EagerAttnBlock', () => {
  // The reference model's sdpa scales by 1/√headDim = 1/8.
  it('scales the scores by 1/√headDim', () => {
    const block = standInWeights(new EagerAttnBlock());
    const t = 5;
    const out = block.forward(input([t, 640]), input([t, HEAD]), input([t, HEAD]));
    expect(attentionScales(out)).toEqual([1 / Math.sqrt(HEAD)]);
  });

  // HF's OpenAIPrivacyFilter rope takes x[..., ::2] and x[..., 1::2]: pairs
  // (2p, 2p + 1), unlike gpt-oss, which splits a head in half.
  it('ropes q and k in the interleaved layout', () => {
    const block = standInWeights(new EagerAttnBlock());
    const t = 5;
    const out = block.forward(input([t, 640]), input([t, HEAD]), input([t, HEAD]));
    expect(ropeLayouts(out)).toEqual(Array(2).fill({ layout: 'interleaved', ropesPerHead: 1 }));
  });
});
