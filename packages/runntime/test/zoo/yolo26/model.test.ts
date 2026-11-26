import { describe, expect, it } from 'vitest';
import { Yolo26Model } from '../../../src/zoo/yolo26/model.ts';
import { conv2dDtypes } from '../conv2dDtypes.ts';

describe('Yolo26Model', () => {
  // The other parameters are the transposed conv of the segment head, whose
  // weight is f16 too, and its bias, which half() makes f16.
  it.each(['detect', 'segment', 'pose'] as const)('builds its %s convs f16', (task) => {
    expect(conv2dDtypes(new Yolo26Model('n', { task })).conv2d).toEqual(['f16']);
  });
});
