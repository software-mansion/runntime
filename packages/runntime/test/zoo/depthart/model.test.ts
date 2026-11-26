import { describe, expect, it } from 'vitest';
import { DepthartModel } from '../../../src/zoo/depthart/model.ts';
import { conv2dDtypes } from '../conv2dDtypes.ts';

describe('DepthartModel', () => {
  // The norm tables and the scan parameters feed kernels that read f32.
  it('builds its convs f16 and keeps every other parameter f32', () => {
    expect(conv2dDtypes(new DepthartModel())).toEqual({ conv2d: ['f16'], other: ['f32'] });
  });
});
