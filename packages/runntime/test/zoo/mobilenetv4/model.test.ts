import { describe, expect, it } from 'vitest';
import { MobileNetV4Model } from '../../../src/zoo/mobilenetv4/model.ts';
import { conv2dDtypes } from '../conv2dDtypes.ts';

describe('MobileNetV4Model', () => {
  it('builds its convs and its classifier f16', () => {
    expect(conv2dDtypes(new MobileNetV4Model())).toEqual({ conv2d: ['f16'], other: ['f16'] });
  });
});
