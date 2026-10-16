import { describe, expect, it } from 'vitest';
import {
  decodeBox,
  scaleBox,
  type BoundingBox,
  type BoxFormat,
} from '../../../../../src/zoo/tasks/cv/ops/box.ts';
import type { ScalePointOptions } from '../../../../../src/zoo/tasks/cv/ops/point.ts';

// A 640×480 photo fitted into a 320×320 model input
const from = { width: 320, height: 320 };
const to = { width: 640, height: 480 };

const close = (want: number) => expect.closeTo(want, 6);

describe('decodeBox', () => {
  it.each([
    { format: 'xyxy', want: { format: 'xyxy', xmin: 1, ymin: 2, xmax: 3, ymax: 4 } },
    { format: 'xywh', want: { format: 'xywh', xmin: 1, ymin: 2, w: 3, h: 4 } },
    { format: 'cxcywh', want: { format: 'cxcywh', cx: 1, cy: 2, w: 3, h: 4 } },
  ] as const)('reads (1, 2, 3, 4) as $format', ({ format, want }) => {
    expect(decodeBox([1, 2, 3, 4], format)).toEqual(want);
  });

  it('rejects an unknown format', () => {
    // A format name that only plain JS callers can pass
    const format = 'yxyx' as BoxFormat;
    expect(() => decodeBox([1, 2, 3, 4], format)).toThrow(
      expect.objectContaining({ code: 'INVALID_ARGUMENT' }),
    );
  });
});

describe('scaleBox', () => {
  // Corners and centers move like points; sizes only divide by the scale.
  it.each([
    {
      name: 'a letterboxed xyxy box covering the image',
      opts: { from, to, resizeMode: 'letterbox' },
      box: { format: 'xyxy', xmin: 0, ymin: 40, xmax: 320, ymax: 280 },
      want: { format: 'xyxy', xmin: 0, ymin: 0, xmax: 640, ymax: 480 },
    },
    {
      name: 'a letterboxed xywh box, whose size ignores the bar',
      opts: { from, to, resizeMode: 'letterbox' },
      box: { format: 'xywh', xmin: 0, ymin: 40, w: 320, h: 240 },
      want: { format: 'xywh', xmin: 0, ymin: 0, w: 640, h: 480 },
    },
    {
      name: 'a letterboxed cxcywh box',
      opts: { from, to, resizeMode: 'letterbox' },
      box: { format: 'cxcywh', cx: 160, cy: 160, w: 320, h: 240 },
      want: { format: 'cxcywh', cx: 320, cy: 240, w: 640, h: 480 },
    },
    {
      name: 'a stretched xywh box, scaled per axis',
      opts: { from, to, resizeMode: 'stretch' },
      box: { format: 'xywh', xmin: 10, ymin: 20, w: 30, h: 40 },
      want: { format: 'xywh', xmin: 20, ymin: 30, w: 60, h: 60 },
    },
    {
      name: 'a cropped cxcywh box',
      opts: { from, to, resizeMode: 'crop' },
      box: { format: 'cxcywh', cx: 160, cy: 160, w: 32, h: 32 },
      want: { format: 'cxcywh', cx: 320, cy: 240, w: 48, h: 48 },
    },
  ] satisfies {
    name: string;
    opts: ScalePointOptions;
    box: BoundingBox;
    want: BoundingBox;
  }[])('maps $name', ({ opts, box, want }) => {
    const { format, ...fields } = want;
    const closeFields = Object.fromEntries(
      Object.entries(fields).map(([key, value]) => [key, close(value)]),
    );
    expect(scaleBox(box, opts)).toEqual({ format, ...closeFields });
  });

  it('gives the same rectangle in every format', () => {
    // The input square (40, 80)–(200, 240) under a half crop: the 240×240
    // middle of the image starts at (200, 120) and is 4/3 input px per px.
    const opts = { from, to, resizeMode: 'crop', cropFraction: 0.5 } as const;
    expect(scaleBox({ format: 'xyxy', xmin: 40, ymin: 80, xmax: 200, ymax: 240 }, opts)).toEqual({
      format: 'xyxy',
      xmin: close(230),
      ymin: close(180),
      xmax: close(350),
      ymax: close(300),
    });
    expect(scaleBox({ format: 'xywh', xmin: 40, ymin: 80, w: 160, h: 160 }, opts)).toEqual({
      format: 'xywh',
      xmin: close(230),
      ymin: close(180),
      w: close(120),
      h: close(120),
    });
    expect(scaleBox({ format: 'cxcywh', cx: 120, cy: 160, w: 160, h: 160 }, opts)).toEqual({
      format: 'cxcywh',
      cx: close(290),
      cy: close(240),
      w: close(120),
      h: close(120),
    });
  });
});
