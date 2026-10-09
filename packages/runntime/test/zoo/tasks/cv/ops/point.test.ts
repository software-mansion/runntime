import { describe, expect, it } from 'vitest';
import {
  distance,
  resizeTransform,
  scalePoint,
  type ScalePointOptions,
} from '../../../../../src/zoo/tasks/cv/ops/point.ts';

// A 640×480 photo fitted into a 320×320 model input
const from = { width: 320, height: 320 };
const landscape = { width: 640, height: 480 };
const portrait = { width: 480, height: 640 };

const close = (want: number) => expect.closeTo(want, 6);

describe('resizeTransform', () => {
  it.each([
    {
      name: 'stretch scales each axis on its own',
      opts: { from, to: landscape, resizeMode: 'stretch' },
      want: { scaleX: 1 / 2, scaleY: 2 / 3, offsetX: 0, offsetY: 0 },
    },
    {
      name: 'letterbox fits the long side and bars the short one',
      // 640·0.5 = 320 wide, 480·0.5 = 240 tall, so 40 px bars above and below
      opts: { from, to: landscape, resizeMode: 'letterbox' },
      want: { scaleX: 1 / 2, scaleY: 1 / 2, offsetX: 0, offsetY: 40 },
    },
    {
      name: 'letterbox bars the sides of a portrait image',
      opts: { from, to: portrait, resizeMode: 'letterbox' },
      want: { scaleX: 1 / 2, scaleY: 1 / 2, offsetX: 40, offsetY: 0 },
    },
    {
      name: 'crop fills the input with the centered square',
      // The 480×480 square starts 80 px in: −80 · 2/3 in input pixels
      opts: { from, to: landscape, resizeMode: 'crop' },
      want: { scaleX: 2 / 3, scaleY: 2 / 3, offsetX: -160 / 3, offsetY: 0 },
    },
    {
      name: 'crop with cropFraction fills it with the middle of that square',
      // The 240×240 middle starts at (200, 120): times 4/3 in input pixels
      opts: { from, to: landscape, resizeMode: 'crop', cropFraction: 0.5 },
      want: { scaleX: 4 / 3, scaleY: 4 / 3, offsetX: -800 / 3, offsetY: -160 },
    },
    {
      name: 'stretch to the same size does nothing',
      opts: { from: landscape, to: landscape, resizeMode: 'stretch' },
      want: { scaleX: 1, scaleY: 1, offsetX: 0, offsetY: 0 },
    },
    {
      name: 'letterbox to the same size does nothing',
      opts: { from: landscape, to: landscape, resizeMode: 'letterbox' },
      want: { scaleX: 1, scaleY: 1, offsetX: 0, offsetY: 0 },
    },
  ] satisfies { name: string; opts: ScalePointOptions; want: object }[])(
    '$name',
    ({ opts, want }) => {
      const got = resizeTransform(opts);
      expect(got).toEqual({
        scaleX: close(want.scaleX),
        scaleY: close(want.scaleY),
        offsetX: close(want.offsetX),
        offsetY: close(want.offsetY),
      });
    },
  );
});

describe('scalePoint', () => {
  it.each([
    { resizeMode: 'stretch', point: { x: 0, y: 0 }, want: { x: 0, y: 0 } },
    { resizeMode: 'stretch', point: { x: 320, y: 320 }, want: { x: 640, y: 480 } },
    { resizeMode: 'letterbox', point: { x: 0, y: 40 }, want: { x: 0, y: 0 } },
    { resizeMode: 'letterbox', point: { x: 160, y: 160 }, want: { x: 320, y: 240 } },
    { resizeMode: 'letterbox', point: { x: 320, y: 280 }, want: { x: 640, y: 480 } },
    // A point on the top bar lands above the image
    { resizeMode: 'letterbox', point: { x: 0, y: 0 }, want: { x: 0, y: -80 } },
    { resizeMode: 'crop', point: { x: 0, y: 0 }, want: { x: 80, y: 0 } },
    { resizeMode: 'crop', point: { x: 320, y: 320 }, want: { x: 560, y: 480 } },
  ] as const)('$resizeMode maps input $point to image $want', ({ resizeMode, point, want }) => {
    const got = scalePoint(point, { from, to: landscape, resizeMode });
    expect(got).toEqual({ x: close(want.x), y: close(want.y) });
  });

  it('maps the corners of a partial crop to the middle of the image', () => {
    const opts = { from, to: landscape, resizeMode: 'crop', cropFraction: 0.5 } as const;
    expect(scalePoint({ x: 0, y: 0 }, opts)).toEqual({ x: close(200), y: close(120) });
    expect(scalePoint({ x: 320, y: 320 }, opts)).toEqual({ x: close(440), y: close(360) });
  });
});

describe('distance', () => {
  it.each([
    { a: { x: 0, y: 0 }, b: { x: 3, y: 4 }, want: 5 },
    { a: { x: 1, y: 1 }, b: { x: 1, y: 1 }, want: 0 },
    { a: { x: 3, y: 4 }, b: { x: 0, y: 0 }, want: 5 },
  ])('from $a to $b is $want', ({ a, b, want }) => {
    expect(distance(a, b)).toBe(want);
  });
});
