import { describe, expect, it } from 'vitest';
import { imageBuffer, type ImageFormat } from '../../../../../src/zoo/tasks/cv/image.ts';
import {
  createImagePreprocessor,
  type ImagePreprocessorOptions,
} from '../../../../../src/zoo/tasks/cv/utils/imagePreprocessor.ts';

const close = (want: number) => expect.closeTo(want, 5);

/** `width`×`height` image whose pixel bytes are `values`, row by row. */
function image(
  width: number,
  height: number,
  values: readonly number[],
  format: ImageFormat = 'gray',
) {
  return imageBuffer(new Uint8Array(values), width, height, format);
}

/** Normalization that keeps the bytes as they are, so expected values read
 *  straight off the input. */
const bytes = { alpha: 1, beta: 0 };

describe('createImagePreprocessor', () => {
  describe('same size', () => {
    it.each([
      {
        name: 'splits RGBA into R, G and B planes and drops alpha',
        format: 'rgba' as const,
        // Red, green, blue and (51, 102, 204), each with alpha 7
        data: [255, 0, 0, 7, 0, 255, 0, 7, 0, 0, 255, 7, 51, 102, 204, 7],
        want: [255, 0, 0, 51, 0, 255, 0, 102, 0, 0, 255, 204],
      },
      {
        name: 'reads BGR in reverse byte order',
        format: 'bgr' as const,
        data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        want: [3, 6, 9, 12, 2, 5, 8, 11, 1, 4, 7, 10],
      },
      {
        name: 'copies gray into all three planes',
        format: 'gray' as const,
        data: [10, 20, 30, 40],
        want: [10, 20, 30, 40, 10, 20, 30, 40, 10, 20, 30, 40],
      },
    ])('$name', ({ format, data, want }) => {
      const pre = createImagePreprocessor(
        { resizeMode: 'stretch', normalizeOpts: bytes },
        { width: 2, height: 2 },
      );
      expect([...pre.process(image(2, 2, data, format))]).toEqual(want.map(close));
    });

    it('scales bytes to [0, 1] by default', () => {
      const pre = createImagePreprocessor({ resizeMode: 'stretch' }, { width: 1, height: 1 });
      expect([...pre.process(image(1, 1, [51, 102, 204], 'rgb'))]).toEqual(
        [0.2, 0.4, 0.8].map(close),
      );
    });

    it('applies alpha and beta per channel', () => {
      const pre = createImagePreprocessor(
        { resizeMode: 'stretch', normalizeOpts: { alpha: [1, 2, 3], beta: [10, 20, 30] } },
        { width: 1, height: 1 },
      );
      expect([...pre.process(image(1, 1, [1, 1, 1], 'rgb'))]).toEqual([11, 22, 33]);
    });

    it('writes into the given array', () => {
      const pre = createImagePreprocessor(
        { resizeMode: 'stretch', normalizeOpts: bytes },
        { width: 1, height: 1 },
      );
      const out = new Float32Array(3);
      expect(pre.process(image(1, 1, [9]), out)).toBe(out);
      expect([...out]).toEqual([9, 9, 9]);
    });
  });

  describe('resize', () => {
    it.each([
      {
        name: 'stretch downsamples with linear blending',
        // Output pixel centers land halfway between source pairs
        options: { resizeMode: 'stretch' },
        size: { width: 2, height: 1 },
        input: image(4, 1, [0, 100, 200, 40]),
        want: [50, 120],
      },
      {
        name: 'stretch downsamples with nearest sampling',
        options: { resizeMode: 'stretch', interpolation: 'nearest' },
        size: { width: 2, height: 1 },
        input: image(4, 1, [0, 100, 200, 40]),
        want: [100, 40],
      },
      {
        name: 'stretch upsamples and clamps at the edges',
        // Centers at 0.25, 0.75, 1.25, 1.75 source px; the outer two clamp
        options: { resizeMode: 'stretch' },
        size: { width: 4, height: 1 },
        input: image(2, 1, [0, 100]),
        want: [0, 25, 75, 100],
      },
      {
        name: 'letterbox pads the sides with padValue',
        options: { resizeMode: 'letterbox', padValue: 7 },
        size: { width: 4, height: 2 },
        input: image(2, 2, [10, 20, 30, 40]),
        want: [7, 10, 20, 7, 7, 30, 40, 7],
      },
      {
        name: 'letterbox pads with 114 by default',
        options: { resizeMode: 'letterbox' },
        size: { width: 1, height: 2 },
        input: image(1, 1, [5]),
        // The 1 px image sits at row 0, half a pixel of bar on each side
        want: [5, 114],
      },
      {
        name: 'crop keeps the centered square',
        options: { resizeMode: 'crop' },
        size: { width: 2, height: 2 },
        input: image(4, 2, [1, 2, 3, 4, 5, 6, 7, 8]),
        want: [2, 3, 6, 7],
      },
      {
        name: 'crop with cropFraction keeps the middle of the square',
        options: { resizeMode: 'crop', cropFraction: 0.5 },
        size: { width: 2, height: 2 },
        input: image(
          4,
          4,
          Array.from({ length: 16 }, (_, i) => i),
        ),
        want: [5, 6, 9, 10],
      },
      {
        name: 'crop downsamples by averaging 2×2 blocks',
        options: { resizeMode: 'crop' },
        size: { width: 2, height: 2 },
        input: image(
          4,
          4,
          Array.from({ length: 16 }, (_, i) => i),
        ),
        want: [2.5, 4.5, 10.5, 12.5],
      },
    ] satisfies {
      name: string;
      options: Omit<ImagePreprocessorOptions, 'normalizeOpts'>;
      size: { width: number; height: number };
      input: ReturnType<typeof image>;
      want: number[];
    }[])('$name', ({ options, size, input, want }) => {
      const pre = createImagePreprocessor({ ...options, normalizeOpts: bytes }, size);
      const out = pre.process(input);
      // Gray input: every plane holds the same values
      expect([...out.subarray(0, want.length)]).toEqual(want.map(close));
      expect([...out.subarray(want.length, 2 * want.length)]).toEqual(want.map(close));
      expect([...out.subarray(2 * want.length)]).toEqual(want.map(close));
    });

    it('normalizes the padding like a pixel', () => {
      const pre = createImagePreprocessor(
        { resizeMode: 'letterbox', padValue: 10, normalizeOpts: { alpha: 2, beta: [1, 2, 3] } },
        { width: 1, height: 2 },
      );
      expect([...pre.process(image(1, 1, [0]))]).toEqual([1, 21, 2, 22, 3, 23]);
    });
  });

  it('scaleOptions maps the model input back onto the image', () => {
    const pre = createImagePreprocessor(
      { resizeMode: 'crop', cropFraction: 0.8 },
      { width: 224, height: 224 },
    );
    expect(pre.scaleOptions({ width: 640, height: 480 })).toEqual({
      from: { width: 224, height: 224 },
      to: { width: 640, height: 480 },
      resizeMode: 'crop',
      cropFraction: 0.8,
    });
  });

  it('gpuArgs carries the same fit, normalization and padding', () => {
    const pre = createImagePreprocessor(
      {
        resizeMode: 'letterbox',
        padValue: 10,
        normalizeOpts: { alpha: [2, 3, 4], beta: [1, 0, -1] },
      },
      { width: 4, height: 3 },
    );
    expect(pre.gpuArgs).toEqual({
      mode: 'letterbox',
      cropFraction: 1,
      outW: 4,
      outH: 3,
      alpha: [2, 3, 4],
      beta: [1, 0, -1],
      // 10 * alpha + beta, per channel
      pad: [21, 30, 39],
    });
  });

  describe('rejects', () => {
    it.each([
      { name: 'a zero model size', options: {}, size: { width: 0, height: 2 } },
      { name: 'a fractional model size', options: {}, size: { width: 2.5, height: 2 } },
      { name: 'cropFraction 0', options: { cropFraction: 0 }, size: { width: 2, height: 2 } },
      {
        name: 'cropFraction above 1',
        options: { cropFraction: 1.5 },
        size: { width: 2, height: 2 },
      },
      {
        name: 'alpha with two values',
        options: { normalizeOpts: { alpha: [1, 2] } },
        size: { width: 2, height: 2 },
      },
    ])('$name', ({ options, size }) => {
      expect(() => createImagePreprocessor({ resizeMode: 'stretch', ...options }, size)).toThrow(
        expect.objectContaining({ code: 'INVALID_ARGUMENT' }),
      );
    });

    it('an out array of the wrong length', () => {
      const pre = createImagePreprocessor({ resizeMode: 'stretch' }, { width: 2, height: 2 });
      expect(() => pre.process(image(2, 2, [1, 2, 3, 4]), new Float32Array(3))).toThrow(
        expect.objectContaining({ code: 'INVALID_ARGUMENT' }),
      );
    });

    it('an image whose bytes do not match its size', () => {
      const pre = createImagePreprocessor({ resizeMode: 'stretch' }, { width: 2, height: 2 });
      const input = { ...image(2, 2, [1, 2, 3, 4]), width: 3 };
      expect(() => pre.process(input)).toThrow(
        expect.objectContaining({ code: 'INVALID_ARGUMENT' }),
      );
    });
  });
});
