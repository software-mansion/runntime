import { describe, expect, it } from 'vitest';
import { imageBuffer } from '../../../../../src/zoo/tasks/cv/image.ts';
import { resizeBicubic } from '../../../../../src/zoo/tasks/cv/utils/resizeBicubic.ts';

/** A `width`×`height` RGB test pattern with sharp steps, so the cubic
 *  kernel overshoots and the result must clamp. */
function pattern(width: number, height: number): Uint8Array {
  const data = new Uint8Array(width * height * 3);
  for (let y = 0; y < height; y++) {
    for (let x = 0; x < width; x++) {
      for (let c = 0; c < 3; c++) {
        data[(y * width + x) * 3 + c] = (17 * x * x + 31 * y + 59 * c + 7 * x * y) % 256;
      }
    }
  }
  return data;
}

/** `rgb` bytes in another format, alpha 9 where it has one. */
function convert(rgb: Uint8Array, format: 'rgba' | 'bgr' | 'bgra'): Uint8Array {
  const n = rgb.length / 3;
  const ch = format === 'bgr' ? 3 : 4;
  const out = new Uint8Array(n * ch);
  for (let i = 0; i < n; i++) {
    const [r, g, b] = [rgb[3 * i]!, rgb[3 * i + 1]!, rgb[3 * i + 2]!];
    const px = format === 'rgba' ? [r, g, b] : [b, g, r];
    out.set(ch === 4 ? [...px, 9] : px, ch * i);
  }
  return out;
}

describe('resizeBicubic', () => {
  // Pillow 11.3 output for the same pattern resized with Image.BICUBIC, as
  // [3, height, width].
  it.each<{ name: string; from: [number, number]; to: [number, number]; want: number[] }>([
    {
      name: 'shrinks',
      from: [9, 7],
      to: [4, 3],
      want: [
        45, 108, 142, 141, 128, 135, 142, 140, 176, 139, 138, 138, 107, 123, 134, 145, 153, 122,
        151, 116, 133, 114, 101, 91, 161, 126, 116, 91, 168, 126, 157, 138, 86, 130, 137, 130,
      ],
    },
    {
      name: 'grows',
      from: [4, 3],
      to: [6, 5],
      want: [
        0, 2, 20, 54, 112, 155, 9, 15, 35, 70, 137, 187, 29, 39, 62, 101, 165, 211, 50, 63, 91, 141,
        111, 76, 62, 76, 107, 164, 73, 0, 56, 61, 79, 112, 181, 234, 67, 74, 95, 135, 143, 139, 88,
        98, 124, 175, 89, 0, 109, 122, 151, 206, 117, 23, 121, 135, 166, 222, 142, 54, 115, 120,
        141, 187, 94, 0, 126, 133, 157, 214, 119, 19, 147, 157, 183, 234, 148, 56, 168, 190, 189,
        109, 94, 104, 180, 210, 189, 23, 54, 131,
      ],
    },
    {
      name: 'grows one side and shrinks the other',
      from: [5, 6],
      to: [8, 2],
      want: [
        34, 42, 63, 109, 130, 119, 100, 88, 117, 137, 166, 177, 148, 92, 137, 177, 92, 105, 128,
        153, 136, 87, 120, 152, 181, 173, 148, 85, 85, 148, 149, 139, 158, 172, 184, 160, 119, 80,
        136, 182, 170, 150, 118, 99, 109, 144, 138, 128,
      ],
    },
    {
      name: 'shrinks by more than five',
      from: [16, 5],
      to: [3, 2],
      want: [
        94, 131, 96, 139, 138, 109, 119, 143, 133, 142, 128, 125, 137, 111, 141, 150, 148, 150,
      ],
    },
  ])('$name like Pillow', ({ from: [w, h], to: [width, height], want }) => {
    const out = resizeBicubic(imageBuffer(pattern(w, h), w, h, 'rgb'), width, height);
    expect(Array.from(out)).toEqual(want);
  });

  it('keeps an image of the same size', () => {
    const rgb = pattern(5, 4);
    const out = resizeBicubic(imageBuffer(rgb, 5, 4, 'rgb'), 5, 4);
    const planes = [0, 1, 2].flatMap((c) => Array.from({ length: 20 }, (_, i) => rgb[3 * i + c]));
    expect(Array.from(out)).toEqual(planes);
  });

  it.each([
    [7, 5, 3, 2],
    [3, 2, 7, 5],
  ])('keeps a flat image flat from %ix%i to %ix%i', (w, h, width, height) => {
    const data = new Uint8Array(w * h * 3).fill(77);
    const out = resizeBicubic(imageBuffer(data, w, h, 'rgb'), width, height);
    expect(out.length).toBe(3 * width * height);
    expect(new Set(out)).toEqual(new Set([77]));
  });

  it.each(['rgba', 'bgr', 'bgra'] as const)('reads %s like rgb', (format) => {
    const rgb = pattern(9, 7);
    const want = resizeBicubic(imageBuffer(rgb, 9, 7, 'rgb'), 4, 3);
    expect(resizeBicubic(imageBuffer(convert(rgb, format), 9, 7, format), 4, 3)).toEqual(want);
  });

  it('reads gray into three equal planes', () => {
    const gray = pattern(9, 7).filter((_, i) => i % 3 === 0);
    const asRgb = new Uint8Array(gray.length * 3).map((_, i) => gray[Math.floor(i / 3)]!);
    const want = resizeBicubic(imageBuffer(asRgb, 9, 7, 'rgb'), 4, 3);
    expect(resizeBicubic(imageBuffer(gray, 9, 7, 'gray'), 4, 3)).toEqual(want);
  });

  const invalid = expect.objectContaining({ code: 'INVALID_ARGUMENT' });

  it.each([
    [0, 2],
    [2, -1],
    [2.5, 2],
  ])('throws INVALID_ARGUMENT for a %sx%s output', (width, height) => {
    const image = imageBuffer(pattern(2, 2), 2, 2, 'rgb');
    expect(() => resizeBicubic(image, width, height)).toThrow(invalid);
  });

  it('throws INVALID_ARGUMENT when the bytes do not fit the size', () => {
    const image = { ...imageBuffer(pattern(2, 2), 2, 2, 'rgb'), width: 3 };
    expect(() => resizeBicubic(image, 1, 1)).toThrow(invalid);
  });
});
