/** Bounding boxes in three formats, plus scaling back onto the original image. */

import { resizeTransform, scalePoint, type ScalePointOptions } from './point.ts';

/** Each box format and the fields it carries. */
export type BoxMap = Readonly<{
  xyxy: Readonly<{ xmin: number; ymin: number; xmax: number; ymax: number }>;
  xywh: Readonly<{ xmin: number; ymin: number; w: number; h: number }>;
  cxcywh: Readonly<{ cx: number; cy: number; w: number; h: number }>;
}>;

export type BoxFormat = keyof BoxMap;

/** A box tagged with its format. */
export type BoundingBox<F extends BoxFormat = BoxFormat> = F extends BoxFormat
  ? { readonly format: F } & BoxMap[F]
  : never;

export type ScaleBoxOptions = ScalePointOptions;

/** Builds a box from four numbers in the given format. */
export function decodeBox<F extends BoxFormat>(
  tuple: readonly [number, number, number, number],
  format: F,
): BoundingBox<F> {
  const [a, b, c, d] = tuple;
  switch (format) {
    case 'xyxy':
      return { format: 'xyxy', xmin: a, ymin: b, xmax: c, ymax: d } as BoundingBox<F>;
    case 'xywh':
      return { format: 'xywh', xmin: a, ymin: b, w: c, h: d } as BoundingBox<F>;
    case 'cxcywh':
      return { format: 'cxcywh', cx: a, cy: b, w: c, h: d } as BoundingBox<F>;
    default:
      throw new Error(`unknown box format ${String(format)}`);
  }
}

/** Maps a box from model-input coordinates to original-image coordinates. */
export function scaleBox<F extends BoxFormat>(
  box: BoundingBox<F>,
  options: ScaleBoxOptions,
): BoundingBox<F> {
  const { scaleX, scaleY } = resizeTransform(options);
  switch (box.format) {
    case 'xyxy': {
      const min = scalePoint({ x: box.xmin, y: box.ymin }, options);
      const max = scalePoint({ x: box.xmax, y: box.ymax }, options);
      return {
        format: 'xyxy',
        xmin: min.x,
        ymin: min.y,
        xmax: max.x,
        ymax: max.y,
      } as BoundingBox<F>;
    }
    case 'xywh': {
      const min = scalePoint({ x: box.xmin, y: box.ymin }, options);
      return {
        format: 'xywh',
        xmin: min.x,
        ymin: min.y,
        w: box.w / scaleX,
        h: box.h / scaleY,
      } as BoundingBox<F>;
    }
    case 'cxcywh': {
      const center = scalePoint({ x: box.cx, y: box.cy }, options);
      return {
        format: 'cxcywh',
        cx: center.x,
        cy: center.y,
        w: box.w / scaleX,
        h: box.h / scaleY,
      } as BoundingBox<F>;
    }
  }
}
