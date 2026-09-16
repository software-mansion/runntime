/** 2D points, and the math that maps model-input coordinates back onto the
 *  original image. */

import type { ResizeMode } from '../image.ts';

export interface Point {
  readonly x: number;
  readonly y: number;
}

export interface Size {
  readonly width: number;
  readonly height: number;
}

/** One resize, described from the model input back to the original image. */
export interface ScalePointOptions {
  /** Model input size. */
  readonly from: Size;
  /** Original image size. */
  readonly to: Size;
  /** How the image was fitted into `from`. */
  readonly resizeMode: ResizeMode;
  /** `crop` only: how much of the centered square fills the input, measured
   *  from the middle. 1 is the whole square. Default 1. */
  readonly cropFraction?: number;
}

/** The resize as numbers: how many input pixels one image pixel takes, and
 *  where the image's top-left corner lands inside the input. */
export function resizeTransform(options: ScalePointOptions): {
  scaleX: number;
  scaleY: number;
  offsetX: number;
  offsetY: number;
} {
  const { from, to, resizeMode } = options;
  switch (resizeMode) {
    case 'stretch':
      return {
        scaleX: from.width / to.width,
        scaleY: from.height / to.height,
        offsetX: 0,
        offsetY: 0,
      };
    case 'letterbox': {
      const scale = Math.min(from.width / to.width, from.height / to.height);
      return {
        scaleX: scale,
        scaleY: scale,
        offsetX: (from.width - to.width * scale) / 2,
        offsetY: (from.height - to.height * scale) / 2,
      };
    }
    case 'crop': {
      // The centered square of the image, or its middle part, fills the
      // whole input, so the image's corner sits outside it (negative offset).
      const side = Math.min(to.width, to.height) * (options.cropFraction ?? 1);
      const scaleX = from.width / side;
      const scaleY = from.height / side;
      return {
        scaleX,
        scaleY,
        offsetX: (-(to.width - side) / 2) * scaleX,
        offsetY: (-(to.height - side) / 2) * scaleY,
      };
    }
  }
}

/** Maps a point from model-input coordinates to original-image coordinates. */
export function scalePoint(point: Point, options: ScalePointOptions): Point {
  const { scaleX, scaleY, offsetX, offsetY } = resizeTransform(options);
  return { x: (point.x - offsetX) / scaleX, y: (point.y - offsetY) / scaleY };
}

/** Straight-line distance between two points. */
export function distance(a: Point, b: Point): number {
  return Math.hypot(b.x - a.x, b.y - a.y);
}
