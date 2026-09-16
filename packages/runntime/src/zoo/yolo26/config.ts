/** YOLO26 constants + compound scaling: a variant scales channel counts by
 *  `width` and block repeat counts by `depth`. */

export interface Yolo26Scale {
  depth: number;
  width: number;
  maxChannels: number;
}

export const YOLO26_SCALES = {
  n: { depth: 0.5, width: 0.25, maxChannels: 1024 },
  s: { depth: 0.5, width: 0.5, maxChannels: 1024 },
  m: { depth: 0.5, width: 1.0, maxChannels: 512 },
  l: { depth: 1.0, width: 1.0, maxChannels: 512 },
  x: { depth: 1.0, width: 1.5, maxChannels: 512 },
} as const satisfies Record<string, Yolo26Scale>;

export type Yolo26Variant = keyof typeof YOLO26_SCALES;

/** Capped before scaling, then rounded to a multiple of 8 to keep hwc4 tiles
 *  aligned. */
export function scaleChannels(baseChannels: number, scale: Yolo26Scale): number {
  const scaled = Math.min(baseChannels, scale.maxChannels) * scale.width;
  return Math.max(Math.round(scaled / 8) * 8, 8);
}

/** Only stacks scale; single-repeat layers stay at 1. */
export function scaleRepeats(baseRepeats: number, scale: Yolo26Scale): number {
  if (baseRepeats <= 1) return baseRepeats;
  return Math.max(Math.round(baseRepeats * scale.depth), 1);
}

export interface Yolo26Config {
  numClasses: number;
  inputSize: number;
  strides: readonly [number, number, number];
  regMax: number;
}

export const YOLO26_CONFIG: Yolo26Config = {
  numClasses: 80,
  inputSize: 640,
  strides: [8, 16, 32],
  regMax: 1,
};
