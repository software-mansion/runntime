/** CPU pre- and post-processing around the yolo26 forward. One-to-one assignment
 *  means no NMS: box channels are ltrb grid units, class channels are logits. */

export interface Detection {
  x1: number;
  y1: number;
  x2: number;
  y2: number;
  score: number;
  classId: number;
  label: string;
}

export interface RawLevel {
  data: Float32Array;
  h: number;
  w: number;
  stride: number;
}

export interface ProtoData {
  data: Float32Array;
  h: number;
  w: number;
}

export interface SegDetection extends Detection {
  mask: Uint8Array; // maskW·maskH, row-major
  maskX: number;
  maskY: number;
  maskW: number;
  maskH: number;
}

export function preprocessRgba(
  rgba: Uint8ClampedArray,
  size: number,
  out?: Float32Array,
): Float32Array {
  const hw = size * size;
  out ??= new Float32Array(3 * hw);
  for (let i = 0; i < hw; i++) {
    out[i] = rgba[i * 4]! / 255;
    out[hw + i] = rgba[i * 4 + 1]! / 255;
    out[2 * hw + i] = rgba[i * 4 + 2]! / 255;
  }
  return out;
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

interface CellHit {
  levelIdx: number;
  cell: number;
  det: Detection;
}

function scanCells(
  levels: readonly RawLevel[],
  numClasses: number,
  confThreshold: number,
): CellHit[] {
  const logitThreshold = Math.log(confThreshold / (1 - confThreshold));
  const hits: CellHit[] = [];
  levels.forEach(({ data, h, w, stride }, levelIdx) => {
    const hw = h * w;
    for (let i = 0; i < hw; i++) {
      let bestLogit = -Infinity;
      let bestClass = 0;
      for (let c = 0; c < numClasses; c++) {
        const v = data[(4 + c) * hw + i]!;
        if (v > bestLogit) {
          bestLogit = v;
          bestClass = c;
        }
      }
      if (bestLogit < logitThreshold) continue;
      const score = sigmoid(bestLogit);
      const ax = (i % w) + 0.5;
      const ay = Math.floor(i / w) + 0.5;
      const l = data[i]!;
      const t = data[hw + i]!;
      const r = data[2 * hw + i]!;
      const b = data[3 * hw + i]!;
      hits.push({
        levelIdx,
        cell: i,
        det: {
          x1: (ax - l) * stride,
          y1: (ay - t) * stride,
          x2: (ax + r) * stride,
          y2: (ay + b) * stride,
          score,
          classId: bestClass,
          label: COCO_NAMES[bestClass] ?? `cls${bestClass}`,
        },
      });
    }
  });
  hits.sort((a, b) => b.det.score - a.det.score);
  return hits;
}

export function decodeDetections(
  levels: readonly RawLevel[],
  opts: { numClasses?: number; confThreshold?: number; maxDet?: number } = {},
): Detection[] {
  const { numClasses = 80, confThreshold = 0.3, maxDet = 300 } = opts;
  return scanCells(levels, numClasses, confThreshold)
    .slice(0, maxDet)
    .map((h) => h.det);
}

export function decodeSegmentations(
  levels: readonly RawLevel[],
  proto: ProtoData,
  opts: {
    numClasses?: number;
    numMasks?: number;
    confThreshold?: number;
    maxDet?: number;
    inputSize?: number;
  } = {},
): SegDetection[] {
  const {
    numClasses = 80,
    numMasks = 32,
    confThreshold = 0.3,
    maxDet = 300,
    inputSize = 640,
  } = opts;
  const protoStride = inputSize / proto.w;
  const phw = proto.h * proto.w;
  return scanCells(levels, numClasses, confThreshold)
    .slice(0, maxDet)
    .map(({ levelIdx, cell, det }) => {
      const { data, h, w } = levels[levelIdx]!;
      const hw = h * w;
      const coeffs = new Float32Array(numMasks);
      for (let c = 0; c < numMasks; c++) coeffs[c] = data[(4 + numClasses + c) * hw + cell]!;
      const px1 = Math.min(proto.w, Math.max(0, Math.floor(det.x1 / protoStride)));
      const py1 = Math.min(proto.h, Math.max(0, Math.floor(det.y1 / protoStride)));
      const px2 = Math.min(proto.w, Math.max(px1, Math.ceil(det.x2 / protoStride)));
      const py2 = Math.min(proto.h, Math.max(py1, Math.ceil(det.y2 / protoStride)));
      const maskW = px2 - px1;
      const maskH = py2 - py1;
      const mask = new Uint8Array(maskW * maskH);
      for (let y = 0; y < maskH; y++) {
        for (let x = 0; x < maskW; x++) {
          const pi = (py1 + y) * proto.w + (px1 + x);
          let v = 0;
          for (let c = 0; c < numMasks; c++) v += coeffs[c]! * proto.data[c * phw + pi]!;
          // Soft coverage: interiors saturate, only the boundary band lands
          // in between.
          mask[y * maskW + x] = Math.round(255 * sigmoid(v));
        }
      }
      return { ...det, mask, maskX: px1, maskY: py1, maskW, maskH };
    });
}

export interface Keypoint {
  x: number;
  y: number;
  v: number;
}

export interface PoseDetection extends Detection {
  kpts: Keypoint[]; // COCO_LANDMARKS order
}

export function decodePoses(
  levels: readonly RawLevel[],
  opts: {
    numClasses?: number;
    numKeypoints?: number;
    confThreshold?: number;
    maxDet?: number;
  } = {},
): PoseDetection[] {
  const { numClasses = 1, numKeypoints = 17, confThreshold = 0.3, maxDet = 300 } = opts;
  return scanCells(levels, numClasses, confThreshold)
    .slice(0, maxDet)
    .map(({ levelIdx, cell, det }) => {
      const { data, h, w, stride } = levels[levelIdx]!;
      const hw = h * w;
      const ax = (cell % w) + 0.5;
      const ay = Math.floor(cell / w) + 0.5;
      const kpts: Keypoint[] = [];
      for (let k = 0; k < numKeypoints; k++) {
        const o = (4 + numClasses + 3 * k) * hw + cell;
        kpts.push({
          x: (data[o]! + ax) * stride,
          y: (data[o + hw]! + ay) * stride,
          v: sigmoid(data[o + 2 * hw]!),
        });
      }
      return { ...det, kpts };
    });
}

/** The 17 COCO body landmarks, in model output order. */
export const COCO_LANDMARKS = [
  'nose',
  'leftEye',
  'rightEye',
  'leftEar',
  'rightEar',
  'leftShoulder',
  'rightShoulder',
  'leftElbow',
  'rightElbow',
  'leftWrist',
  'rightWrist',
  'leftHip',
  'rightHip',
  'leftKnee',
  'rightKnee',
  'leftAnkle',
  'rightAnkle',
] as const;
export type CocoLandmark = (typeof COCO_LANDMARKS)[number];

/** Limbs to draw between COCO keypoints; ear-to-shoulder is left out because it
 *  cuts across the face. */
export const COCO_SKELETON: readonly [number, number][] = [
  [15, 13],
  [13, 11],
  [16, 14],
  [14, 12],
  [11, 12],
  [5, 11],
  [6, 12],
  [5, 6],
  [5, 7],
  [6, 8],
  [7, 9],
  [8, 10],
  [1, 2],
  [0, 1],
  [0, 2],
  [1, 3],
  [2, 4],
];

export const COCO_NAMES = [
  'person',
  'bicycle',
  'car',
  'motorcycle',
  'airplane',
  'bus',
  'train',
  'truck',
  'boat',
  'traffic light',
  'fire hydrant',
  'stop sign',
  'parking meter',
  'bench',
  'bird',
  'cat',
  'dog',
  'horse',
  'sheep',
  'cow',
  'elephant',
  'bear',
  'zebra',
  'giraffe',
  'backpack',
  'umbrella',
  'handbag',
  'tie',
  'suitcase',
  'frisbee',
  'skis',
  'snowboard',
  'sports ball',
  'kite',
  'baseball bat',
  'baseball glove',
  'skateboard',
  'surfboard',
  'tennis racket',
  'bottle',
  'wine glass',
  'cup',
  'fork',
  'knife',
  'spoon',
  'bowl',
  'banana',
  'apple',
  'sandwich',
  'orange',
  'broccoli',
  'carrot',
  'hot dog',
  'pizza',
  'donut',
  'cake',
  'chair',
  'couch',
  'potted plant',
  'bed',
  'dining table',
  'toilet',
  'tv',
  'laptop',
  'mouse',
  'remote',
  'keyboard',
  'cell phone',
  'microwave',
  'oven',
  'toaster',
  'sink',
  'refrigerator',
  'book',
  'clock',
  'vase',
  'scissors',
  'teddy bear',
  'hair drier',
  'toothbrush',
] as const;
