import { d } from 'typegpu';
import { describe, expect, it, onTestFinished } from 'vitest';
import type { F32Buffer, U32Buffer } from '../../../../src/core/kernels/common.ts';
import {
  createDecodeDetections,
  detectionList,
  instanceSegmentation,
  keypointDetection,
  objectDetection,
  type DecodeTask,
} from '../../../../src/core/kernels/detect/decodeDetections.ts';
import type { FitMode } from '../../../../src/core/kernels/image/imageToChw.ts';
import { expectClose, gpu, randomF32 } from '../../../gpu.ts';

const { root } = await gpu();

interface Case {
  name: string;
  task: DecodeTask;
  inputSize: number;
  numClasses: number;
  numExtra: number;
  confidenceThreshold: number;
  maxDetections: number;
  mode: FitMode;
  cropFraction?: number;
  image: [number, number];
  maskSize?: number;
  /** Rounds the logits to this many steps per unit, so cells tie. */
  quantize?: number;
}

interface Det {
  box: number[];
  confidence: number;
  classId: number;
  /** x, y, confidence per landmark. */
  landmarks: number[];
  mask: number[];
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

/** The decode from its definition: every cell's best class (lowest index on
 *  a tie), kept at sigmoid >= threshold, sorted by score then by cell in
 *  level order; boxes are the ltrb distances around the cell center times
 *  the stride, mapped from input to image pixels by the fit and cut at the
 *  image edge; landmarks are offsets from the cell center; a mask value is
 *  the bilinear mix of the four nearest prototype pixels' sigmoid(coef ·
 *  proto) at the box point it stands for. */
function decodeRef(c: Case, levels: Float32Array[], proto: Float32Array | undefined): Det[] {
  const { inputSize: size, numClasses, numExtra, mode, maskSize = 0 } = c;
  const [w, h] = c.image;
  let sx = size / w;
  let sy = size / h;
  if (mode === 'letterbox') sx = sy = Math.min(sx, sy);
  if (mode === 'crop') sx = sy = size / (Math.min(w, h) * (c.cropFraction ?? 1));
  const ox = (size - w * sx) / 2;
  const oy = (size - h * sy) / 2;
  const toImage = (x: number, y: number) => [(x - ox) / sx, (y - oy) / sy] as const;
  const clamp = (v: number, hi: number) => Math.min(hi, Math.max(0, v));

  const hits: { logit: number; order: number; level: number; cell: number; cls: number }[] = [];
  let order = 0;
  [8, 16, 32].forEach((stride, level) => {
    const g = size / stride;
    for (let cell = 0; cell < g * g; cell++, order++) {
      let best = -Infinity;
      let cls = 0;
      for (let k = 0; k < numClasses; k++) {
        const v = levels[level]![(4 + k) * g * g + cell]!;
        if (v > best) [best, cls] = [v, k];
      }
      if (sigmoid(best) >= c.confidenceThreshold)
        hits.push({ logit: best, order, level, cell, cls });
    }
  });
  hits.sort((a, b) => b.logit - a.logit || a.order - b.order);

  return hits.slice(0, c.maxDetections).map(({ logit, level, cell, cls }) => {
    const stride = [8, 16, 32][level]!;
    const g = size / stride;
    const at = (ch: number) => levels[level]![ch * g * g + cell]!;
    const ax = (cell % g) + 0.5;
    const ay = Math.floor(cell / g) + 0.5;
    const [x1, y1] = toImage((ax - at(0)) * stride, (ay - at(1)) * stride);
    const [x2, y2] = toImage((ax + at(2)) * stride, (ay + at(3)) * stride);
    const box = [clamp(x1, w), clamp(y1, h), clamp(x2, w), clamp(y2, h)];
    const landmarks: number[] = [];
    const mask: number[] = [];
    if (c.task === 'pose') {
      for (let k = 0; k < numExtra; k++) {
        const ch = 4 + numClasses + 3 * k;
        const [x, y] = toImage((at(ch) + ax) * stride, (at(ch + 1) + ay) * stride);
        landmarks.push(x, y, sigmoid(at(ch + 2)));
      }
    }
    if (c.task === 'segment') {
      const pw = size / 4;
      const coverage = (px: number, py: number) => {
        const x = Math.min(pw - 1, Math.max(0, px));
        const y = Math.min(pw - 1, Math.max(0, py));
        let s = 0;
        for (let m = 0; m < numExtra; m++)
          s += at(4 + numClasses + m) * proto![m * pw * pw + y * pw + x]!;
        return sigmoid(s);
      };
      for (let j = 0; j < maskSize; j++) {
        for (let i = 0; i < maskSize; i++) {
          const bx = box[0]! + ((i + 0.5) / maskSize) * (box[2]! - box[0]!);
          const by = box[1]! + ((j + 0.5) / maskSize) * (box[3]! - box[1]!);
          const u = (bx * sx + ox) / 4 - 0.5;
          const v = (by * sy + oy) / 4 - 0.5;
          const x0 = Math.floor(u);
          const y0 = Math.floor(v);
          const fx = u - x0;
          const fy = v - y0;
          const top = coverage(x0, y0) * (1 - fx) + coverage(x0 + 1, y0) * fx;
          const bot = coverage(x0, y0 + 1) * (1 - fx) + coverage(x0 + 1, y0 + 1) * fx;
          mask.push(top * (1 - fy) + bot * fy);
        }
      }
    }
    return { box, confidence: sigmoid(logit), classId: cls, landmarks, mask };
  });
}

function storage(data: Float32Array | Uint32Array): GPUBuffer {
  const buf = root.device.createBuffer({
    size: data.byteLength,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
  });
  // A copy: writeBuffer takes no view of a possibly shared buffer.
  root.device.queue.writeBuffer(
    buf,
    0,
    new Uint8Array(data.buffer, data.byteOffset, data.byteLength).slice(),
  );
  onTestFinished(() => buf.destroy());
  return buf;
}

const f32Buffer = (data: Float32Array) =>
  root.createBuffer(d.arrayOf(d.f32, data.length), storage(data)).$usage('storage') as F32Buffer;

const CASES: Case[] = [
  // 84 cells at size 64, more than one 64-wide rank tile once most pass.
  {
    name: 'detect, letterbox, every cell',
    task: 'detect',
    inputSize: 64,
    numClasses: 3,
    numExtra: 0,
    confidenceThreshold: 0,
    maxDetections: 84,
    mode: 'letterbox',
    image: [100, 50],
  },
  {
    name: 'detect, letterbox, a threshold',
    task: 'detect',
    inputSize: 64,
    numClasses: 3,
    numExtra: 0,
    confidenceThreshold: 0.8,
    maxDetections: 84,
    mode: 'letterbox',
    image: [50, 100],
  },
  {
    name: 'detect, maxDetections cuts the list',
    task: 'detect',
    inputSize: 64,
    numClasses: 5,
    numExtra: 0,
    confidenceThreshold: 0.3,
    maxDetections: 7,
    mode: 'stretch',
    image: [64, 64],
  },
  {
    name: 'detect, nothing passes',
    task: 'detect',
    inputSize: 64,
    numClasses: 3,
    numExtra: 0,
    confidenceThreshold: 1,
    maxDetections: 10,
    mode: 'stretch',
    image: [64, 64],
  },
  {
    name: 'detect, crop',
    task: 'detect',
    inputSize: 64,
    numClasses: 2,
    numExtra: 0,
    confidenceThreshold: 0.5,
    maxDetections: 84,
    mode: 'crop',
    cropFraction: 0.875,
    image: [80, 120],
  },
  {
    name: 'detect, tied logits go by cell',
    task: 'detect',
    inputSize: 64,
    numClasses: 2,
    numExtra: 0,
    confidenceThreshold: 0.5,
    maxDetections: 30,
    mode: 'stretch',
    image: [64, 64],
    quantize: 2,
  },
  // 8400 cells: the size the zoo runs at.
  {
    name: 'detect, 640 input',
    task: 'detect',
    inputSize: 640,
    numClasses: 80,
    numExtra: 0,
    confidenceThreshold: 0.9,
    maxDetections: 300,
    mode: 'letterbox',
    image: [1280, 720],
  },
  {
    name: 'pose, 17 landmarks',
    task: 'pose',
    inputSize: 64,
    numClasses: 1,
    numExtra: 17,
    confidenceThreshold: 0.5,
    maxDetections: 20,
    mode: 'letterbox',
    image: [90, 60],
  },
  {
    name: 'segment, 4 prototypes',
    task: 'segment',
    inputSize: 64,
    numClasses: 3,
    numExtra: 4,
    confidenceThreshold: 0.6,
    maxDetections: 12,
    mode: 'letterbox',
    image: [100, 70],
    maskSize: 8,
  },
  {
    name: 'segment, maxDetections cuts the list',
    task: 'segment',
    inputSize: 64,
    numClasses: 2,
    numExtra: 32,
    confidenceThreshold: 0.2,
    maxDetections: 3,
    mode: 'stretch',
    image: [64, 64],
    maskSize: 16,
  },
];

describe('decodeDetections', () => {
  it.each(CASES)('$name', async (c) => {
    const { task, inputSize: size, numClasses, numExtra, maskSize = 0 } = c;
    const channels =
      4 + numClasses + (task === 'pose' ? 3 * numExtra : task === 'segment' ? numExtra : 0);
    const levels = [8, 16, 32].map((stride, i) => {
      const g = size / stride;
      const data = randomF32(channels * g * g, i + 1).map((v) => 4 * v);
      return c.quantize ? data.map((v) => Math.round(v * c.quantize!) / c.quantize!) : data;
    });
    const proto = task === 'segment' ? randomF32(numExtra * (size / 4) ** 2, 9) : undefined;
    const want = decodeRef(c, levels, proto);

    const element =
      task === 'detect'
        ? objectDetection
        : task === 'pose'
          ? keypointDetection(numExtra)
          : instanceSegmentation(maskSize);
    const bytes = d.sizeOf(detectionList(element, c.maxDetections));
    const out = root.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const staging = root.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    const imageSize = root
      .createBuffer(d.arrayOf(d.u32, 2), storage(new Uint32Array(c.image)))
      .$usage('storage') as U32Buffer;
    const decode = createDecodeDetections(root, task, size);
    onTestFinished(() => {
      decode.dispose();
      out.destroy();
      staging.destroy();
    });

    const encoder = root.device.createCommandEncoder();
    decode.encode(
      encoder,
      [f32Buffer(levels[0]!), f32Buffer(levels[1]!), f32Buffer(levels[2]!)],
      proto && f32Buffer(proto),
      imageSize,
      out,
      { ...c, cropFraction: c.cropFraction ?? 1, maskSize },
    );
    encoder.copyBufferToBuffer(out, 0, staging, 0, bytes);
    root.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const words = staging.getMappedRange();
    const u32 = new Uint32Array(words);
    const f32 = new Float32Array(words);

    expect(u32[0]).toBe(want.length);
    const stride =
      task === 'detect' ? 6 : task === 'pose' ? 5 + 3 * numExtra : 6 + maskSize * maskSize;
    want.forEach((det, i) => {
      const base = 1 + i * stride;
      expectClose(f32.subarray(base, base + 5), [...det.box, det.confidence], {
        atol: 1e-3,
        rtol: 1e-4,
      });
      if (task !== 'pose') expect(u32[base + 5]).toBe(det.classId);
      if (task === 'pose')
        expectClose(f32.subarray(base + 5, base + stride), det.landmarks, {
          atol: 1e-3,
          rtol: 1e-4,
        });
      if (task === 'segment')
        expectClose(f32.subarray(base + 6, base + stride), det.mask, { atol: 1e-4 });
    });
    staging.unmap();
  });
});
