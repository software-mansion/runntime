/** GPU decode of a YOLO26 one-to-one head: the cells scored at or above a
 *  threshold, best first, with boxes and landmarks in source-image pixels
 *  and a coverage mask over each box. The same results as decodeDetections,
 *  decodePoses and decodeSegmentations in zoo/yolo26/pipeline.ts, mapped
 *  back onto the image like the zoo runners do. */

import tgpu, { d, std } from 'typegpu';
import type { TgpuRoot } from 'typegpu';
import {
  flatIndex,
  GRID_Y_STRIDE,
  MAX_WORKGROUPS_PER_DIM,
  WORKGROUP_SIZE,
  type F32Buffer,
  type U32Buffer,
} from '../common.ts';
import { cachedBindGroup, cachedUniform } from '../../gpu/dispatchCache.ts';
import type { FitMode } from '../image/imageToChw.ts';

export type DecodeTask = 'detect' | 'pose' | 'segment';

/** Input pixels per cell of the three head levels. */
const STRIDES = [8, 16, 32] as const;
/** Input pixels per mask prototype pixel. */
const PROTO_STRIDE = 4;
/** Largest finite f32: the threshold that keeps every cell, or none. */
const F32_MAX = 3.4028234663852886e38;

const FIT_MODES: Record<FitMode, number> = { stretch: 0, letterbox: 1, crop: 2 };

const Box = d.struct({ xmin: d.f32, ymin: d.f32, xmax: d.f32, ymax: d.f32 });
const Landmark = d.struct({ x: d.f32, y: d.f32, confidence: d.f32 });

/** One object: its box in image pixels, its score 0 to 1 and its class. */
export const objectDetection = d.struct({ box: Box, confidence: d.f32, classId: d.u32 });

/** One person: box, score and every landmark in image pixels. */
export const keypointDetection = (keypoints: number) =>
  d.struct({ box: Box, confidence: d.f32, landmarks: d.arrayOf(Landmark, keypoints) });

/** One object with `mask`: `maskSize` × `maskSize` coverage values 0 to 1,
 *  row by row, stretched over `box`. */
export const instanceSegmentation = (maskSize: number) =>
  d.struct({
    box: Box,
    confidence: d.f32,
    classId: d.u32,
    mask: d.arrayOf(d.f32, maskSize * maskSize),
  });

/** The list a decode writes: `count` detections, best first, in room for
 *  `capacity`. Every field is 4 bytes, so nothing is padded and the kernels
 *  write it as words. */
export type DetectionList<T extends d.AnyWgslData> = d.WgslStruct<{
  count: d.U32;
  detections: d.WgslArray<T>;
}>;

export const detectionList = <T extends d.AnyWgslData>(
  detection: T,
  capacity: number,
): DetectionList<T> => d.struct({ count: d.u32, detections: d.arrayOf(detection, capacity) });

const Params = d.struct({
  inputSize: d.u32,
  numClasses: d.u32,
  /** Landmarks for pose, mask prototypes for segment. */
  numExtra: d.u32,
  logitThreshold: d.f32,
  maxDet: d.u32,
  mode: d.u32,
  cropFraction: d.f32,
  maskSize: d.u32,
});

const headLayout = tgpu.bindGroupLayout({
  p3: { storage: d.arrayOf(d.f32), access: 'readonly' },
  p4: { storage: d.arrayOf(d.f32), access: 'readonly' },
  p5: { storage: d.arrayOf(d.f32), access: 'readonly' },
  params: { uniform: Params },
});

/** `state` is [candidate count, image width, image height]. A candidate is
 *  three words: its cell, its best logit's bits and its class. */
const scoreLayout = tgpu.bindGroupLayout({
  state: { storage: d.arrayOf(d.atomic(d.u32)), access: 'mutable' },
  cand: { storage: d.arrayOf(d.u32), access: 'mutable' },
});

const rankLayout = tgpu.bindGroupLayout({
  // Read-only, so the candidate count is uniform and the tile loop can
  // hold barriers.
  state: { storage: d.arrayOf(d.u32), access: 'readonly' },
  cand: { storage: d.arrayOf(d.u32), access: 'readonly' },
  slotCell: { storage: d.arrayOf(d.u32), access: 'mutable' },
  out: { storage: d.arrayOf(d.u32), access: 'mutable' },
});

const maskLayout = tgpu.bindGroupLayout({
  proto: { storage: d.arrayOf(d.f32), access: 'readonly' },
  state: { storage: d.arrayOf(d.u32), access: 'readonly' },
  slotCell: { storage: d.arrayOf(d.u32), access: 'readonly' },
  out: { storage: d.arrayOf(d.u32), access: 'mutable' },
});

/** Where a cell of the flat list over all three levels sits: its level, its
 *  index in that level, the level's grid width and stride. */
const Cell = d.struct({ level: d.u32, local: d.u32, grid: d.u32, stride: d.f32 });

const locate = (cell: number): d.Infer<typeof Cell> => {
  'use gpu';
  const size = headLayout.$.params.inputSize;
  const g3 = std.intdiv(size, 8);
  const g4 = std.intdiv(size, 16);
  if (cell < g3 * g3) {
    return Cell({ level: 0, local: cell, grid: g3, stride: 8 });
  }
  if (cell < g3 * g3 + g4 * g4) {
    return Cell({ level: 1, local: cell - g3 * g3, grid: g4, stride: 16 });
  }
  return Cell({
    level: 2,
    local: cell - g3 * g3 - g4 * g4,
    grid: std.intdiv(size, 32),
    stride: 32,
  });
};

/** Channel `ch` of the cell at `c`. */
const head = (c: d.Infer<typeof Cell>, ch: number): number => {
  'use gpu';
  const i = d.u32(ch) * c.grid * c.grid + c.local;
  if (c.level === 0) {
    return headLayout.$.p3[i]!;
  }
  if (c.level === 1) {
    return headLayout.$.p4[i]!;
  }
  return headLayout.$.p5[i]!;
};

const sigmoid = (x: number): number => {
  'use gpu';
  return 1 / (1 + std.exp(-x));
};

/** Input pixel = image pixel * scale + offset, as imageToChw fits the
 *  image. `.xy` is the scale, `.zw` the offset. Keep it in step with
 *  resizeTransform in zoo/tasks/cv/ops/point.ts. */
const fitOf = (img: d.v2f): d.v4f => {
  'use gpu';
  const P = headLayout.$.params;
  const out = d.vec2f(d.f32(P.inputSize));
  let scale = out / img;
  let offset = d.vec2f();
  if (P.mode === 1) {
    scale = d.vec2f(std.min(scale.x, scale.y));
    offset = (out - img * scale) / 2;
  }
  if (P.mode === 2) {
    const side = std.min(img.x, img.y) * P.cropFraction;
    scale = out / side;
    offset = ((side - img) / 2) * scale;
  }
  return d.vec4f(scale, offset);
};

/** One thread per cell: the best class, and the cell into the candidate
 *  list when it clears the threshold. Ties go to the lower class, as on the
 *  CPU. */
const scoreKernel = tgpu.computeFn({
  in: { gid: d.builtin.globalInvocationId },
  workgroupSize: [WORKGROUP_SIZE],
})(({ gid }) => {
  'use gpu';
  const P = headLayout.$.params;
  const cell = flatIndex(gid);
  const g3 = std.intdiv(P.inputSize, 8);
  const g4 = std.intdiv(P.inputSize, 16);
  const g5 = std.intdiv(P.inputSize, 32);
  if (cell >= g3 * g3 + g4 * g4 + g5 * g5) {
    return;
  }
  const c = locate(cell);
  let best = d.f32(-F32_MAX);
  let cls = d.u32(0);
  for (let k = d.u32(0); k < P.numClasses; k++) {
    const v = head(c, 4 + k);
    if (v > best) {
      best = v;
      cls = k;
    }
  }
  if (best < P.logitThreshold) {
    return;
  }
  const slot = std.atomicAdd(scoreLayout.$.state[0]!, 1);
  scoreLayout.$.cand[3 * slot] = cell;
  scoreLayout.$.cand[3 * slot + 1] = std.bitcastF32toU32(best);
  scoreLayout.$.cand[3 * slot + 2] = cls;
});

const tileLogit = tgpu.workgroupVar(d.arrayOf(d.f32, WORKGROUP_SIZE));
const tileCell = tgpu.workgroupVar(d.arrayOf(d.u32, WORKGROUP_SIZE));

/** Writes candidate `cell` at list position `slot`, from word `base` on. */
type Writer = (slot: number, base: number, cell: number, logit: number, cls: number) => void;

/** The box of `cell` in image pixels, cut at the image edge, written at
 *  `base`, then the score. */
const writeBox = (base: number, c: d.Infer<typeof Cell>, logit: number) => {
  'use gpu';
  const ax = d.f32(c.local % c.grid) + 0.5;
  const ay = d.f32(std.intdiv(c.local, c.grid)) + 0.5;
  const lo = d.vec2f(ax - head(c, 0), ay - head(c, 1)) * c.stride;
  const hi = d.vec2f(ax + head(c, 2), ay + head(c, 3)) * c.stride;
  const img = d.vec2f(d.f32(rankLayout.$.state[1]!), d.f32(rankLayout.$.state[2]!));
  const fit = fitOf(img);
  const min = std.clamp((lo - fit.zw) / fit.xy, d.vec2f(), img);
  const max = std.clamp((hi - fit.zw) / fit.xy, d.vec2f(), img);
  const out = rankLayout.$.out;
  out[base] = std.bitcastF32toU32(min.x);
  out[base + 1] = std.bitcastF32toU32(min.y);
  out[base + 2] = std.bitcastF32toU32(max.x);
  out[base + 3] = std.bitcastF32toU32(max.y);
  out[base + 4] = std.bitcastF32toU32(sigmoid(logit));
};

const writers: Record<DecodeTask, Writer> = {
  detect: (_slot, base, cell, logit, cls) => {
    'use gpu';
    writeBox(base, locate(cell), logit);
    rankLayout.$.out[base + 5] = cls;
  },
  pose: (_slot, base, cell, logit) => {
    'use gpu';
    const c = locate(cell);
    writeBox(base, c, logit);
    const img = d.vec2f(d.f32(rankLayout.$.state[1]!), d.f32(rankLayout.$.state[2]!));
    const fit = fitOf(img);
    const P = headLayout.$.params;
    const anchor = d.vec2f(d.f32(c.local % c.grid), d.f32(std.intdiv(c.local, c.grid))) + 0.5;
    for (let k = d.u32(0); k < P.numExtra; k++) {
      const ch = 4 + P.numClasses + 3 * k;
      const p = (d.vec2f(head(c, ch), head(c, ch + 1)) + anchor) * c.stride;
      const q = (p - fit.zw) / fit.xy;
      const o = base + 5 + 3 * k;
      rankLayout.$.out[o] = std.bitcastF32toU32(q.x);
      rankLayout.$.out[o + 1] = std.bitcastF32toU32(q.y);
      rankLayout.$.out[o + 2] = std.bitcastF32toU32(sigmoid(head(c, ch + 2)));
    }
  },
  segment: (slot, base, cell, logit, cls) => {
    'use gpu';
    writeBox(base, locate(cell), logit);
    rankLayout.$.out[base + 5] = cls;
    rankLayout.$.slotCell[slot] = cell;
  },
};

/** One thread per candidate: its rank is how many candidates beat it, by
 *  logit and then by cell, so the order is the CPU's. The candidates stream
 *  through workgroup memory a tile at a time. */
const makeRankKernel = (write: Writer, words: () => number) =>
  tgpu.computeFn({
    in: { gid: d.builtin.globalInvocationId, lid: d.builtin.localInvocationId },
    workgroupSize: [WORKGROUP_SIZE],
  })(({ gid, lid }) => {
    'use gpu';
    const P = headLayout.$.params;
    const count = rankLayout.$.state[0]!;
    const i = gid.x;
    let mine = d.f32(-F32_MAX);
    let myCell = d.u32(0xffffffff);
    let myClass = d.u32(0);
    if (i < count) {
      myCell = rankLayout.$.cand[3 * i]!;
      mine = std.bitcastU32toF32(rankLayout.$.cand[3 * i + 1]!);
      myClass = rankLayout.$.cand[3 * i + 2]!;
    }
    let rank = d.u32(0);
    for (let tile = d.u32(0); tile < count; tile += WORKGROUP_SIZE) {
      const j = tile + lid.x;
      if (j < count) {
        tileCell.$[lid.x] = rankLayout.$.cand[3 * j]!;
        tileLogit.$[lid.x] = std.bitcastU32toF32(rankLayout.$.cand[3 * j + 1]!);
      }
      std.workgroupBarrier();
      const n = std.min(d.u32(WORKGROUP_SIZE), count - tile);
      for (let t = d.u32(0); t < n; t++) {
        const v = tileLogit.$[t]!;
        if (v > mine || (v === mine && tileCell.$[t]! < myCell)) {
          rank += d.u32(1);
        }
      }
      std.workgroupBarrier();
    }
    if (i === 0) {
      rankLayout.$.out[0] = std.min(count, P.maxDet);
    }
    if (i >= count || rank >= P.maxDet) {
      return;
    }
    write(rank, 1 + rank * words(), myCell, mine, myClass);
  });

/** Words per detection, as the list schemas lay them out. */
const wordsOf: Record<DecodeTask, () => number> = {
  detect: () => {
    'use gpu';
    return d.u32(6);
  },
  pose: () => {
    'use gpu';
    return 5 + 3 * headLayout.$.params.numExtra;
  },
  segment: () => {
    'use gpu';
    const m = headLayout.$.params.maskSize;
    return 6 + m * m;
  },
};

/** One thread per mask value of every listed detection: the box point it
 *  stands for, mapped into the prototypes, bilinear over the four nearest
 *  prototype pixels, each the sigmoid of its coefficient sum. */
const maskKernel = tgpu.computeFn({
  in: { gid: d.builtin.globalInvocationId },
  workgroupSize: [WORKGROUP_SIZE],
})(({ gid }) => {
  'use gpu';
  const P = headLayout.$.params;
  const idx = flatIndex(gid);
  const m2 = P.maskSize * P.maskSize;
  const slot = std.intdiv(idx, m2);
  const out = maskLayout.$.out;
  if (slot >= std.min(maskLayout.$.state[0]!, P.maxDet)) {
    return;
  }
  const t = idx % m2;
  const base = 1 + slot * (6 + m2);
  const c = locate(maskLayout.$.slotCell[slot]!);
  const lo = d.vec2f(std.bitcastU32toF32(out[base]!), std.bitcastU32toF32(out[base + 1]!));
  const hi = d.vec2f(std.bitcastU32toF32(out[base + 2]!), std.bitcastU32toF32(out[base + 3]!));
  const img = d.vec2f(d.f32(maskLayout.$.state[1]!), d.f32(maskLayout.$.state[2]!));
  const fit = fitOf(img);
  const uv =
    (d.vec2f(d.f32(t % P.maskSize), d.f32(std.intdiv(t, P.maskSize))) + 0.5) / d.f32(P.maskSize);
  const q = (lo + uv * (hi - lo)) * fit.xy + fit.zw;
  const pw = std.intdiv(P.inputSize, PROTO_STRIDE);
  const f = q / PROTO_STRIDE - 0.5;
  const a = std.floor(f);
  const w = f - a;
  const last = d.vec2i(d.i32(pw) - 1);
  const i0 = d.vec2u(std.clamp(d.vec2i(a), d.vec2i(), last));
  const i1 = d.vec2u(std.clamp(d.vec2i(a) + 1, d.vec2i(), last));
  const phw = pw * pw;
  let s00 = d.f32(0);
  let s10 = d.f32(0);
  let s01 = d.f32(0);
  let s11 = d.f32(0);
  for (let k = d.u32(0); k < P.numExtra; k++) {
    const coef = head(c, 4 + P.numClasses + k);
    const plane = k * phw;
    s00 += coef * maskLayout.$.proto[plane + i0.y * pw + i0.x]!;
    s10 += coef * maskLayout.$.proto[plane + i0.y * pw + i1.x]!;
    s01 += coef * maskLayout.$.proto[plane + i1.y * pw + i0.x]!;
    s11 += coef * maskLayout.$.proto[plane + i1.y * pw + i1.x]!;
  }
  const top = std.mix(sigmoid(s00), sigmoid(s10), w.x);
  const bot = std.mix(sigmoid(s01), sigmoid(s11), w.x);
  out[base + 6 + t] = std.bitcastF32toU32(std.mix(top, bot, w.y));
});

export interface DecodeDetectionsArgs {
  readonly numClasses: number;
  /** Landmarks for pose, mask prototypes for segment, 0 for detect. */
  readonly numExtra: number;
  /** Keeps cells whose best class scores at least this, 0 to 1. */
  readonly confidenceThreshold: number;
  readonly maxDetections: number;
  /** How the image was fitted into the input, `stretch` for an input that
   *  is the image. */
  readonly mode: FitMode;
  readonly cropFraction: number;
  /** Segment only: mask values per box side. */
  readonly maskSize: number;
}

/** Records head-to-list passes. Create once per model and input size: it
 *  holds the pipelines and the candidate scratch. */
export function createDecodeDetections(root: TgpuRoot, task: DecodeTask, inputSize: number) {
  const cells = STRIDES.reduce((sum, s) => sum + (inputSize / s) ** 2, 0);
  const scorePipeline = root.createComputePipeline({ compute: scoreKernel });
  const rankPipeline = root.createComputePipeline({
    compute: makeRankKernel(writers[task], wordsOf[task]),
  });
  const maskPipeline = root.createComputePipeline({ compute: maskKernel });
  const state = root.createBuffer(d.arrayOf(d.u32, 3)).$usage('storage');
  const cand = root.createBuffer(d.arrayOf(d.u32, 3 * cells)).$usage('storage');
  const slotCell = root.createBuffer(d.arrayOf(d.u32, cells)).$usage('storage');
  const workgroups = Math.ceil(cells / WORKGROUP_SIZE);
  return {
    /** Decodes `levels`, the three `[channels, h, w]` f32 head levels, and
     *  `proto` for segment, into `out` as a detectionList. `imageSize` holds
     *  the image's `[width, height]`. */
    encode(
      encoder: GPUCommandEncoder,
      levels: readonly [F32Buffer, F32Buffer, F32Buffer],
      proto: F32Buffer | undefined,
      imageSize: U32Buffer,
      out: GPUBuffer,
      args: DecodeDetectionsArgs,
    ): void {
      const { confidenceThreshold: conf } = args;
      // log(c / (1 - c)), the logit the CPU compares against; ±F32_MAX for 0
      // and 1, as Infinity does not survive the uniform cache's JSON key.
      const logit = Math.log(conf / (1 - conf));
      const params = cachedUniform(root, Params, {
        inputSize,
        numClasses: args.numClasses,
        numExtra: args.numExtra,
        logitThreshold: Math.min(F32_MAX, Math.max(-F32_MAX, logit)),
        maxDet: args.maxDetections,
        mode: FIT_MODES[args.mode],
        cropFraction: args.cropFraction,
        maskSize: args.maskSize,
      });
      const [p3, p4, p5] = levels;
      const headGroup = cachedBindGroup(root, headLayout, { p3, p4, p5, params });
      encoder.clearBuffer(root.unwrap(state), 0, 4);
      encoder.copyBufferToBuffer(root.unwrap(imageSize), 0, root.unwrap(state), 4, 8);
      const pass = encoder.beginComputePass();
      scorePipeline
        .with(pass)
        .with(headGroup)
        .with(cachedBindGroup(root, scoreLayout, { state, cand }))
        .dispatchWorkgroups(workgroups);
      rankPipeline
        .with(pass)
        .with(headGroup)
        .with(cachedBindGroup(root, rankLayout, { state, cand, slotCell, out }))
        .dispatchWorkgroups(workgroups);
      if (task === 'segment' && proto !== undefined) {
        const threads = args.maxDetections * args.maskSize * args.maskSize;
        maskPipeline
          .with(pass)
          .with(headGroup)
          .with(cachedBindGroup(root, maskLayout, { proto, state, slotCell, out }))
          .dispatchWorkgroups(
            Math.min(Math.ceil(threads / WORKGROUP_SIZE), MAX_WORKGROUPS_PER_DIM),
            Math.ceil(threads / GRID_Y_STRIDE),
          );
      }
      pass.end();
    },
    dispose(): void {
      state.destroy();
      cand.destroy();
      slotCell.destroy();
    },
  };
}
