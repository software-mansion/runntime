/** GPU memory leak checks: each zoo model, built small with random weights,
 *  must free every GPU buffer it made on dispose, and must stop allocating
 *  once warm. */
import type { TgpuRoot } from 'typegpu';
import { describe, expect, it } from 'vitest';
import { tensor3d, toArray } from '../../src/core/index.ts';
import { tensor } from '../../src/core/gpu/buffers.ts';
import { gpuExecutor } from '../../src/core/gpu/eval.ts';
import { CpuParameter, type AnyModule } from '../../src/core/nn/module.ts';
import { gpuUpload } from '../../src/core/nn/loadStateDict.ts';
import { f32ArrayToF16Bits } from '../../src/core/weights/convert.ts';
import { MINILM_L6 } from '../../src/zoo/minilm/config.ts';
import { createMinilmEmbedCore } from '../../src/zoo/minilm/embedCore.ts';
import { MinilmModel } from '../../src/zoo/minilm/model.ts';
import { MobileNetV4Model } from '../../src/zoo/mobilenetv4/model.ts';
import { buffersSince, gpu, liveBuffers, randomF32, repeated, requireFeature } from '../gpu.ts';

/** Calls a model makes before its buffer count must hold still: the first
 *  call per shape records replays and fills the pool. */
const WARM_CALLS = 3;
/** Calls after warm-up that must not allocate a single buffer. */
const STEADY_CALLS = 20;

/** Binds every parameter and buffer the way loadStateDict does, with random
 *  values. Skips the checkpoint transforms: the names and layouts they adapt
 *  are the checkpoint's, and the model's own shapes are what runs. */
function loadRandomWeights(model: AnyModule, root: TgpuRoot): void {
  const upload = gpuUpload(root);
  let seed = 1;
  for (const [, p] of model.namedParameters()) {
    const f32 = randomF32(p.shape.elems, seed++);
    if (p instanceof CpuParameter) {
      p.bindData(f32);
      continue;
    }
    // f16 slots get their bits before preprocess, as from a checkpoint.
    const data = p.shape.dtype === 'f16' ? f32ArrayToF16Bits(f32).data : f32;
    p.bind(upload(p.preprocess ? p.preprocess(data) : data, p.shape));
  }
  for (const [, b] of model.namedBuffers()) b.bind(upload(b.data, b.shape));
}

/** A model as its runner holds it: one call, and the dispose that frees it. */
interface Runner {
  call(): Promise<unknown>;
  dispose(): void;
}

interface LeakCase {
  name: string;
  needs?: GPUFeatureName;
  build(root: TgpuRoot): Runner;
  /** Buffers a known leak, not fixed yet, leaves behind after dispose. */
  leftOnDispose?: string[];
  /** Buffers a known leak, not fixed yet, leaves behind on every warm call. */
  leftPerCall?: string[];
}

/** all-MiniLM-L6-v2 at two layers and a small vocab: the replay path for one
 *  text and the packed eager path for a batch. */
const TINY_MINILM = { ...MINILM_L6, layers: 2, vocab: 64 };

/** MobileNetV4 cut to one conv and one UIR block on a 32×32 input. */
const TINY_MNV4 = {
  stemOut: 8,
  stages: [
    [{ kind: 'cn', out: 16, kernelSize: 3, stride: 2 }],
    [{ kind: 'uir', out: 16, expand: 2, dwStartK: 3, dwMidK: 3, stride: 1 }],
  ],
  headHidden: 32,
  numClasses: 10,
  inputSize: 32,
} as const;

/** The uploads one MiniLM batch call of 5 tokens never frees, in the order
 *  they are made: the [5, 2] segments from MinilmModel.forward, the [2, 5]
 *  selector from embedPooledBatch, and the id and position lists gatherRows
 *  uploads for Embedding.forward. */
const MINILM_BATCH_UPLOADS = [
  ...['40 B, usage 140', '40 B, usage 140'],
  ...['20 B, usage 140', '20 B, usage 140'],
];
/** The [2, 384] embeddings toArray keeps out of the pool. */
const MINILM_BATCH_RESULT = '3072 B, usage 140';

const CASES: LeakCase[] = [
  {
    name: 'MiniLM, one text',
    // Without subgroups attention takes the rows route, which leaves other
    // buffers.
    needs: 'subgroups',
    build(root) {
      const model = new MinilmModel(TINY_MINILM);
      loadRandomWeights(model, root);
      const core = createMinilmEmbedCore(model);
      return {
        call: () => core.embedPooled([1, 2, 3, 4, 5]),
        dispose: () => {
          core.dispose();
          model.dispose();
        },
      };
    },
    // The bucket executor's dispose() keeps its dummy and the kernels'
    // uniforms (usage 76).
    leftOnDispose: [
      ...['8 B, usage 76', '8 B, usage 76', '16 B, usage 76', '16 B, usage 140'],
      ...Array<string>(4).fill('16 B, usage 76'),
    ],
  },
  {
    name: 'MiniLM, a batch',
    // Without subgroups attention takes the rows route, which leaves other
    // buffers.
    needs: 'subgroups',
    build(root) {
      const model = new MinilmModel(TINY_MINILM);
      loadRandomWeights(model, root);
      const core = createMinilmEmbedCore(model);
      return {
        call: () =>
          core.embedPooledBatch([
            [1, 2, 3],
            [4, 5],
          ]),
        dispose: () => {
          core.dispose();
          model.dispose();
        },
      };
    },
    // dispose() also keeps the kernels' uniforms.
    leftOnDispose: [
      ...MINILM_BATCH_UPLOADS,
      ...Array<string>(5).fill('16 B, usage 76'),
      MINILM_BATCH_RESULT,
    ],
    leftPerCall: [...MINILM_BATCH_UPLOADS, MINILM_BATCH_RESULT],
  },
  {
    name: 'MobileNetV4',
    needs: 'shader-f16',
    build(root) {
      const model = new MobileNetV4Model(TINY_MNV4, { root });
      loadRandomWeights(model, root);
      const ex = gpuExecutor(root);
      const size = TINY_MNV4.inputSize;
      const input = tensor(root, randomF32(3 * size * size), tensor3d(3, size, size));
      return {
        call: () => toArray(model.forward(input), ex),
        dispose: () => {
          model.dispose();
          ex.dispose();
          input.buffer.destroy();
        },
      };
    },
    // executor.dispose() keeps its 8 B f16 dummy, and toArray keeps its
    // 10-class result out of the pool.
    leftOnDispose: ['8 B, usage 140', '40 B, usage 140'],
    leftPerCall: ['40 B, usage 140'],
  },
];

describe.for(CASES)('$name', ({ build, needs, leftOnDispose = [], leftPerCall = [] }) => {
  it('frees every buffer on dispose', async (ctx) => {
    const { root } = await gpu();
    if (needs) requireFeature(ctx, root, needs);
    // A throwaway run first: pipelines and the readback staging buffer belong
    // to the root and outlive every model.
    const first = build(root);
    await first.call();
    first.dispose();

    const before = liveBuffers();
    const runner = build(root);
    await runner.call();
    runner.dispose();
    expect(buffersSince(before)).toEqual(leftOnDispose);
  });

  it('stops allocating once warm', async (ctx) => {
    const { root } = await gpu();
    if (needs) requireFeature(ctx, root, needs);
    const runner = build(root);
    try {
      for (let i = 0; i < WARM_CALLS; i++) await runner.call();
      const before = liveBuffers();
      for (let i = 0; i < STEADY_CALLS; i++) await runner.call();
      expect(buffersSince(before)).toEqual(repeated(leftPerCall, STEADY_CALLS));
    } finally {
      runner.dispose();
    }
  });
});
