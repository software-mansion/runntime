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
import { buffersSince, gpu, liveBuffers, randomF32, requireFeature } from '../gpu.ts';

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
  /** A known leak, not fixed yet, that fails the dispose check. */
  disposeLeak?: string;
  /** A known leak, not fixed yet, that fails the warm check. */
  callLeak?: string;
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

const CASES: LeakCase[] = [
  {
    name: 'MiniLM embed core',
    build(root) {
      const model = new MinilmModel(TINY_MINILM);
      loadRandomWeights(model, root);
      const core = createMinilmEmbedCore(model);
      return {
        call: async () => {
          await core.embedPooled([1, 2, 3, 4, 5]);
          await core.embedPooledBatch([
            [1, 2, 3],
            [4, 5],
          ]);
        },
        dispose: () => {
          core.dispose();
          model.dispose();
        },
      };
    },
    disposeLeak: 'replayCache.dispose() keeps each bucket executor and its recorded buffers',
    callLeak: 'embedPooledBatch never destroys its uploads, and toArray keeps its result',
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
    disposeLeak: 'executor.dispose() keeps its dummy buffers, and toArray keeps its result',
    callLeak: 'toArray keeps its result buffer out of the pool',
  },
];

describe.for(CASES)('$name', ({ build, needs, disposeLeak, callLeak }) => {
  (disposeLeak ? it.fails : it)('frees every buffer on dispose', async (ctx) => {
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
    expect(buffersSince(before)).toEqual([]);
  });

  (callLeak ? it.fails : it)('stops allocating once warm', async (ctx) => {
    const { root } = await gpu();
    if (needs) requireFeature(ctx, root, needs);
    const runner = build(root);
    try {
      for (let i = 0; i < WARM_CALLS; i++) await runner.call();
      const before = liveBuffers();
      for (let i = 0; i < STEADY_CALLS; i++) await runner.call();
      expect(buffersSince(before)).toEqual([]);
    } finally {
      runner.dispose();
    }
  });
});
