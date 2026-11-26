/** encodeEstimateDepth against estimateDepth on the same model input, and
 *  what a load and the first encode compile. Random weights, so the values
 *  mean nothing; the two paths must agree. */
import { describe, expect, it, onTestFinished, vi } from 'vitest';
import { ConvBlock } from '../../../../src/zoo/depthart/blocks.ts';
import { BatchNormC, RepDW } from '../../../../src/zoo/depthart/layers.ts';
import { DEPTHART_S_448, DepthartModel } from '../../../../src/zoo/depthart/model.ts';
import { createDepthEstimator } from '../../../../src/zoo/tasks/cv/depthEstimation.ts';
import { imageBuffer } from '../../../../src/zoo/tasks/cv/image.ts';
import { expectClose, freshDefaultRoot, gpu, requireFeature } from '../../../gpu.ts';
import { randomTensorsFile, type RandomTensor } from '../../randomWeights.ts';
import { randomRgba, rgbaTexture } from './images.ts';

const { root } = await gpu();
const SIZE = DEPTHART_S_448.inputSize;

/** Random weights in the checkpoint's layout: the model's own tensors, with
 *  the ones a load-time transform reads put back under the checkpoint's names. */
function randomCheckpoint() {
  const model = new DepthartModel(DEPTHART_S_448);
  const tensors = new Map<string, RandomTensor>();
  for (const [name, p] of [...model.namedParameters(), ...model.namedBuffers()]) {
    tensors.set(name, { name, shape: [...(p.shape.dims ?? [p.shape.elems])] });
  }
  const swap = (prefix: string, drop: string[], add: Record<string, number[]>) => {
    for (const name of drop) tensors.delete(`${prefix}.${name}`);
    for (const [name, shape] of Object.entries(add)) {
      const offset = name.endsWith('running_var') ? 1 : 0;
      tensors.set(`${prefix}.${name}`, { name: `${prefix}.${name}`, shape, offset });
    }
  };
  const batchNorm = (at: string, c: number) => ({
    [`${at}weight`]: [c],
    [`${at}bias`]: [c],
    [`${at}running_mean`]: [c],
    [`${at}running_var`]: [c],
  });
  for (const [prefix, m] of model.namedModules()) {
    if (m instanceof ConvBlock) {
      const c = m.channels;
      swap(prefix, ['mlp.fc2.c.bias'], { layer_scale: [c, 1, 1], ...batchNorm('mlp.fc2.bn.', c) });
    } else if (m instanceof BatchNormC) {
      swap(prefix, ['scale', 'shift'], batchNorm('', m.channels));
    } else if (m instanceof RepDW) {
      const c = m.channels;
      swap(prefix, ['merged.weight', 'merged.bias'], {
        'conv.c.weight': tensors.get(`${prefix}.merged.weight`)!.shape,
        ...batchNorm('conv.bn.', c),
        'conv1.weight': [c, 1, 1, 1],
        'conv1.bias': [c],
        ...batchNorm('bn.', c),
      });
    }
  }
  // The head ends in a ReLU: a positive bias keeps the map from being all zero.
  const last = 'depth_head.scratch.output_conv2.2.bias';
  tensors.set(last, { ...tensors.get(last)!, offset: 1 });
  model.dispose();
  // Scaled down so f16 activations stay finite through the random layers.
  return randomTensorsFile([...tensors.values()], 0.1);
}

const weights = randomCheckpoint();

async function estimator(warmUp = true) {
  const runner = await createDepthEstimator({ modelPath: weights }, { warmUp });
  onTestFinished(() => runner.dispose());
  return runner;
}

describe('encodeEstimateDepth', () => {
  it.for([
    { name: 'with a warm-up', warmUp: true },
    // The frame is recorded on the first call, so the load does no GPU work.
    { name: 'without a warm-up', warmUp: false },
  ])('the load compiles no pipeline on first use $name', async ({ warmUp }, ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    // A device of its own: on the shared one an earlier test compiled the model.
    const fresh = await freshDefaultRoot();
    const blocking = vi.spyOn(fresh.device, 'createComputePipeline');
    await estimator(warmUp);
    expect(blocking).not.toHaveBeenCalled();
  });

  it.for([
    { name: 'nothing after a load with warm-up', warmUp: true, compiles: false },
    { name: 'its own pipelines after a load without warm-up', warmUp: false, compiles: true },
  ])('compiles $name', async ({ warmUp, compiles }, ctx) => {
    requireFeature(ctx, root, 'shader-f16');
    const fresh = await freshDefaultRoot();
    const runner = await estimator(warmUp);
    const compute = vi.spyOn(fresh.device, 'createComputePipeline');
    const render = vi.spyOn(fresh.device, 'createRenderPipeline');
    const source = rgbaTexture(fresh, randomRgba(96, 48, 3), 96, 48);
    // The format the load compiles the draw pipeline for.
    const target = fresh.device.createTexture({
      size: [64, 64],
      format: navigator.gpu.getPreferredCanvasFormat(),
      usage: GPUTextureUsage.RENDER_ATTACHMENT,
    });
    onTestFinished(() => target.destroy());

    // minmax, so the range pipeline runs too.
    runner.encodeEstimateDepth({ source, target, normalize: 'minmax' });
    expect(compute.mock.calls.length > 0).toBe(compiles);
    expect(render.mock.calls.length > 0).toBe(compiles);
  });

  // #129: the recording inside encodeEstimateDepth starts a compile without
  // blocking while another load warms up, and the replay then uses that pipeline.
  it.fails(
    'encodes right after a load without warm-up that ran during another load',
    async (ctx) => {
      requireFeature(ctx, root, 'shader-f16');
      const fresh = await freshDefaultRoot();
      // The second load records its frame while the first compiles the same
      // pipelines without blocking.
      const warming = estimator(true);
      const runner = await estimator(false);
      // The input size and stretch, so the fit is the identity on both paths.
      const bytes = randomRgba(SIZE, SIZE, 3);
      const image = imageBuffer(bytes, SIZE, SIZE);
      const source = rgbaTexture(fresh, bytes, SIZE, SIZE);
      const target = fresh.createBuffer(runner.schemas.output).$usage('storage');
      onTestFinished(() => target.destroy());

      try {
        runner.encodeEstimateDepth({ source, target });
        const got = (await target.read()).flat();
        const want = await runner.estimateDepth(image);
        expectClose(got, want.data, { atol: 1e-3, rtol: 1e-3 });
      } finally {
        await warming;
      }
    },
  );
});
