/** The classifier's encode(): one forward recorded into a caller's encoder
 *  gives the same logits as run(). */
import { describe, expect, it } from 'vitest';
import { writeF32 } from '../../../src/core/gpu/buffers.ts';
import { memoryStateDict, type MemoryStateDict } from '../../../src/core/weights/safetensors.ts';
import { createClassifier } from '../../../src/zoo/mobilenetv4/classifier.ts';
import type { Mnv4Config } from '../../../src/zoo/mobilenetv4/config.ts';
import { MobileNetV4Model } from '../../../src/zoo/mobilenetv4/model.ts';
import { gpu, randomF32, requireFeature } from '../../gpu.ts';

/** MobileNetV4 cut to one conv and one UIR block on a 32×32 input. */
const TINY_MNV4: Mnv4Config = {
  stemOut: 8,
  stages: [
    [{ kind: 'cn', out: 16, kernelSize: 3, stride: 2 }],
    [{ kind: 'uir', out: 16, expand: 2, dwStartK: 3, dwMidK: 3, stride: 1 }],
  ],
  headHidden: 32,
  numClasses: 10,
  inputSize: 32,
};

describe('Classifier.encode', () => {
  it('reads the same logits as run()', async (ctx) => {
    const { root } = await gpu();
    requireFeature(ctx, root, 'shader-f16');
    const template = new MobileNetV4Model(TINY_MNV4, { root });
    const entries: MemoryStateDict = {};
    let seed = 1;
    for (const [name, p] of [...template.namedParameters(), ...template.namedBuffers()]) {
      entries[name] = {
        data: randomF32(p.shape.elems, seed++),
        shape: [...(p.shape.dims ?? [p.shape.elems])],
      };
    }
    template.dispose();
    // The checkpoint holds the classifier as [out, in]; the model keeps it [in, out].
    entries['classifier.weight'] = {
      data: randomF32(TINY_MNV4.numClasses * TINY_MNV4.headHidden, seed),
      shape: [TINY_MNV4.numClasses, TINY_MNV4.headHidden],
    };
    const classifier = await createClassifier(memoryStateDict(entries), { cfg: TINY_MNV4 });
    const size = TINY_MNV4.inputSize;
    const pixels = randomF32(3 * size * size, 7);
    const bytes = TINY_MNV4.numClasses * 4;
    const out = root.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    try {
      const want = await classifier.run(pixels);

      const encoder = root.device.createCommandEncoder();
      classifier.encode(
        encoder,
        (input) => writeF32(root, input, pixels),
        (logits) => encoder.copyBufferToBuffer(root.unwrap(logits), 0, out, 0, bytes),
      );
      root.device.queue.submit([encoder.finish()]);
      await out.mapAsync(GPUMapMode.READ);
      const got = new Float32Array(out.getMappedRange().slice(0));

      expect(want.some((v) => v !== 0 && Number.isFinite(v))).toBe(true);
      expect(got).toEqual(want);
    } finally {
      out.destroy();
      classifier.dispose();
    }
  });
});
