import { d } from 'typegpu';
import { describe, expect, it, onTestFinished } from 'vitest';
import type { F32Buffer } from '../../../../src/core/kernels/common.ts';
import {
  classScoreList,
  createSoftmaxTopK,
} from '../../../../src/core/kernels/reduce/softmaxTopK.ts';
import { expectClose, gpu, randomF32 } from '../../../gpu.ts';

const { root } = await gpu();
const softmaxTopK = createSoftmaxTopK(root);

/** Softmax from the definition, in f64, and the k largest probabilities;
 *  equal logits keep index order. */
function topKRef(logits: Float32Array, k: number): { classId: number; confidence: number }[] {
  const max = Math.max(...logits);
  const exps = Array.from(logits, (v) => Math.exp(v - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps
    .map((e, classId) => ({ classId, confidence: e / sum }))
    .sort((a, b) => b.confidence - a.confidence || a.classId - b.classId)
    .slice(0, k);
}

const CASES = [
  { name: 'the 5 best of 1000', n: 1000, k: 5 },
  { name: 'every entry of 1000', n: 1000, k: 1000 },
  { name: 'one entry', n: 1, k: 1 },
  { name: 'fewer entries than a workgroup', n: 10, k: 3 },
  // Not a multiple of the 64-wide workgroup.
  { name: 'an odd count', n: 131, k: 131 },
  { name: 'tied logits keep index order', n: 200, k: 50, quantize: 1 },
  { name: 'k past n stops at n', n: 20, k: 30 },
];

describe('softmaxTopK', () => {
  it.each(CASES)('$name', async ({ n, k, quantize }) => {
    let logits = randomF32(n, n).map((v) => 8 * v);
    if (quantize) logits = logits.map((v) => Math.round(v * quantize) / quantize);
    const want = topKRef(logits, k);

    const input = root.createBuffer(d.arrayOf(d.f32, n), Array.from(logits)).$usage('storage');
    const bytes = d.sizeOf(classScoreList(k));
    const out = root.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
    });
    const staging = root.device.createBuffer({
      size: bytes,
      usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
    });
    onTestFinished(() => {
      input.destroy();
      out.destroy();
      staging.destroy();
    });

    const encoder = root.device.createCommandEncoder();
    softmaxTopK.encode(encoder, input as F32Buffer, out, { n, k });
    encoder.copyBufferToBuffer(out, 0, staging, 0, bytes);
    root.device.queue.submit([encoder.finish()]);
    await staging.mapAsync(GPUMapMode.READ);
    const words = staging.getMappedRange();
    const ids = new Uint32Array(words);
    const conf = new Float32Array(words);

    expect(Array.from(want, (_, i) => ids[2 * i])).toEqual(want.map((w) => w.classId));
    expectClose(
      Array.from(want, (_, i) => conf[2 * i + 1]!),
      want.map((w) => w.confidence),
      { atol: 1e-7, rtol: 1e-5 },
    );
    staging.unmap();
  });
});
