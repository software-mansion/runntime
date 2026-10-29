import { describe, expect, it } from 'vitest';
import { bufferSource, parseSafetensorsHeader } from '../../../src/core/index.ts';
import { loadDecoderEngine, type DecoderEngine } from '../../../src/zoo/decoder/engine.ts';
import { buffersSince, expectClose, gpu, liveBuffers } from '../../gpu.ts';
import {
  argmax,
  referenceLogits,
  tinyConfig,
  tinyDecoder,
  type TinyDecoder,
} from './tinyDecoder.ts';

const { root } = await gpu();
const device = root.device;

/** f32 kernels against an f64 reference, through 3 layers of norms. */
const TOL = { atol: 2e-3, rtol: 2e-3 };

const randomIds = (n: number, vocab: number, seed: number) => {
  let s = seed;
  return Array.from({ length: n }, () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s % vocab;
  });
};

async function load(model: TinyDecoder, opts: { maxContext?: number; maxChunk?: number } = {}) {
  const source = bufferSource(model.file);
  const { tensors } = await parseSafetensorsHeader(source);
  return loadDecoderEngine(device, model.cfg, { tensors, source }, opts);
}

/** Runs ids through the engine in maxChunk pieces; the last one's logits. */
async function prefill(engine: DecoderEngine, ids: readonly number[], from = 0) {
  let logits: Float32Array | undefined;
  for (let p = from; p < ids.length; p += engine.maxChunk) {
    const part = ids.slice(p, Math.min(ids.length, p + engine.maxChunk));
    if (p + part.length >= ids.length) logits = await engine.forward(part, p, true);
    else await engine.forward(part, p, false);
  }
  return logits!;
}

describe.each(['gemma3', 'qwen3'] as const)('decoder engine, %s', (arch) => {
  const model = tinyDecoder(tinyConfig(arch));
  const ids = randomIds(300, model.cfg.vocab, arch === 'gemma3' ? 7 : 11);
  const ref = referenceLogits(model, ids);

  it('matches the reference after a chunked prefill', async () => {
    // 40 tokens = chunks of 16, 16, 8: the GEMM route, then the GEMV one.
    const engine = await load(model, { maxContext: 64, maxChunk: 16 });
    try {
      expectClose(await prefill(engine, ids.slice(0, 40)), ref[39]!, TOL);
    } finally {
      engine.dispose();
    }
  });

  it('matches the reference token by token, past the sliding-window ring', async () => {
    // Ring of the sliding layers: window 6 + chunk 4 = 10 slots, wrapped
    // three times by 32 positions.
    const engine = await load(model, { maxContext: 64, maxChunk: 4 });
    try {
      expectClose(await prefill(engine, ids.slice(0, 3)), ref[2]!, TOL);
      for (let p = 3; p < 32; p++) {
        expectClose(await engine.forward([ids[p]!], p, true), ref[p]!, TOL);
      }
    } finally {
      engine.dispose();
    }
  });

  it('splits attention over the keys at long context', async () => {
    // 300 positions: 3 key blocks of 128 for the global layer, so a decode
    // step runs split-K with 3 partials and the combine pass.
    const engine = await load(model, { maxContext: 320, maxChunk: 32 });
    try {
      expectClose(await prefill(engine, ids.slice(0, 299)), ref[298]!, TOL);
      expectClose(await engine.forward([ids[299]!], 299, true), ref[299]!, TOL);
    } finally {
      engine.dispose();
    }
  });

  it('decodes a greedy burst like single steps with a CPU argmax', async () => {
    const engine = await load(model, { maxContext: 64, maxChunk: 8 });
    try {
      const prompt = ids.slice(0, 10);
      const first = argmax(await prefill(engine, prompt));
      const want: number[] = [];
      let tok = first;
      for (let p = prompt.length; p < prompt.length + 12; p++) {
        tok = argmax(await engine.forward([tok], p, true));
        want.push(tok);
      }
      expect(await engine.decodeGreedy(first, prompt.length, 12)).toEqual(want);
    } finally {
      engine.dispose();
    }
  });

  it('rejects out-of-range steps', async () => {
    const engine = await load(model, { maxContext: 16, maxChunk: 4 });
    try {
      await expect(engine.forward([], 0)).rejects.toThrow(/1\.\.4 tokens/);
      await expect(engine.forward([1, 2, 3, 4, 5], 0)).rejects.toThrow(/1\.\.4 tokens/);
      await expect(engine.forward([1], 16)).rejects.toThrow(/exceeds maxContext/);
      await expect(engine.decodeGreedy(1, 0, 0)).rejects.toThrow(/outside 1\.\.32/);
      await expect(engine.decodeGreedy(1, 10, 8)).rejects.toThrow(/exceeds maxContext/);
    } finally {
      engine.dispose();
    }
  });

  it('bounds rewinds by the sliding-window ring', async () => {
    const engine = await load(model, { maxContext: 64, maxChunk: 8 });
    try {
      // Ring of window 6 + chunk 8 slots keeps 8 positions beyond the window.
      expect(engine.maxRewind).toBe(arch === 'gemma3' ? 8 : Infinity);
    } finally {
      engine.dispose();
    }
  });

  it('frees every GPU buffer on dispose', async () => {
    const before = liveBuffers();
    const engine = await load(model, { maxContext: 64, maxChunk: 8 });
    await prefill(engine, ids.slice(0, 20));
    await engine.decodeGreedy(1, 20, 4);
    engine.dispose();
    expect(buffersSince(before)).toEqual([]);
  });
});

describe('decoder engine weights', () => {
  it('names the tensor a checkpoint misses or shapes wrong', async () => {
    const model = tinyDecoder(tinyConfig('qwen3'));
    const source = bufferSource(model.file);
    const { tensors } = await parseSafetensorsHeader(source);
    const before = liveBuffers();
    const missing = new Map(tensors);
    missing.delete('model.layers.1.mlp.up_proj.weight');
    await expect(
      loadDecoderEngine(device, model.cfg, { tensors: missing, source }),
    ).rejects.toThrow(/missing tensor 'model\.layers\.1\.mlp\.up_proj\.weight'/);
    const wrong = { ...model.cfg, ffn: 64 };
    await expect(loadDecoderEngine(device, wrong, { tensors, source })).rejects.toThrow(
      /'model\.layers\.0\.mlp\.gate_proj\.weight' is \[96,64\], expected \[64,64\]/,
    );
    await expect(loadDecoderEngine(device, wrong, { tensors, source })).rejects.toThrow(
      expect.objectContaining({ code: 'CHECKPOINT_MISMATCH' }),
    );
    expect(buffersSince(before)).toEqual([]);
  });
});
