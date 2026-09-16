import { inGpuErrorScopes, toArray } from '../../core/index.ts';
import type { LabelInfo } from './config.ts';
import type { EagerModel } from './model.ts';
import { detectedSpansFromLabels, type DetectedSpan } from './spans.ts';
import type { Tokenizer } from './tokenizer.ts';
import { logSoftmaxRows, ViterbiDecoder } from './viterbi.ts';

export interface DetectResult {
  spans: DetectedSpan[];
  text: string;
  timings: { tokenizeMs: number; gpuMs: number; decodeMs: number };
}

export interface Detector {
  detect(text: string): Promise<DetectResult>;
  readonly tokenizer: Tokenizer;
  readonly weightFormat: string;
  dispose(): void;
}

export function assembleDetect(deps: {
  device: GPUDevice;
  tokenizer: Tokenizer;
  info: LabelInfo;
  decoder: ViterbiDecoder;
  model: EagerModel;
}): (text: string) => Promise<DetectResult> {
  const { device, tokenizer, info, decoder, model } = deps;
  return async function detect(text: string): Promise<DetectResult> {
    const t0 = performance.now();
    const tokenIds = tokenizer.encode(text);
    const t1 = performance.now();
    if (tokenIds.length === 0) {
      return { spans: [], text, timings: { tokenizeMs: t1 - t0, gpuMs: 0, decodeMs: 0 } };
    }
    const logits = await inGpuErrorScopes(device, 'inference', () =>
      toArray(model.forward(tokenIds)),
    );
    // Last-resort sanity check: error scopes can't see zeros produced without a
    // reported error (e.g. robust-access clamping). Real logits are never
    // all-zero (33 biased classes) and never non-finite.
    if (!logits.every(Number.isFinite)) {
      throw new Error('GPU returned non-finite logits — inference failed on this device');
    }
    if (logits.every((v) => v === 0)) {
      throw new Error('GPU returned all-zero logits — inference failed on this device');
    }
    const t2 = performance.now();
    const logProbs = logSoftmaxRows(logits, 33);
    const labels = decoder.decode(logProbs, 33);
    const spans = detectedSpansFromLabels(labels, tokenIds, tokenizer, info);
    const t3 = performance.now();
    const { text: decoded } = tokenizer.tokenCharOffsets(tokenIds);
    return {
      spans,
      text: decoded,
      timings: { tokenizeMs: t1 - t0, gpuMs: t2 - t1, decodeMs: t3 - t2 },
    };
  };
}
