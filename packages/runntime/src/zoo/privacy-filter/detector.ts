/** Privacy filter detector. Loads the weights once into one EagerModel,
 *  then each detect is one forward pass. Any text length runs on the same
 *  model, rope tables are computed per call. Runs on the initRunntime() default
 *  device. Dispose frees the weights. */

import { defaultRoot, supportsF16, toArray, warmUp, type LazyStateDict } from '../../core/index.ts';
import { buildLabelInfo, V2_NER_CLASS_NAMES, ZERO_BIASES, type ViterbiBiases } from './config.ts';
import { createTokenizer } from './tokenizer.ts';
import { ViterbiDecoder } from './viterbi.ts';
import { assembleDetect, type Detector } from './pipeline.ts';
import { EagerModel } from './model.ts';
import { transformHfPrivacyFilterStateDict } from './stateDictHooks.ts';

const BLOCKS = 8;
const EXPERTS = 128;
const LOGITS = 33;

export interface LoadedPrivacyFilterModel {
  model: EagerModel;
  weightFormat: string;
}

const mismatch = (message: string) => new Error(`privacy-filter weights: ${message}`);

export async function loadPrivacyFilterModel(
  sd: LazyStateDict,
  opts: {
    onProgress?: (name: string, doneBytes: number, totalBytes: number) => void;
  } = {},
): Promise<LoadedPrivacyFilterModel> {
  const device = defaultRoot().device;
  // The Hub file has no model metadata and transformers tensor names, so
  // map them. Exported files carry the metadata and our names already.
  if (sd.metadata['model'] === undefined) {
    try {
      transformHfPrivacyFilterStateDict(sd);
    } catch (err) {
      throw mismatch((err as Error).message);
    }
  }
  if (sd.metadata['model'] !== 'privacy-filter') {
    throw mismatch(`file is for model '${sd.metadata['model']}', expected 'privacy-filter'`);
  }
  let quantization: { bits: 8 | 4; groupSize: number } | undefined;
  if (sd.metadata['quantization']) {
    try {
      quantization = JSON.parse(sd.metadata['quantization']) as { bits: 8 | 4; groupSize: number };
    } catch {
      throw mismatch(`malformed quantization metadata '${sd.metadata['quantization']}'`);
    }
    if ((quantization.bits !== 8 && quantization.bits !== 4) || !(quantization.groupSize > 0)) {
      throw mismatch(
        `unsupported quantization ${JSON.stringify(quantization)}, expected bits 8|4 and a positive groupSize`,
      );
    }
  }
  const weightFormat: 'f32' | 'quantInt8' | 'quantInt4' = quantization
    ? quantization.bits === 4
      ? 'quantInt4'
      : 'quantInt8'
    : 'f32';
  // Vocab comes from the checkpoint: embedding.weight is [vocab, hidden].
  const embedding = sd.tensors.get('embedding.weight');
  if (!embedding) throw mismatch("missing 'embedding.weight' tensor");
  const vocab = embedding.shape[0];
  if (embedding.shape.length !== 2 || vocab === undefined || vocab <= 0) {
    throw mismatch(
      `embedding.weight has shape ${JSON.stringify(embedding.shape)}, expected [vocab, hidden]`,
    );
  }
  const model = new EagerModel({
    blocks: BLOCKS,
    experts: EXPERTS,
    vocab,
    logits: LOGITS,
    weightFormat,
    groupSize: quantization?.groupSize,
  });
  // Unquantized weights run in half precision: the checkpoint is F16 on disk,
  // so this is the zero-copy load path and half the GPU memory. Quantized
  // formats carry their own narrowing and stay as they are.
  const halved = weightFormat === 'f32' && supportsF16();
  if (halved) model.half();
  const formatLabel = quantization
    ? `int${quantization.bits}/g${quantization.groupSize}`
    : halved
      ? 'f16'
      : 'f32';
  try {
    await model.loadStateDict(sd, { onProgress: opts.onProgress });
    await warmUp(device, () => toArray(model.forward([1000])));
  } catch (err) {
    model.dispose();
    throw err;
  }
  return { model, weightFormat: formatLabel };
}

export async function createDetector(
  sd: LazyStateDict,
  opts: {
    biases?: ViterbiBiases;
    onProgress?: (name: string, doneBytes: number, totalBytes: number) => void;
  } = {},
): Promise<Detector> {
  const device = defaultRoot().device;
  const { model, weightFormat } = await loadPrivacyFilterModel(sd, {
    onProgress: opts.onProgress,
  });
  const tokenizer = createTokenizer();
  const info = buildLabelInfo(V2_NER_CLASS_NAMES);
  const decoder = new ViterbiDecoder(info, opts.biases ?? ZERO_BIASES);
  return {
    tokenizer,
    weightFormat,
    detect: assembleDetect({
      device,
      tokenizer,
      info,
      decoder,
      model,
    }),
    dispose: () => model.dispose(),
  };
}
