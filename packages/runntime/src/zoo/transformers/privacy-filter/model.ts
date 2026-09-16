import { PreTrainedModel, Tensor, type PretrainedConfig } from '@huggingface/transformers';
import { defaultRoot, inGpuErrorScopes, toArray } from '../../../core/index.ts';
import { V2_NER_CLASS_NAMES } from '../../privacy-filter/config.ts';
import type { EagerModel } from '../../privacy-filter/model.ts';

/** Labels per token: 'O' plus 8 span types × 4 boundary tags. */
export const PRIVACY_FILTER_LABELS = V2_NER_CLASS_NAMES.length;

export class RunntimePrivacyFilterForTokenClassification extends PreTrainedModel {
  constructor(
    config: PretrainedConfig,
    private readonly runntime: Pick<EagerModel, 'forward' | 'dispose'>,
  ) {
    super(config, {}, {});
  }

  /** transformers.js's release hook (`await model.dispose()`): frees the
   *  GPU weights. The ONNX-session array it normally returns is empty here. */
  override async dispose(): Promise<unknown[]> {
    this.runntime.dispose();
    return [];
  }

  override async forward(modelInputs: Record<string, Tensor>): Promise<Record<string, Tensor>> {
    const inputIds = modelInputs['input_ids'];
    if (!inputIds) throw new Error('runntime backend: model inputs are missing input_ids');
    const mask = modelInputs['attention_mask'];
    const [batch, seqLen] = inputIds.dims as [number, number];
    const idsData = inputIds.data as BigInt64Array;
    const maskData = mask?.data;

    const L = PRIVACY_FILTER_LABELS;
    const out = new Float32Array(batch * seqLen * L);
    const device = defaultRoot().device;
    await inGpuErrorScopes(device, 'inference', async () => {
      for (let b = 0; b < batch; b++) {
        // the row's real tokens and the slots they sit in
        const row: number[] = [];
        const slots: number[] = [];
        for (let t = 0; t < seqLen; t++) {
          const i = b * seqLen + t;
          if (maskData === undefined || Number(maskData[i]) !== 0) {
            row.push(Number(idsData[i]));
            slots.push(t);
          }
        }
        if (row.length === 0) continue;
        const logits = await toArray(this.runntime.forward(row));
        checkLogits(logits);
        for (const [k, t] of slots.entries()) {
          out.set(logits.subarray(k * L, (k + 1) * L), (b * seqLen + t) * L);
        }
      }
    });
    return { logits: new Tensor('float32', out, [batch, seqLen, L]) };
  }
}

function checkLogits(logits: Float32Array): void {
  if (!logits.every(Number.isFinite)) {
    throw new Error(
      'runntime backend: GPU returned non-finite logits — inference failed on this device',
    );
  }
  if (logits.every((v) => v === 0)) {
    throw new Error(
      'runntime backend: GPU returned all-zero logits — inference failed on this device',
    );
  }
}
