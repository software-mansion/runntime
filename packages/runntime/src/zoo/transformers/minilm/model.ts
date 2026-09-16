import { PreTrainedModel, Tensor, type PretrainedConfig } from '@huggingface/transformers';
import {
  createReplayCache,
  defaultRoot,
  inGpuErrorScopes,
  toArray,
  type ReplayCache,
} from '../../../core/index.ts';
import { MINILM_L6 } from '../../minilm/config.ts';
import { fillBucketInputs, MINILM_BUCKETS } from '../../minilm/embedCore.ts';
import { MinilmModel } from '../../minilm/model.ts';

/** The three input buffers a replay rewrites on every call. */
type ReplayInput = 'ids' | 'positions' | 'segments';

function fillReplayInputs(
  T: number,
  ids: readonly number[],
  out: Record<ReplayInput, Float32Array>,
): void {
  fillBucketInputs(T, ids, { ...out, selector: new Float32Array(T) });
}

export class RunntimeBertModel extends PreTrainedModel {
  private readonly replays: ReplayCache<ReplayInput>;

  constructor(
    config: PretrainedConfig,
    private readonly runntime: MinilmModel,
  ) {
    super(config, {}, {});
    this.replays = createReplayCache({
      buckets: MINILM_BUCKETS,
      inputs: (T) => ({ ids: [T, 1], positions: [T, 1], segments: [T, 2] }),
      captureFill: (T, scratch) => fillReplayInputs(T, new Array<number>(T).fill(0), scratch),
      // The replay ends at the hidden states; the pipeline does the pooling.
      build: (T, inp) =>
        this.runntime.forward(inp.ids, {
          positions: inp.positions,
          segments: inp.segments,
          maxSegment: T,
        }),
    });
  }

  /** transformers.js's release hook: frees the recorded replays and the
   *  GPU weights. Returns the (empty) ONNX-session list callers may await. */
  override async dispose(): Promise<unknown[]> {
    this.replays.dispose();
    this.runntime.dispose();
    return [];
  }

  warm(): Promise<void> {
    return this.replays.warmAll();
  }

  override async forward(modelInputs: Record<string, Tensor>): Promise<Record<string, Tensor>> {
    const inputIds = modelInputs['input_ids'];
    if (!inputIds) throw new Error('runntime backend: model inputs are missing input_ids');
    const mask = modelInputs['attention_mask'];
    const [batch, seqLen] = inputIds.dims as [number, number];
    const idsData = inputIds.data as BigInt64Array;
    const maskData = mask?.data;

    // slots remember where each kept token sits in its row, so writeRow
    // works for left and right padding alike
    const rows: number[][] = [];
    const slots: number[][] = [];
    for (let b = 0; b < batch; b++) {
      const row: number[] = [];
      const slot: number[] = [];
      for (let t = 0; t < seqLen; t++) {
        const i = b * seqLen + t;
        if (maskData === undefined || Number(maskData[i]) !== 0) {
          row.push(Number(idsData[i]));
          slot.push(t);
        }
      }
      rows.push(row);
      slots.push(slot);
    }

    const hidden = MINILM_L6.hidden;
    const out = new Float32Array(batch * seqLen * hidden);
    // copies row b's computed hidden states into the output rectangle,
    // each one at its token's original slot
    const writeRow = (flat: Float32Array, srcRow: number, b: number) => {
      for (const [k, t] of slots[b]!.entries()) {
        out.set(
          flat.subarray((srcRow + k) * hidden, (srcRow + k + 1) * hidden),
          (b * seqLen + t) * hidden,
        );
      }
    };

    const bucket = batch === 1 ? this.replays.pick(rows[0]!.length) : undefined;
    if (bucket !== undefined) {
      const row = rows[0]!;
      // No error scopes here: each costs a GPU round-trip, and a replay
      // resubmits work already validated in warm().
      const flat = await this.replays.run(bucket, (T, scratch) =>
        fillReplayInputs(T, row, scratch),
      );
      writeRow(flat, 0, 0);
      return { last_hidden_state: new Tensor('float32', out, [batch, seqLen, hidden]) };
    }

    const packed = rows.flat();
    const lengths = rows.map((r) => r.length);
    const device = defaultRoot().device;
    const flat = await inGpuErrorScopes(device, 'inference', () =>
      toArray(this.runntime.forward(packed, batch > 1 ? lengths : undefined)),
    );

    let srcRow = 0;
    rows.forEach((row, b) => {
      writeRow(flat, srcRow, b);
      srcRow += row.length;
    });
    return { last_hidden_state: new Tensor('float32', out, [batch, seqLen, hidden]) };
  }
}
