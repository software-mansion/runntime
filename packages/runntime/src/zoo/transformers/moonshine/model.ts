import { PreTrainedModel, Tensor, type PretrainedConfig } from '@huggingface/transformers';
import { defaultRoot, inGpuErrorScopes } from '../../../core/index.ts';
import { MIN_SAMPLES } from '../../moonshine/convStem.ts';
import type { Transcriber } from '../../moonshine/transcriber.ts';

/** Speech-to-text stand-in running the zoo moonshine transcriber. The
 *  pipeline hands over 16 kHz audio as `input_values` [1, samples] and
 *  gets back the generated ids as int64 [1, L], BOS first, EOS last when
 *  reached; it decodes the text with the hub tokenizer. The transcriber
 *  does everything from audio to ids: encoder, greedy burst decode, and
 *  splitting clips longer than its chunk limit. */
export class RunntimeMoonshineForConditionalGeneration extends PreTrainedModel {
  /** Encode and decode wall time of the last generate() call, for
   *  benchmarks; the pipeline itself never reads it. */
  lastTimings: Awaited<ReturnType<Transcriber['transcribe']>>['timings'] | undefined;

  constructor(
    config: PretrainedConfig,
    private readonly runntime: Pick<Transcriber, 'transcribe' | 'dispose'>,
    private readonly tokens: { bos: number; eos: number },
  ) {
    super(config, {}, {});
  }

  /** transformers.js's release hook: frees the weights, KV caches and rope
   *  tables. Returns the (empty) ONNX-session list callers may await. */
  override async dispose(): Promise<unknown[]> {
    this.runntime.dispose();
    return [];
  }

  override async generate(options: {
    inputs?: Tensor | null;
    [key: string]: unknown;
  }): Promise<Tensor> {
    const audio = (options['input_values'] as Tensor | undefined) ?? options.inputs;
    if (!audio) throw new Error('runntime backend: moonshine generate() needs input_values');
    if (audio.dims.length !== 2 || audio.dims[0] !== 1) {
      throw new Error(
        `runntime backend: moonshine takes one clip per call, got input_values [${audio.dims}]`,
      );
    }
    const samples = audio.data as Float32Array;
    // Too short for one encoder frame: the empty transcript.
    let ids = [this.tokens.bos, this.tokens.eos];
    if (samples.length >= MIN_SAMPLES) {
      const result = await inGpuErrorScopes(defaultRoot().device, 'inference', () =>
        this.runntime.transcribe(samples),
      );
      ids = result.ids;
      this.lastTimings = result.timings;
    }
    return new Tensor('int64', BigInt64Array.from(ids.map(BigInt)), [1, ids.length]);
  }
}
