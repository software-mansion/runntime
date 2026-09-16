import { PreTrainedModel, Tensor, type PretrainedConfig } from '@huggingface/transformers';
import { defaultRoot, inGpuErrorScopes } from '../../../core/index.ts';
import type { Estimator } from '../../depthart/estimator.ts';

/** Depth-estimation stand-in running the zoo DepthART model. Takes the
 *  processor's normalized `pixel_values` [N, 3, H, W] and returns
 *  `predicted_depth` [N, S, S] at the model's own size; the stock
 *  post-processing resizes it to the image. Images run one at a time. */
export class RunntimeDepthartForDepthEstimation extends PreTrainedModel {
  constructor(
    config: PretrainedConfig,
    private readonly runntime: Estimator,
  ) {
    super(config, {}, {});
  }

  override async forward(inputs: Record<string, Tensor>): Promise<Record<string, Tensor>> {
    const pixels = inputs['pixel_values'];
    if (!pixels) throw new Error('runntime backend: depthart forward() needs pixel_values');
    if (pixels.dims.length !== 4 || pixels.dims[1] !== 3) {
      throw new Error(
        `runntime backend: depthart takes pixel_values [N, 3, H, W], got [${pixels.dims}]`,
      );
    }
    const [n, , h, w] = pixels.dims as [number, number, number, number];
    const size = this.runntime.inputSize;
    const plane = size * size;
    const out = new Float32Array(n * plane);
    const data = pixels.data as Float32Array;
    const stride = 3 * h * w;
    for (let i = 0; i < n; i++) {
      const depth = await inGpuErrorScopes(defaultRoot().device, 'inference', () =>
        this.runntime.run(data.subarray(i * stride, (i + 1) * stride), h, w),
      );
      out.set(depth, i * plane);
    }
    return { predicted_depth: new Tensor('float32', out, [n, size, size]) };
  }
}
