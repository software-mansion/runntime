import { PreTrainedModel, Tensor, type PretrainedConfig } from '@huggingface/transformers';
import { defaultRoot, inGpuErrorScopes } from '../../../core/index.ts';
import type { Classifier } from '../../mobilenetv4/classifier.ts';

export class RunntimeMobileNetV4ForImageClassification extends PreTrainedModel {
  constructor(
    config: PretrainedConfig,
    private readonly runntime: Classifier,
  ) {
    super(config, {}, {});
  }

  /** transformers.js's release hook; the ONNX-session array it normally
   *  returns is empty here. */
  override async dispose(): Promise<unknown[]> {
    this.runntime.dispose();
    return [];
  }

  override async forward(inputs: Record<string, Tensor>): Promise<Record<string, Tensor>> {
    const pixels = inputs['pixel_values'];
    if (!pixels) throw new Error('runntime backend: mobilenetv4 forward() needs pixel_values');
    const { inputSize: size, numClasses } = this.runntime;
    if (pixels.dims.length !== 4 || pixels.dims[1] !== 3) {
      throw new Error(
        `runntime backend: mobilenetv4 takes pixel_values [N, 3, ${size}, ${size}], got [${pixels.dims}]`,
      );
    }
    const [n, , h, w] = pixels.dims as [number, number, number, number];
    if (h !== size || w !== size) {
      throw new Error(
        `runntime backend: mobilenetv4 takes ${size}×${size} images, got pixel_values [${pixels.dims}]`,
      );
    }
    const out = new Float32Array(n * numClasses);
    const data = pixels.data as Float32Array;
    const stride = 3 * size * size;
    for (let i = 0; i < n; i++) {
      const logits = await inGpuErrorScopes(defaultRoot().device, 'inference', () =>
        this.runntime.run(data.subarray(i * stride, (i + 1) * stride)),
      );
      out.set(logits, i * numClasses);
    }
    return { logits: new Tensor('float32', out, [n, numClasses]) };
  }
}
