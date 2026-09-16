import { PreTrainedModel, Tensor, type PretrainedConfig } from '@huggingface/transformers';
import { defaultRoot, inGpuErrorScopes } from '../../../core/index.ts';
import type { DetectorOutput } from '../../yolo26/detector.ts';
import { selectTopAnchors } from './output.ts';

/** The GPU side of the stand-in: one image in, the raw detect levels out. */
export interface Yolo26Detector {
  inputSize: number;
  numClasses: number;
  maxDet: number;
  run(pixels: Float32Array): Promise<DetectorOutput>;
  /** Frees the model's weights and the input buffer. Every later run()
   *  rejects. The GPU keeps them until this is called: buffers the executor
   *  has bound once are never garbage-collected. */
  dispose(): void;
}

export class RunntimeYolosForObjectDetection extends PreTrainedModel {
  constructor(
    config: PretrainedConfig,
    private readonly runntime: Yolo26Detector,
  ) {
    super(config, {}, {});
  }

  /** transformers.js's release hook (`await model.dispose()`): frees the
   *  GPU weights. The ONNX-session array it normally returns is empty here. */
  override async dispose(): Promise<unknown[]> {
    this.runntime.dispose();
    return [];
  }

  override async forward(inputs: Record<string, Tensor>): Promise<Record<string, Tensor>> {
    const pixels = inputs['pixel_values'];
    if (!pixels) throw new Error('runntime backend: yolo26 forward() needs pixel_values');
    const { inputSize: size, numClasses } = this.runntime;
    const [n, c, h, w] = pixels.dims;
    if (n !== 1 || c !== 3 || h !== size || w !== size) {
      throw new Error(
        `runntime backend: yolo26 takes one ${size}×${size} RGB image per call, got pixel_values [${pixels.dims}]`,
      );
    }
    const { levels } = await inGpuErrorScopes(defaultRoot().device, 'inference', () =>
      this.runntime.run(pixels.data as Float32Array),
    );
    const top = selectTopAnchors(levels, {
      numClasses,
      maxDet: this.runntime.maxDet,
      inputSize: size,
    });
    return {
      logits: new Tensor('float32', top.logits, [1, top.count, numClasses]),
      pred_boxes: new Tensor('float32', top.boxes, [1, top.count, 4]),
    };
  }
}
