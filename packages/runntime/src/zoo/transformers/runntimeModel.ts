import { RunntimeBertModel } from './minilm/model.ts';
import { RunntimePrivacyFilterForTokenClassification } from './privacy-filter/model.ts';
import { RunntimeMoonshineForConditionalGeneration } from './moonshine/model.ts';
import { RunntimeYolosForObjectDetection } from './yolo26/model.ts';
import { RunntimeDepthartForDepthEstimation } from './depthart/model.ts';
import { RunntimeMobileNetV4ForImageClassification } from './mobilenetv4/model.ts';

/** True when a model is one of the plugin's TypeGPU ones, not the ONNX
 *  fallback. */
export function isRunntimeModel(model: unknown): boolean {
  return (
    model instanceof RunntimeBertModel ||
    model instanceof RunntimePrivacyFilterForTokenClassification ||
    model instanceof RunntimeMoonshineForConditionalGeneration ||
    model instanceof RunntimeYolosForObjectDetection ||
    model instanceof RunntimeDepthartForDepthEstimation ||
    model instanceof RunntimeMobileNetV4ForImageClassification
  );
}
