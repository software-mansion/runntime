/** runntime/zoo root: what a zoo user calls. Setup, load options and the
 *  input and result types every task shares. Each task runner
 *  (create<Task>) adds its export here. Building blocks under tasks/ stay
 *  internal; the package imports them by path. */

export {
  bufferSource,
  createOpfsCache,
  initRunntime,
  type RangeSource,
  type WeightCache,
} from '../core/index.ts';

export type { LoadOptions, ModelPath } from './load.ts';
export { models } from './models.ts';

export {
  createTextEmbedder,
  similarity,
  type TextEmbedder,
  type TextEmbedderModel,
} from './tasks/nlp/textEmbedding.ts';

export {
  createPrivacyFilter,
  type PrivacyFilter,
  type PrivacyFilterModel,
  type PrivacySpan,
} from './tasks/nlp/privacyFilter.ts';

export {
  createSpeechToText,
  type SpeechStreamUpdate,
  type SpeechToText,
  type SpeechToTextArch,
  type SpeechToTextModel,
} from './tasks/audio/speechToText.ts';
export { decodeAudio, resampleAudio, SPEECH_SAMPLE_RATE } from './tasks/audio/audioInput.ts';

export {
  createObjectDetector,
  type DetectObjectsOptions,
  type ObjectDetection,
  type ObjectDetector,
  type ObjectDetectorModel,
} from './tasks/cv/objectDetection.ts';
export type { Yolo26Variant } from './yolo26/config.ts';

export {
  createInstanceSegmenter,
  type InstanceSegmentation,
  type InstanceSegmenter,
  type InstanceSegmenterModel,
  type SegmentInstancesOptions,
} from './tasks/cv/instanceSegmentation.ts';

export {
  createKeypointDetector,
  type DetectKeypointsOptions,
  type KeypointDetection,
  type KeypointDetector,
  type KeypointDetectorModel,
  type Landmark,
} from './tasks/cv/keypointDetection.ts';
export { COCO_LANDMARKS, type CocoLandmark } from './yolo26/pipeline.ts';

export {
  createDepthEstimator,
  type DepthEstimator,
  type DepthEstimatorModel,
  type DepthMap,
} from './tasks/cv/depthEstimation.ts';
export type { DepthartVariant } from './depthart/estimator.ts';

export {
  createImageClassifier,
  type ClassifyOptions,
  type ImageClassifier,
  type ImageClassifierModel,
} from './tasks/cv/imageClassification.ts';
export { type Classification } from './mobilenetv4/pipeline.ts';

export {
  imageBuffer,
  imageBufferFromImageData,
  type ImageBuffer,
  type ImageFormat,
  type ResizeMode,
} from './tasks/cv/image.ts';
export type { Point, Size } from './tasks/cv/ops/point.ts';
export type { BoundingBox, BoxFormat, BoxMap } from './tasks/cv/ops/box.ts';
